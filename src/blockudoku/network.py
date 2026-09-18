"""Equinox network for Rainbow: a vision transformer with noisy dueling C51 heads.

    net = RainbowNet(load_config("full").net, key=key)
    logits = net(obs, key=noise_key)   # (243, atoms); key=None -> noise-free (eval/export)

Single-example layout without a batch dim (vmap for batches). Every op here has a
1:1 ONNX counterpart in `export_onnx.py`; keep them in sync (tests/test_export.py).

Tokens (T = 81 + 3 + R):
    81 cell tokens   Dense(4 -> D) of [filled, legal-anchor slot0..2] + learned 3x3-box embedding
     3 piece tokens  Dense(25 -> D) of the 5x5 bitmap + learned slot embedding
     R registers     learned "memory" tokens (Darcet et al., 2024), read by the value head
Trunk: `depth` pre-LayerNorm blocks (MHA + GELU MLP). Attention uses 2D axial RoPE
on cell tokens: the first half of every head rotates with the row index, the
second half with the column index; piece and register tokens are not rotated.
RoPE only encodes relative offsets, so the absolute 3x3-box structure comes from
the box embedding.
Heads (noisy nets, Fortunato et al., 2018):
    advantage  per cell token: NoisyLinear(D->Ha) relu NoisyLinear(Ha->3*atoms)
               -> action (slot, r, c) is read from cell (r, c)
    value      concat(registers): NoisyLinear(R*D->Hv) relu NoisyLinear(Hv->atoms)
    logits = V + A - mean_a(A)   over atoms of a log-spaced return support (value_support)
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from blockudoku.config import NetConfig
from blockudoku.env import BOARD_SIZE, NUM_ACTIONS, NUM_CELLS, NUM_PLANES, NUM_SLOTS, Obs
from blockudoku.pieces import PIECE_SIZE

PIECE_PIXELS = PIECE_SIZE * PIECE_SIZE
LN_EPS = 1e-5
_rows, _cols = np.divmod(np.arange(NUM_CELLS), BOARD_SIZE)
BOX_OF_CELL = (_rows // 3) * 3 + _cols // 3  # (81,)


# --------------------------------------------------------------------------- support
def symlog(x):
    return np.sign(x) * np.log1p(np.abs(x))


def symexp(x):
    return np.sign(x) * np.expm1(np.abs(x))


def value_support(v_min: float, v_max: float, num_atoms: int, scale: str) -> np.ndarray:
    """Atom locations. "symlog": uniform in symlog space, i.e. logarithmically spaced
    returns: fine resolution near 0, coarse for large returns."""
    if scale == "linear":
        return np.linspace(v_min, v_max, num_atoms, dtype=np.float32)
    return symexp(np.linspace(symlog(v_min), symlog(v_max), num_atoms)).astype(np.float32)


# --------------------------------------------------------------------------- RoPE
def rope_tables(num_tokens: int, head_dim: int, base: float):
    """2D axial RoPE constants.

    Returns cos, sin of shape (T, head_dim) and a (head_dim, head_dim) matrix P such
    that rope(x) = x * cos + (x @ P) * sin. Each axis (row, col) owns half of the
    head; within an axis half we use the rotate-half pairing (i, i + quarter).
    Tokens >= 81 (pieces, registers) get angle 0, i.e. no rotation.
    """
    assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
    half, quarter = head_dim // 2, head_dim // 4
    freqs = base ** (-np.arange(quarter) / quarter)  # (quarter,)
    angles = np.zeros((num_tokens, head_dim))
    for axis, pos in enumerate((_rows, _cols)):
        a = pos[:, None] * freqs[None, :]  # (81, quarter)
        angles[:NUM_CELLS, axis * half : (axis + 1) * half] = np.concatenate([a, a], axis=1)
    rot = np.zeros((head_dim, head_dim))
    for axis in range(2):
        o = axis * half
        for i in range(quarter):
            rot[o + quarter + i, o + i] = -1.0  # out[i] = -x[i + quarter]
            rot[o + i, o + quarter + i] = 1.0  # out[i + quarter] = x[i]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32), rot.astype(np.float32)


# --------------------------------------------------------------------------- layers
class Dense(eqx.Module):
    weight: jax.Array  # (out, in)
    bias: jax.Array

    def __init__(self, d_in: int, d_out: int, *, key, scale: float = 1.0):
        self.weight = jax.random.normal(key, (d_out, d_in)) * (scale / math.sqrt(d_in))
        self.bias = jnp.zeros((d_out,))

    def __call__(self, x):
        return x @ self.weight.T.astype(x.dtype) + self.bias.astype(x.dtype)


class LayerNorm(eqx.Module):
    weight: jax.Array
    bias: jax.Array

    def __init__(self, dim: int):
        self.weight = jnp.ones((dim,))
        self.bias = jnp.zeros((dim,))

    def __call__(self, x):
        x32 = x.astype(jnp.float32)  # statistics in float32 even under bf16 compute
        mean = x32.mean(-1, keepdims=True)
        var = ((x32 - mean) ** 2).mean(-1, keepdims=True)
        y = (x32 - mean) / jnp.sqrt(var + LN_EPS) * self.weight + self.bias
        return y.astype(x.dtype)


class NoisyLinear(eqx.Module):
    """Factorised-Gaussian noisy linear layer (Fortunato et al., 2018)."""

    weight_mu: jax.Array
    weight_sigma: jax.Array
    bias_mu: jax.Array
    bias_sigma: jax.Array

    def __init__(self, in_features: int, out_features: int, sigma0: float, *, key):
        bound = 1.0 / math.sqrt(in_features)
        k1, k2 = jax.random.split(key)
        self.weight_mu = jax.random.uniform(k1, (out_features, in_features), minval=-bound, maxval=bound)
        self.bias_mu = jax.random.uniform(k2, (out_features,), minval=-bound, maxval=bound)
        self.weight_sigma = jnp.full((out_features, in_features), sigma0 * bound)
        self.bias_sigma = jnp.full((out_features,), sigma0 * bound)

    def params(self, key: jax.Array | None) -> tuple[jax.Array, jax.Array]:
        """(weight, bias); noise-free means when key is None."""
        if key is None:
            return self.weight_mu, self.bias_mu

        def f(x):
            return jnp.sign(x) * jnp.sqrt(jnp.abs(x))

        k_in, k_out = jax.random.split(key)
        dtype = self.weight_mu.dtype
        eps_in = f(jax.random.normal(k_in, (self.weight_mu.shape[1],), dtype))
        eps_out = f(jax.random.normal(k_out, (self.weight_mu.shape[0],), dtype))
        w = self.weight_mu + self.weight_sigma * jnp.outer(eps_out, eps_in)
        b = self.bias_mu + self.bias_sigma * eps_out
        return w, b

    def __call__(self, x: jax.Array, *, key: jax.Array | None = None) -> jax.Array:
        """x: (..., in) -> (..., out). The same noise sample is shared across leading dims."""
        w, b = self.params(key)
        return x @ w.T.astype(x.dtype) + b.astype(x.dtype)


class Attention(eqx.Module):
    q: Dense
    k: Dense
    v: Dense
    o: Dense
    heads: int = eqx.field(static=True)

    def __init__(self, dim: int, heads: int, out_scale: float, *, key):
        kq, kk, kv, ko = jax.random.split(key, 4)
        self.q, self.k, self.v = Dense(dim, dim, key=kq), Dense(dim, dim, key=kk), Dense(dim, dim, key=kv)
        self.o = Dense(dim, dim, key=ko, scale=out_scale)
        self.heads = heads

    def __call__(self, x, cos, sin, rot):
        T, D = x.shape
        hd = D // self.heads

        def split(y):  # (T, D) -> (H, T, hd)
            return y.reshape(T, self.heads, hd).transpose(1, 0, 2)

        def rope(y):
            return y * cos + (y @ rot) * sin

        q, k, v = rope(split(self.q(x))), rope(split(self.k(x))), split(self.v(x))
        scores = (q @ k.transpose(0, 2, 1)).astype(jnp.float32) / math.sqrt(hd)
        attn = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
        out = (attn @ v).transpose(1, 0, 2).reshape(T, D)
        return self.o(out)


class Block(eqx.Module):
    ln1: LayerNorm
    attn: Attention
    ln2: LayerNorm
    fc1: Dense
    fc2: Dense

    def __init__(self, dim: int, heads: int, mlp_ratio: int, out_scale: float, *, key):
        ka, k1, k2 = jax.random.split(key, 3)
        self.ln1, self.ln2 = LayerNorm(dim), LayerNorm(dim)
        self.attn = Attention(dim, heads, out_scale, key=ka)
        self.fc1 = Dense(dim, mlp_ratio * dim, key=k1)
        self.fc2 = Dense(mlp_ratio * dim, dim, key=k2, scale=out_scale)

    def __call__(self, x, cos, sin, rot):
        x = x + self.attn(self.ln1(x), cos, sin, rot)
        return x + self.fc2(jax.nn.gelu(self.fc1(self.ln2(x)), approximate=False))


# --------------------------------------------------------------------------- model
class RainbowNet(eqx.Module):
    cell_embed: Dense
    box_embed: jax.Array  # (9, D)
    piece_embed: Dense
    slot_embed: jax.Array  # (3, D)
    registers: jax.Array  # (R, D)
    blocks: tuple[Block, ...]
    ln_f: LayerNorm
    adv_hidden: NoisyLinear
    adv_out: NoisyLinear
    value_hidden: NoisyLinear
    value_out: NoisyLinear
    num_atoms: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    rope_base: float = eqx.field(static=True)

    def __init__(self, cfg: NetConfig, *, key):
        if cfg.dim % cfg.heads or (cfg.dim // cfg.heads) % 4:
            raise ValueError("net.dim / net.heads must be an integer divisible by 4 (2D RoPE)")
        keys = iter(jax.random.split(key, 10 + cfg.depth))
        d, s0 = cfg.dim, cfg.noisy_sigma0
        self.num_atoms, self.heads, self.rope_base = cfg.num_atoms, cfg.heads, cfg.rope_base
        self.cell_embed = Dense(NUM_PLANES, d, key=next(keys))
        self.box_embed = 0.02 * jax.random.normal(next(keys), (9, d))
        self.piece_embed = Dense(PIECE_PIXELS, d, key=next(keys))
        self.slot_embed = 0.02 * jax.random.normal(next(keys), (NUM_SLOTS, d))
        self.registers = 0.02 * jax.random.normal(next(keys), (cfg.num_registers, d))
        out_scale = 1.0 / math.sqrt(2 * cfg.depth)  # keep the residual stream O(1) at init
        self.blocks = tuple(Block(d, cfg.heads, cfg.mlp_ratio, out_scale, key=next(keys))
                            for _ in range(cfg.depth))
        self.ln_f = LayerNorm(d)
        self.adv_hidden = NoisyLinear(d, cfg.adv_hidden, s0, key=next(keys))
        self.adv_out = NoisyLinear(cfg.adv_hidden, NUM_SLOTS * cfg.num_atoms, s0, key=next(keys))
        self.value_hidden = NoisyLinear(cfg.num_registers * d, cfg.value_hidden, s0, key=next(keys))
        self.value_out = NoisyLinear(cfg.value_hidden, cfg.num_atoms, s0, key=next(keys))

    @property
    def num_tokens(self) -> int:
        return NUM_CELLS + NUM_SLOTS + self.registers.shape[0]

    def tokens(self, obs: Obs) -> jax.Array:
        dtype = self.cell_embed.weight.dtype
        cells = obs.planes.astype(dtype).reshape(NUM_PLANES, NUM_CELLS).T  # (81, 4)
        cells = self.cell_embed(cells) + self.box_embed[BOX_OF_CELL]
        pieces = obs.pieces.astype(dtype).reshape(NUM_SLOTS, PIECE_PIXELS)
        pieces = self.piece_embed(pieces) + self.slot_embed
        return jnp.concatenate([cells, pieces, self.registers])  # (T, D)

    def __call__(self, obs: Obs, *, key: jax.Array | None = None) -> jax.Array:
        """Returns atom logits of shape (243, num_atoms) in float32."""
        ks = [None] * 4 if key is None else list(jax.random.split(key, 4))
        n = self.num_atoms
        x = self.tokens(obs)
        cos, sin, rot = (jnp.asarray(t, x.dtype) for t in
                         rope_tables(self.num_tokens, x.shape[1] // self.heads, self.rope_base))
        for block in self.blocks:
            x = block(x, cos, sin, rot)
        x = self.ln_f(x)

        a = jax.nn.relu(self.adv_hidden(x[:NUM_CELLS], key=ks[0]))
        a = self.adv_out(a, key=ks[1])  # (81, 3*atoms), channel = slot*atoms + atom
        a = a.reshape(NUM_CELLS, NUM_SLOTS, n).transpose(1, 0, 2).reshape(NUM_ACTIONS, n)

        v = x[NUM_CELLS + NUM_SLOTS :].reshape(-1)  # registers, flattened token-major
        v = jax.nn.relu(self.value_hidden(v, key=ks[2]))
        v = self.value_out(v, key=ks[3])  # (atoms,)

        return (v[None, :] + a - a.mean(axis=0, keepdims=True)).astype(jnp.float32)


def cast_floats(tree, dtype):
    """Cast every floating array leaf (mixed-precision compute copy of the params)."""
    return jax.tree.map(lambda x: x.astype(dtype) if eqx.is_inexact_array(x) else x, tree)


def q_values(logits: jax.Array, support: jax.Array) -> jax.Array:
    """Expected return per action from atom logits (..., atoms) -> (...)."""
    return (jax.nn.softmax(logits, axis=-1) * support).sum(axis=-1)


def masked_argmax(q: jax.Array, mask: jax.Array) -> jax.Array:
    return jnp.argmax(jnp.where(mask, q, -jnp.inf), axis=-1)


def num_params(model: eqx.Module) -> int:
    return sum(x.size for x in jax.tree.leaves(eqx.filter(model, eqx.is_array)))
