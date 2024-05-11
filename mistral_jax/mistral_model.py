import jax
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu

import equinox as eqx
from functools import partial
from equinox._misc import default_floating_dtype
from jaxtyping import Array, Float

from rope import calculate_rope
from rope import precompute_frequencies


# We are not using `eqx.RMSNorm` because there is a bug
# in the calculation. Will send a PR for that separately!
class RMSNorm(eqx.Module):
    eps: float
    weight: Float[Array, "*shape"]

    def __init__(self, dim, eps, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x **2 , keepdims=True) + self.eps)

    def __call__(self, x):
        output = self._norm(x.astype(jnp.float32)).astype(x.dtype)
        return output * self.weight
    

class FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        key1, key2, key3 = jax.random.split(key, 3)

        self.w1 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key1, dtype=dtype)
        self.w2 = eqx.nn.Linear(args.hidden_dim, args.dim, use_bias=False, key=key2, dtype=dtype)
        self.w3 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key3, dtype=dtype)

    def __call__(self, x):
        h = jax.nn.silu(self.w1(x).astype(jnp.float32)).astype(x.dtype)
        return self.w2(h * self.w3(x))
    

class Attention(eqx.Module):
    dim: int
    n_heads: int
    head_dim: int
    n_kv_heads: int
    kv_repeats: int
    sliding_window: int
    scale: float
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(self, args, key, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        key1, key2, key3, key4 = jax.random.split(key, 4)

        self.n_heads = args.n_heads
        self.head_dim = args.head_dim
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window

        self.scale = args.head_dim**-0.5

        self.wq = eqx.nn.Linear(args.dim, args.n_heads * args.head_dim, use_bias=False, key=key1, dtype=dtype)
        self.wk = eqx.nn.Linear(args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key2, dtype=dtype)
        self.wv = eqx.nn.Linear(args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key3, dtype=dtype)
        self.wo = eqx.nn.Linear(args.n_heads * args.head_dim, args.dim, use_bias=False, key=key4, dtype=dtype)

    @partial(jax.jit, static_argnums=(2, 3))
    def get_cache_slice(self, x, pos, kv_repeats):
        x_slice = x.at[:pos, :, :].get()
        x_slice = jnp.repeat(x_slice, kv_repeats, axis=1)
        return x_slice

    @eqx.filter_jit
    def compute_qkv(self, x):
        seqlen, _ = x.shape
        
        xq = jax.vmap(self.wq)(x)
        xk = jax.vmap(self.wk)(x)
        xv = jax.vmap(self.wv)(x)

        xq = jnp.reshape(xq, (seqlen, self.n_heads, self.head_dim))
        xk = jnp.reshape(xk, (seqlen, self.n_kv_heads, self.head_dim))
        xv = jnp.reshape(xv, (seqlen, self.n_kv_heads, self.head_dim))
        return xq, xk, xv

    @jax.jit
    def update_cache_values(self, xk, xv, cache_k, cache_v, positions):
        cache_k = cache_k.at[positions, ...].set(xk[positions, ...])
        cache_v = cache_v.at[positions, ...].set(xv[positions, ...])
        return cache_k, cache_v

    @eqx.filter_jit
    def prefill(self, xk, xv):
        key = jnp.repeat(xk, self.kv_repeats, axis=1)
        value = jnp.repeat(xv, self.kv_repeats, axis=1)
        return key, value

    @eqx.filter_jit
    def compute_scores_and_output(self, xq, key, value, mask, seqlen):
        query = jnp.transpose(xq, (1, 0, 2))
        key = jnp.transpose(key, (1, 0, 2))
        value = jnp.transpose(value, (1, 0, 2))

        # # # scores : [n_heads, seqlen | 1, seqlen]
        scores = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))) * self.scale

        if mask is not None:
            # Mask will of shape [seqlen, seqlen] but our scores
            # have shape [num_heads, seqlen, seqlen], hence we need
            # to introduce another dimension in the mask
            mask = mask[jnp.newaxis, ...]
            scores = scores + mask

        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        output = jnp.matmul(scores, value)
        output = jnp.reshape(jnp.transpose(output, (1, 0, 2)), (seqlen, -1))
        output = jax.vmap(self.wo)(output)
        return output

    def __call__(self,  x, cos_freq, sin_freq, positions, mask=None, cache_k=None, cache_v=None):
        # x shape: [seqlen, embed_dim]
        seqlen, _ = x.shape
        # 1. Calculate qkv
        xq, xk, xv = self.compute_qkv(x)

        # 2. Calculate RoPE
        xq = calculate_rope(xq, cos_freq, sin_freq, 0)
        xk = calculate_rope(xk, cos_freq, sin_freq, 0)

        # 3. Update cache
        cache_k, cache_v = self.update_cache_values(xk, xv, cache_k, cache_v, positions)

        # 4. Generation
        if positions.shape[0] > 1:
            # prefill
            key, value = self.prefill(xk, xv)
        else:
            # single-token generation
            cur_pos = positions[-1].item() + 1
            key = self.get_cache_slice(cache_k, cur_pos, self.kv_repeats)
            value = self.get_cache_slice(cache_v, cur_pos, self.kv_repeats)

        # 5. Output
        output = self.compute_scores_and_output(xq, key, value, mask, seqlen)
        return output, cache_k, cache_v