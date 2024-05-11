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