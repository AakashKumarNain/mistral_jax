import jax
import jax.numpy as jnp
# from functools import partial


def precompute_frequencies(dim, max_pos, theta=10000.0):
    """Compute frequencies for a given range."""
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    t = jnp.arange(0, max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def calculate_rope(x, cos_freq, sin_freq):
    """Generates Rotary Positional Embeddings (RoPE).

    Args:
        x: Input tensor with shape `[seqlen, num_heads, heads_dim]`
        cos_freq: Cosine frequencies for some position/s.
        sin_freq: Sine frequencies for some position/s.
    Returns:
        Rotary Positional Embeddings with same dtype as the input.
    """

    # x shape  is [seqlen, num_heads, heads_dim]
    # sin_freq, and cos_freq have the same seqlen as x

    # Positional embeddings are 2D while our input is 3D
    # if `num_heads` dimension is present in the inputs.
    # We need to add another dimension to our positional embeddings
    sin = jax.lax.expand_dims(sin_freq, (1,))
    cos = jax.lax.expand_dims(cos_freq, (1,))

    # Get the even-odd positions from the inputs
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]

    # Matmul with the rotation matrix
    # [cos_nθ, -sin_nθ] [x1]
    # [sin_nθ,  cos_nθ] [x2]
    # => [x1 * cos_nθ - x2 * sin_nθ, x1 * sin_nθ + x2 * cos_nθ]
    pos_embed = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    pos_embed = jax.lax.collapse(pos_embed, -2)
    return pos_embed.astype(x.dtype)
