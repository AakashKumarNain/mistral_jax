import gc
import jax
import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from collections import namedtuple


# Utility to convert dtypes
def to_dtype(model, dtype):
    def _to_dtype(leaf):
        if isinstance(
            leaf, jax.Array
        ):  # not eqx.is_array, which also detects NumPy arrays
            leaf_with_dtype = leaf.astype(dtype)
            del leaf
            gc.collect()  # just in case?
            return leaf_with_dtype
        else:
            return leaf

    return jtu.tree_map(_to_dtype, model)


def count_jax_parameters(model):
    return sum(x.size for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array)))


# 1. RoPE
def precompute_frequencies(dim, max_pos, theta=10000.0):
    inv_freq = 1.0 / (
        theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32)[: (dim // 2)] / dim)
    )
    t = jnp.arange(0, max_pos, dtype=jnp.float32)
    freqs = jnp.outer(t, inv_freq)
    return jnp.cos(freqs), jnp.sin(freqs)


def calculate_rope(x, cos_freq, sin_freq, offset=0):
    # x shape  is [seqlen, head_dim, num_heads]

    # Get the sequence length
    seqlen = x.shape[0]

    # Get the corresponding positional embeddings
    sin = sin_freq[offset : offset + seqlen, :]
    cos = cos_freq[offset : offset + seqlen, :]

    # Positional embeddings are 2D while our input is 3D
    # if `num_heads` dimension is present in the inputs.
    # We need to add another dimension to our positional embeddings
    sin = sin[:, jnp.newaxis, :]
    cos = cos[:, jnp.newaxis, :]

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


# 2. Attention layer
class Attention(eqx.Module):
    n_heads: int
    n_kv_heads: int
    sliding_window: int
    scale: float
    kv_repeats: int
    head_dim: int
    wq: eqx.nn.Linear
    wk: eqx.nn.Linear
    wv: eqx.nn.Linear
    wo: eqx.nn.Linear

    def __init__(self, args: namedtuple, key: jax.Array):
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.kv_repeats = self.n_heads // self.n_kv_heads
        self.sliding_window = args.sliding_window
        self.scale = args.head_dim**-0.5
        self.head_dim = args.head_dim

        key1, key2, key3, key4 = jax.random.split(key, 4)
        self.wq = eqx.nn.Linear(
            args.dim, args.n_heads * args.head_dim, use_bias=False, key=key1
        )
        self.wk = eqx.nn.Linear(
            args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key2
        )
        self.wv = eqx.nn.Linear(
            args.dim, args.n_kv_heads * args.head_dim, use_bias=False, key=key3
        )
        self.wo = eqx.nn.Linear(
            args.n_heads * args.head_dim, args.dim, use_bias=False, key=key4
        )

    def __call__(self, x, cos_freq, sin_freq, positions, mask):
        seqlen = x.shape[0]

        xq = jax.vmap(self.wq)(x)
        xk = jax.vmap(self.wk)(x)
        xv = jax.vmap(self.wv)(x)

        xq = jnp.reshape(xq, (seqlen, self.n_heads, self.head_dim))
        xk = jnp.reshape(xk, (seqlen, self.n_kv_heads, self.head_dim))
        xv = jnp.reshape(xv, (seqlen, self.n_kv_heads, self.head_dim))

        xq = calculate_rope(xq, cos_freq, sin_freq, 0)
        xk = calculate_rope(xk, cos_freq, sin_freq, 0)

        if positions.shape[0] > 1:
            # prefill
            key = jnp.repeat(xk, self.kv_repeats, axis=1)
            value = jnp.repeat(xv, self.kv_repeats, axis=1)
        # TODO: else fill from cache

        query = jnp.transpose(
            xq, (1, 0, 2)
        )  # [seqlen, num_heads, head_dim] -> [num_heads, seqlen, head_dim]
        key = jnp.transpose(
            key, (1, 0, 2)
        )  # [seqlen, num_heads, head_dim] -> [num_heads, seqlen, head_dim]
        value = jnp.transpose(
            value, (1, 0, 2)
        )  # [seqlen, num_heads, head_dim] -> [num_heads, seqlen, head_dim]

        # scores : [n_heads, seqlen | 1, seqlen]
        scores = jnp.matmul(query, jnp.transpose(key, (0, 2, 1))) * self.scale

        if mask is not None:
            # Mask will of shape [seqlen, seqlen] but our scores
            # have shape [num_heads, seqlen, seqlen], hence we need
            # to introduce another dimension in the mask
            mask = mask[jnp.newaxis, ...]
            scores = scores + mask

        scores = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(query.dtype)
        output = jnp.matmul(scores, value)
        output = jnp.transpose(output, (0, 2, 1))
        output = jnp.reshape(output, (output.shape[-1], -1))
        output = jax.vmap(self.wo)(output)
        return output


# 3. FeedForward
class FeedForward(eqx.Module):
    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, args, key):
        super().__init__()
        key1, key2, key3 = jax.random.split(key, 3)

        self.w1 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key1)
        self.w2 = eqx.nn.Linear(args.hidden_dim, args.dim, use_bias=False, key=key2)
        self.w3 = eqx.nn.Linear(args.dim, args.hidden_dim, use_bias=False, key=key3)

    def __call__(self, x):
        return self.w2(jax.nn.silu(self.w1(x)) * self.w3(x))


# 4. TransformerBlock
class TransformerBlock(eqx.Module):
    dim: int
    n_heads: int
    attention: Attention
    attention_norm: eqx.nn.RMSNorm
    feed_forward: FeedForward
    ffn_norm: eqx.nn.RMSNorm

    def __init__(self, args, key):
        key1, key2 = jax.random.split(key, 2)
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = Attention(args, key=key1)
        self.attention_norm = eqx.nn.RMSNorm(
            args.dim, eps=args.norm_eps, use_bias=False, use_weight=True
        )

        self.feed_forward = FeedForward(args, key=key2)
        self.ffn_norm = eqx.nn.RMSNorm(
            args.dim, eps=args.norm_eps, use_bias=False, use_weight=True
        )

    def __call__(self, x, cos_freq, sin_freq, positions, mask):
        normed_x = jax.vmap(self.attention_norm)(x.astype(jnp.float32)).astype(
            jnp.float16
        )
        r = self.attention(normed_x, cos_freq, sin_freq, positions, mask)
        h1 = x + r
        h2 = jax.vmap(self.ffn_norm)(h1.astype(jnp.float32)).astype(jnp.float16)
        h2 = jax.vmap(self.feed_forward)(h2)
        out = h1 + h2
        return out


# 5. Transformer
class Transformer(eqx.Module):
    tok_embeddings: eqx.nn.Embedding
    layers: TransformerBlock
    norm: eqx.nn.RMSNorm
    output: eqx.nn.Linear
    vocab_size: int
    n_layers: int

    def __init__(self, args, key):
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        keys = jax.random.split(key, args.n_layers + 2)
        embed_key, linear_key, tf_layers_keys = keys[0], keys[-1], keys[1:-1]

        self.tok_embeddings = eqx.nn.Embedding(args.vocab_size, args.dim, key=embed_key)
        self.norm = eqx.nn.RMSNorm(
            args.dim, eps=args.norm_eps, use_bias=False, use_weight=True
        )
        self.output = eqx.nn.Linear(
            args.dim, args.vocab_size, use_bias=False, key=linear_key
        )

        make_tf_layers = lambda k: TransformerBlock(args, key=k)
        self.layers = eqx.filter_vmap(make_tf_layers)(tf_layers_keys)

    def __call__(self, x, positions):
        # x is of shape (seqlen, ). We need to use vmap
        # as the embedding layer expects single token (scalar)
        # as input.
        h = jax.vmap(self.tok_embeddings)(x)  # output shape: [seqlen, embed_size]
        sin_freq = precomputed_sin_freq[positions]
        cos_freq = precomputed_cos_freq[positions]

        if x.shape[-1] > 1:
            seq_len = x.shape[-1]
            t = jnp.full((seq_len, seq_len), dtype=h.dtype, fill_value=1)
            mask = jnp.tril(t, k=0)
            # make the mask banded to account for sliding window
            mask = jnp.triu(mask, k=-args.sliding_window)
        else:
            mask = None

        # We need to call all the transformer blocks in a loop. Better to use lax.scan
        # as it would reduce compilation overhead and will be much faster.
        dynamic_tf_layers, static_tf_layers = eqx.partition(self.layers, eqx.is_array)

        def f(_x, _dynamic_tf_layers):
            tf_layer = eqx.combine(_dynamic_tf_layers, static_tf_layers)
            return tf_layer(_x, cos_freq, sin_freq, positions, mask), None

        h, _ = jax.lax.scan(f, h, dynamic_tf_layers)
        h = jax.vmap(self.norm)(h)
        h = jax.vmap(self.output)(h)
        # TODO: Calculate logits in this block
        return h


ModelArgs = namedtuple(
    "ModelArgs",
    [
        "dim",
        "n_layers",
        "hidden_dim",
        "n_heads",
        "head_dim",
        "n_kv_heads",
        "sliding_window",
        "norm_eps",
        "vocab_size",
        "max_batch_size",
    ],
)

args = ModelArgs(
    dim=4096,
    n_layers=32,
    n_heads=32,
    n_kv_heads=8,
    head_dim=128,
    hidden_dim=14336,
    vocab_size=32000,
    max_batch_size=1,
    sliding_window=4096,
    norm_eps=1e-5,
)

# Precomputed frequencies
precomputed_cos_freq, precomputed_sin_freq = precompute_frequencies(
    args.head_dim, 128_000
)


# # Example usage:
# max_seq_len = 10
# tok_inp = jnp.asarray(np.random.randint(0, args.vocab_size, size=(args.max_batch_size, max_seq_len)))
# print("Tokenized input shape: ", tok_inp.shape)
# # Get the positions
# positions = jnp.arange(0, max_seq_len)

# transformer = to_dtype(Transformer(args, key=jax.random.PRNGKey(1)), jnp.bfloat16)
# jitted_transformer = eqx.filter_jit(transformer)

# # Run for a single example of shape (seqlen,)
# o = transformer(tok_inp[0], positions)
# o = jitted_transformer(tok_inp[0], positions)
# # output shape: [seqlen, vocab_size]

# # Run for the full batch of shape (batch_size, seqlen)
# o = jax.vmap(transformer, in_axes=(0, None))(tok_inp, positions)
# o = jax.vmap(jitted_transformer, in_axes=(0, None))(tok_inp, positions)
# # output shape: [batch _size, seqlen, vocab_size]
