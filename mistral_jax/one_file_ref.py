import json
import numpy as np
from pathlib import Path
from typing import NamedTuple
from functools import partial

import jax
import torch
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float
from equinox._misc import default_floating_dtype

from tokenizer import MistralTokenizer
from rope import calculate_rope
from rope import precompute_frequencies
from weights_utils import port_weights_from_torch


# Set device to CPU for torch
device = torch.device("cpu")


class RMSNorm(eqx.Module):
    eps: float
    weight: Float[Array, "*shape"]  # noqa: F821

    def __init__(self, dim, eps, dtype=jnp.bfloat16):
        dtype = default_floating_dtype if dtype is None else dtype
        self.eps = eps
        self.weight = jnp.ones(shape=dim, dtype=dtype)

    def _norm(self, x):
        return x * jax.lax.rsqrt(jnp.mean(x**2, keepdims=True) + self.eps)

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

        self.w1 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key1, dtype=dtype
        )
        self.w2 = eqx.nn.Linear(
            args.hidden_dim, args.dim, use_bias=False, key=key2, dtype=dtype
        )
        self.w3 = eqx.nn.Linear(
            args.dim, args.hidden_dim, use_bias=False, key=key3, dtype=dtype
        )

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

        self.wq = eqx.nn.Linear(
            args.dim,
            args.n_heads * args.head_dim,
            use_bias=False,
            key=key1,
            dtype=dtype,
        )
        self.wk = eqx.nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            use_bias=False,
            key=key2,
            dtype=dtype,
        )
        self.wv = eqx.nn.Linear(
            args.dim,
            args.n_kv_heads * args.head_dim,
            use_bias=False,
            key=key3,
            dtype=dtype,
        )
        self.wo = eqx.nn.Linear(
            args.n_heads * args.head_dim,
            args.dim,
            use_bias=False,
            key=key4,
            dtype=dtype,
        )

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

    def __call__(
        self, x, cos_freq, sin_freq, positions, mask=None, cache_k=None, cache_v=None
    ):
        # x shape: [seqlen, embed_dim]
        seqlen, _ = x.shape
        # 1. Calculate qkv
        xq, xk, xv = self.compute_qkv(x)

        # 2. Calculate RoPE
        xq = calculate_rope(xq, cos_freq, sin_freq)
        xk = calculate_rope(xk, cos_freq, sin_freq)

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


class TransformerBlock(eqx.Module):
    dim: int
    n_heads: int
    attention: Attention
    attention_norm: RMSNorm
    feed_forward: FeedForward
    ffn_norm: RMSNorm

    def __init__(self, args, key, dtype=jnp.bfloat16):
        key1, key2 = jax.random.split(key, 2)
        self.n_heads = args.n_heads
        self.dim = args.dim

        self.attention = Attention(args, key=key1, dtype=dtype)
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

        self.feed_forward = FeedForward(args, key=key2, dtype=dtype)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps, dtype=dtype)

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        normed_x = jax.vmap(self.attention_norm)(x)
        r, cache_k, cache_v = self.attention(
            normed_x, cos_freq, sin_freq, positions, mask, cache_k, cache_v
        )
        h = x + r
        r = jax.vmap(self.feed_forward)(jax.vmap(self.ffn_norm)(h))
        out = h + r
        return out, cache_k, cache_v


class Transformer(eqx.Module):
    tok_embeddings: eqx.nn.Embedding
    layers: TransformerBlock
    norm: RMSNorm
    output: eqx.nn.Linear
    vocab_size: int
    n_layers: int
    sliding_window: int

    def __init__(self, args, key, dtype=jnp.bfloat16):
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.sliding_window = args.sliding_window
        keys = jax.random.split(key, args.n_layers + 2)
        embed_key, linear_key, tf_layers_keys = keys[0], keys[1], keys[2:]

        self.tok_embeddings = eqx.nn.Embedding(
            args.vocab_size, args.dim, key=embed_key, dtype=dtype
        )
        self.norm = RMSNorm(dim=args.dim, eps=args.norm_eps, dtype=dtype)
        self.output = eqx.nn.Linear(
            args.dim, args.vocab_size, use_bias=False, key=linear_key, dtype=dtype
        )
        self.layers = [
            TransformerBlock(args, key=tf_layers_keys[i], dtype=dtype)
            for i in range(args.n_layers)
        ]

    @eqx.filter_jit
    def compute_embeddings(self, x):
        return jax.vmap(self.tok_embeddings)(x)

    @eqx.filter_jit
    def compute_mask(self, seqlen):
        t = jnp.full((seqlen, seqlen), dtype=jnp.bfloat16, fill_value=1)
        mask = jnp.tril(t, k=0)
        # make the mask banded to account for sliding window
        mask = jnp.triu(mask, k=-self.sliding_window)
        mask = jnp.log(mask)
        return mask

    @eqx.filter_jit
    def compute_norm(self, x):
        return jax.vmap(self.norm)(x)

    @eqx.filter_jit
    def compute_output(self, x):
        return jax.vmap(self.output)(x)

    @partial(jax.jit, static_argnums=(1,))
    def update_cache_values(
        self, idx, cache_k, cache_v, cache_k_updates, cache_v_updates
    ):
        cache_k = cache_k.at[idx, :, :, :].set(cache_k_updates)
        cache_v = cache_v.at[idx, :, :, :].set(cache_v_updates)
        return cache_k, cache_v

    def __call__(self, x, cos_freq, sin_freq, positions, mask, cache_k, cache_v):
        # x is of shape (seqlen, )
        h = self.compute_embeddings(x)

        if x.shape[-1] > 1:
            seqlen = x.shape[-1]
            mask = self.compute_mask(seqlen)
        else:
            mask = None

        for i, layer in enumerate(self.layers):
            h, cache_ki, cache_vi = layer(
                h, cos_freq, sin_freq, positions, mask, cache_k[i, ...], cache_v[i, ...]
            )
            cache_k, cache_v = self.update_cache_values(
                i, cache_k, cache_v, cache_ki, cache_vi
            )

        h = self.compute_norm(h)
        h = self.compute_output(h).astype(jnp.float32)
        return h, cache_k, cache_v


class ModelArgs(NamedTuple):
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    hidden_dim: int
    vocab_size: int
    sliding_window: int
    norm_eps: float
    max_batch_size: int = 1


def load_torch_state_dict(model_weight_path):
    """Loads the original weights of the Mistral-7B model.

    Args:
        model_weights_path: Path to the `.pth` corresponding to the
            original torch weights.
    Returns:
        Ordered dict containing weights.
    """

    state_dict = torch.load(model_weight_path)
    print("Original torch weights loaded successfully!\n")
    return state_dict


def generate(
    model, tokenizer, cos_freq, sin_freq, cache_k, cache_v, args, max_tokens=36
):
    """Generate `max_tokens` given a prompt.

    Args:
        model: vmapped version of equinox model (Mistral-7B)
        tokenizer: Mistral-7B tokenizer
        cos_freq: Precomputed cosine frequencies
        sin_freq: Precomputed sine frequencies
        cache_k: The key cache of shape `(bs, n_layers, seqlen, n_kv_heads, head_dim)`
        cache_v: The value cache of shape `(bs, n_layers, seqlen, num_kv_heads, head_dim)`
        max_tokens: Number of output tokens to generate

    Returns:
        String containing the original prompt with decoded generated tokens.
    """

    # 1. Encode the prompts
    prompts = ["This is another test"]
    encoded_prompts = [tokenizer.encode(prompt) for prompt in prompts]
    prompt_lens = [len(x) for x in encoded_prompts]
    min_prompt_len = min(prompt_lens)
    max_prompt_len = max(prompt_lens)

    # 2. Using numpy to generate the desired input. Will replace it with something
    # better later on
    input_tokens = np.full(
        (len(prompts), max_prompt_len), tokenizer.pad_id, dtype=np.int32
    )
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, : len(encoded)] = jnp.array((encoded))
    # input_mask = input_tokens != tokenizer.pad_id
    cur_pos = min_prompt_len

    # 3. pre-fill
    positions = jnp.arange(0, min_prompt_len)
    positions_padded = jnp.pad(
        positions,
        (0, args.sliding_window - len(positions)),
        constant_values=args.sliding_window + 2,
    )
    logits, cache_k, cache_v = model(
        jnp.asarray(input_tokens[:, :min_prompt_len]),
        cos_freq[positions],
        sin_freq[positions],
        positions_padded,
        None,
        cache_k,
        cache_v,
    )
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    next_token = jnp.argmax(logprobs[:, -1, :], axis=-1)

    # 4. Generation
    generated = [next_token[0].item()]
    print("Generating...")

    for _ in range(max_tokens):
        cur_pos += 1
        pos = jnp.array([cur_pos])
        logits, cache_k, cache_v = logits, cache_k, cache_v = model(
            jnp.asarray(next_token[:, None]),
            cos_freq[pos],
            sin_freq[pos],
            pos,
            None,
            cache_k,
            cache_v,
        )
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        next_token = jnp.argmax(logprobs[:, -1, :], axis=-1)
        generated.append(next_token[0].item())

    res = prompts[0] + " " + "".join(tokenizer.decode(generated))
    print(res, "\n")
    return res


def main(model_files_path="../model_files/"):
    # Path containing all original model files related to Mitsral-7B
    model_files_path = Path(model_files_path)

    # 1. Load torch state dict
    state_dict = load_torch_state_dict(model_files_path / "consolidated.00.pth")

    # 2. Load arguments required for building the model
    with open(model_files_path / "params.json", "r") as f:
        args = ModelArgs(**json.loads(f.read()))

    # 3. Build equinox mistral-7b model
    model = Transformer(args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16)
    # 4. Port weights from torch to equinox model
    model = port_weights_from_torch(state_dict, model)

    # 5. Load the tokenizer
    tokenizer = MistralTokenizer(model_files_path / "tokenizer.model")

    # 5. Precomputed frequencies
    cos_freq, sin_freq = precompute_frequencies(args.head_dim, 128000)

    # 6. Define KV-cache
    cache_k = jnp.zeros(
        (
            args.max_batch_size,
            args.n_layers,
            args.sliding_window,
            args.n_kv_heads,
            args.head_dim,
        ),
        dtype=jnp.bfloat16,
    )
    cache_v = jnp.zeros(
        (
            args.max_batch_size,
            args.n_layers,
            args.sliding_window,
            args.n_kv_heads,
            args.head_dim,
        ),
        dtype=jnp.bfloat16,
    )

    # The attention layers expect five inputs one of which is the mask.
    # This mask is generated inside the `Transformer` module, and then passed
    # to other blocks. So, there is no need to include the `mask` argument
    # when calling the `Transformer` module. But. we want to `vmap` the entire
    # model in a sophisticated manner, so we will include a fake mask (`None`)
    # in the `__call__` argument of our `Transformer` module.
    # The semantics of the vmap are defined as:
    # (in_axes=(0, None, None, None, None, 0, 0)) where:
    #   0: Batch axis for the tokenized inputs
    #   None: No batch axis for the cosine frequencies
    #   None: No batch axis for the sine frequencies
    #   None: No batch axis for the positions
    #   None: No batch axis for the mask
    #   0: Batch axis for the key cache
    #   0: Batch axis for the value cache

    # 7. Define the vmapped version of the model.
    vmapped_model = eqx.filter_vmap(
        model, in_axes=(0, None, None, None, None, 0, 0)
    )

    # **NOTE:** The first call will be very slow as the model will be compiled
    # If you want to avoid that delay, please warm up your model with some fake inputs.

    # 8. Generate
    res = generate(
        vmapped_model,
        tokenizer,
        cos_freq=cos_freq,
        sin_freq=sin_freq,
        cache_k=cache_k,
        cache_v=cache_v,
        args=args,
        max_tokens=20,
    )
    return res


if __name__ == "__main__":
    model_files_path = Path("../model_files/")
    _ = main(model_files_path)
