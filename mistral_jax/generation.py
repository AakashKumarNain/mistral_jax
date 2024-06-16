import json
import time
from pathlib import Path
from typing import NamedTuple

import jax
import equinox as eqx
import jax.numpy as jnp

from rope import precompute_frequencies
from model import Transformer
from tokenizer import MistralTokenizer


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


def generate(
    prompts, model, tokenizer, cos_freq, sin_freq, args: ModelArgs, max_tokens: int = 36
):
    """Generate `max_tokens` for each prompt in a given list of prompts.

    Args:
        prompts: List of strings(prompts)
        model: vmapped version of equinox model (Mistral-7B)
        tokenizer: Mistral-7B tokenizer
        cos_freq: Precomputed cosine frequencies
        sin_freq: Precomputed sine frequencies
        args: Original arguments used to build the Mistral-7B model
        max_tokens: Number of output tokens to generate
    """

    cache_shape = (
        args.max_batch_size,
        args.n_layers,
        args.sliding_window,
        args.n_kv_heads,
        args.head_dim,
    )
    outputs = []

    for prompt in prompts:
        # 1. Encode the prompt
        encoded = tokenizer.encode(prompt)
        cur_pos = len(encoded)

        # 2. We need to flush the cache with every prompt.
        # Need a better way to do this but for now it's okay!
        cache_k = jnp.zeros(cache_shape, dtype=jnp.bfloat16)
        cache_v = jnp.zeros(cache_shape, dtype=jnp.bfloat16)

        # 3. pre-fill
        positions = jnp.arange(0, cur_pos)
        positions_padded = jnp.pad(
            positions, (0, args.sliding_window - len(positions)), constant_values=-1
        )
        print("Prefilling...", end="   ")
        start = time.time()
        logits, cache_k, cache_v = model(
            jnp.asarray([encoded]),
            cos_freq[positions],
            sin_freq[positions],
            positions_padded,
            None,
            cache_k,
            cache_v,
        )
        print(f"Time taken : {time.time()- start :.2f} seconds")
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        next_token = jnp.argmax(logprobs[:, -1, :], axis=-1)

        # 4. Generation
        generated = [next_token[0].item()]
        print("Generating...", end="   ")
        overall_start = time.time()
        for t in range(max_tokens):
            cur_pos += 1
            pos = jnp.array([cur_pos])
            logits, cache_k, cache_v = logits, cache_k, cache_v = model(
                jax.lax.expand_dims(next_token, (1,)),
                cos_freq[pos],
                sin_freq[pos],
                pos,
                None,
                cache_k,
                cache_v,
            )
            logprobs = jax.nn.log_softmax(logits[:, -1, :], axis=-1)
            next_token = jnp.argmax(logprobs, axis=-1)
            generated.append(next_token[0].item())

        end = time.time()
        print(
            f"Time taken to generate {max_tokens} tokens: {end- overall_start:.2f} seconds"
        )
        print("\nPrompt     : ", prompt)
        print("Completion :", end=" ")
        res = prompt + " " + "".join(tokenizer.decode(generated))
        print(repr(res))
        print("\n", "=" * 75)
        outputs.append(res)
        return outputs


def warmup_model(model, cos_freq, sin_freq, cache_k, cache_v, args):
    """Warmup the model for prefilling(for one seq length) and generation."""
    fake_pos = jnp.array([0, 1, 2, 3, 4], dtype=jnp.int32)
    fake_inp = jnp.asarray([[1, 832, 349, 265, 1369]], dtype=jnp.int32)
    fake_mask = None
    fake_pos_padded = jnp.pad(
        fake_pos, (0, args.sliding_window - len(fake_pos)), constant_values=-1
    )

    # warmup for prefilling
    # Note: Every time the sequence length changes, your model will get compiled for that sequence
    # length? Why? Because we are populating the cache dynamically.
    _ = model(
        fake_inp,
        cos_freq[fake_pos],
        sin_freq[fake_pos],
        fake_pos_padded,
        fake_mask,
        cache_k,
        cache_v,
    )

    # warmup for generation
    fake_pos = jnp.array([5], dtype=jnp.int32)
    fake_inp = jnp.asarray([[1369]], dtype=jnp.int32)
    fake_mask = None
    _ = model(
        fake_inp,
        cos_freq[fake_pos],
        sin_freq[fake_pos],
        fake_pos_padded,
        fake_mask,
        cache_k,
        cache_v,
    )


def main(model_files_path="../model_files/"):
    # Path containing all original model files related to Mitsral-7B
    model_files_path = Path(model_files_path)

    # 1. Load arguments required for building the model
    with open(model_files_path / "params.json", "r") as f:
        args = ModelArgs(**json.loads(f.read()))
    print("Model config loaded successfully!")

    # 2. Build equinox mistral-7b model
    model = Transformer(args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16)
    model = eqx.tree_deserialise_leaves(
        model_files_path / "mistral7B_jax_port_fast.eqx", model
    )
    print("Model weights loaded successfully!")

    # 3. Load the tokenizer
    tokenizer = MistralTokenizer(model_files_path / "tokenizer.model")
    print("Tokenizer loaded successfully!")

    # 4. Precomputed frequencies
    cos_freq, sin_freq = precompute_frequencies(args.head_dim, 128000)

    # 5. Define KV-cache
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

    # 6. Define the vmapped version of the model.
    model = eqx.filter_vmap(
        eqx.filter_jit(model), in_axes=(0, None, None, None, None, 0, 0)
    )

    # **NOTE:** The first call will be very slow as the model will be compiled
    # If you want to avoid that delay, please warm up your model with some fake inputs.
    _ = warmup_model(
        model,
        cos_freq=cos_freq,
        sin_freq=sin_freq,
        cache_k=cache_k,
        cache_v=cache_v,
        args=args,
    )
    print("")

    prompts = [
        "This is a test",
        "Hello, I am a language model,",
        "I am a helpful assistant",
    ]

    # 7. Generate
    _ = generate(
        prompts,
        model,
        tokenizer,
        cos_freq=cos_freq,
        sin_freq=sin_freq,
        args=args,
        max_tokens=64,
    )


if __name__ == "__main__":
    model_files_path = Path("../model_files/")
    _ = main(model_files_path)
