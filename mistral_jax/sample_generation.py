import json
import numpy as np
from pathlib import Path
from typing import NamedTuple

import jax
import torch
import equinox as eqx
import jax.numpy as jnp

from rope import precompute_frequencies
# from mistral_model import Transformer
from mistral_model_optimized import Transformer
from tokenizer import MistralTokenizer
from weights_utils import port_weights_from_torch

# Set device to CPU for torch
device = torch.device("cpu")


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


def generate(model, tokenizer, cos_freq, sin_freq, cache_k, cache_v, args, max_tokens=36):
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
    input_mask = input_tokens != tokenizer.pad_id
    cur_pos = min_prompt_len

    # 3. pre-fill
    positions = jnp.arange(0, min_prompt_len)
    positions_padded = jnp.pad(positions, (0, args.sliding_window - len(positions)), constant_values=args.sliding_window+2)
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
    vmapped_model = eqx.filter_vmap(eqx.filter_jit(model), in_axes=(0, None, None, None, None, 0, 0))

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
