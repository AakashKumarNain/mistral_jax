import numpy as np
from typing import NamedTuple
from functools import partial

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import torch
import equinox as eqx

from rope import precompute_frequencies
from tokenizer import MistralTokenizer


# Set device to CPU for torch
device  = torch.device("cpu")


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


def generate(model, tokenizer, cos_freq, sin_freq, cache_k, cache_v, max_tokens=36):
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
    input_tokens = np.full((len(prompts), max_prompt_len), tokenizer.pad_id, dtype=np.int32)
    for i, encoded in enumerate(encoded_prompts):
        input_tokens[i, :len(encoded)] = jnp.array((encoded))
    input_mask = input_tokens != tokenizer.pad_id
    cur_pos = min_prompt_len
    
    # 3. pre-fill
    positions = jnp.arange(0, min_prompt_len)
    logits, cache_k, cache_v = model(
        jnp.asarray(input_tokens[:, :min_prompt_len]),
        cos_freq[positions],
        sin_freq[positions],
        positions,
        None,
        cache_k,
        cache_v
    )
    logprobs = jax.nn.log_softmax(logits, axis=-1)
    next_token = jnp.argmax(logprobs[:, -1,:], axis=-1)

    # 4. Generation
    generated = [next_token[0].item()]
    print("Generating...")

    for _ in range(max_tokens):
        cur_pos+=1
        pos = jnp.array([cur_pos])
        logits, cache_k, cache_v = logits, cache_k, cache_v = model(
            jnp.asarray(next_token[:, None]),
            cos_freq[pos],
            sin_freq[pos],
            pos,
            None,
            cache_k,
            cache_v
        )
        logprobs = jax.nn.log_softmax(logits, axis=-1)
        next_token = jnp.argmax(logprobs[:, -1,:], axis=-1)
        generated.append(next_token[0].item())

    res = prompts[0] + " " + "".join(tokenizer.decode(generated))
    print(res, "\n")
    return res
