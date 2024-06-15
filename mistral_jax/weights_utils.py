import json
from pathlib import Path
from typing import NamedTuple

import os

# We will be porting weights on CPU.
# This is only necessary if you don't have enough GPU memory. In my
# case, I was using a single A100 40G, and that isn't enough memory to hold
# the weights of three copies (original torch tensors, PyTrees initialized
# randomly, and corresponding PyTrees with modified weights).
os.environ["JAX_PLATFORM_NAME"] = "cpu"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch  # noqa: E402
import jax  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from mistral_model_optimized import Transformer  # noqa: E402


def port_weights_from_torch(torch_weights, eqx_model):
    """Recursively sets the weights of an equinox module
    from torch `state_dict()`.

    Example usage:
    ```
    model = Transformer(args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16)
    model = port_weights_from_torch(state_dict, model)
    ```

    Args:
        torch_weights: State dict containing weights of the torch model
        eqx_model: Equinox model with same layer names as Pytorch model

    Returns:
        Pytree with the desired weight values
    """

    def load_weights(path, leaf):
        path_pieces = []
        for path_elem in path:
            if isinstance(path_elem, jax.tree_util.GetAttrKey):
                path_pieces.append(path_elem.name)
            elif isinstance(path_elem, jax.tree_util.SequenceKey):
                path_pieces.append(str(path_elem.idx))
            else:
                raise ValueError(f"Unsupported path type {type(path_elem)}")

        path_pieces = ".".join(path_pieces)

        if "weight" in path_pieces:
            weight = torch_weights[path_pieces]
            # `bfloat16` weights cannot be directly converted to numpy. Hence
            # we first upscale them to `float32`, and then load them in
            # `bfloat16`
            weight = jnp.asarray(weight.float().numpy(), dtype=jnp.bfloat16)
            assert weight.shape == leaf.shape
            assert weight.dtype == leaf.dtype
            return weight
        else:
            print(f"Weights not ported for: {path_pieces}")
            return leaf

    return jax.tree_util.tree_map_with_path(load_weights, eqx_model)


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


if __name__ == "__main__":
    # 1. Path to the assets
    model_files_path = Path("../model_files/")

    # 2. Set the device to CPU for torch
    device = torch.device("cpu")

    # 3. Load the merged state dict we generated before this
    state_dict = torch.load(model_files_path / "merged_state_dict_mistral7B.pth")

    # 3. Load the args required to build the Mistral model
    with open(model_files_path / "params.json", "r") as f:
        args = ModelArgs(**json.loads(f.read()))

    # 3. Build the Mistral model in Equinox
    model = Transformer(args, key=jax.random.PRNGKey(1), dtype=jnp.bfloat16)

    # 4. Port weights from torch to the Equniox model
    model = port_weights_from_torch(state_dict, model)

    # 5. Serialize the Equinox model so that we can load it directly
    eqx.tree_serialise_leaves(model_files_path / "mistral7B.eqx", model)
