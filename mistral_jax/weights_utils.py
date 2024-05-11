import jax
import jax.numpy as jnp


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
