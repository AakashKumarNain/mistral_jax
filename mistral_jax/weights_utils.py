import jax
import numpy as np
import jax.numpy as jnp


def load_weights(torch_weights, path, leaf):
    """Recursively sets the weights of an equinox module
    from torch `state_dict()`.

    Args:
        torch_weights: State dict containing weights of the torch model
        path: A Pytree
        leaf: Equinox module

    Example usage:
    eqx_model = ...    
    eqx_model jax.tree_util.tree_map_with_path(
        partial(torch_state_dict, load_weights), eqx_model
    )

    Returns:
        Pytree with the desired values
    """

    path_pieces = []
    for path_elem in path:
        if isinstance(path_elem, jax.tree_util.GetAttrKey):
            path_pieces.append(path_elem.name)
        elif isinstance(path_elem, jax.tree_util.SequenceKey):
            path_pieces.append(str(path_elem.idx))
        else:
            raise ValueError(f"Unsupported path type {type(path_elem)}")
    weight = torch_weights[".".join(path_pieces)]
    weight = jnp.asarray(np.asarray(weight))
    assert weight.shape == leaf.shape
    assert weight.dtype == leaf.dtype
    return weight