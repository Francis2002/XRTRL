import jax
import jax.numpy as jnp
from chex import assert_trees_all_equal_shapes
from online_lru.train_helpers import compute_grad
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

base_params = {
    "d_hidden": 10,
    "d_model": 2,
    "seq_length": 100,
}

inputs = jax.random.normal(
    jax.random.PRNGKey(0), (2, base_params["seq_length"], base_params["d_model"])
)

y = jnp.ones((2, base_params["seq_length"]))

mask = jax.numpy.ones((2, 100))

def _round_sig(x, n):
    x = jnp.asarray(x)
    absx = jnp.abs(x)
    is_zero = absx == 0
    # exponent (floor(log10(abs(x)))) for non-zero entries
    e = jnp.where(is_zero, 0.0, jnp.floor(jnp.log10(absx)))
    scale = jnp.power(10.0, (n - 1) - e)
    rounded = jnp.where(is_zero, 0.0, jnp.round(x * scale) / scale)
    return rounded

def equal_up_to_n_sigfigs(a, b, n=5):
    """
    Compare two pytrees (e.g. dicts) of arrays/scalars up to n significant figures.
    Returns True if all corresponding leaves are equal after rounding.
    """
    try:
        # ensure same pytree structure / shapes
        assert_trees_all_equal_shapes(a, b)
    except Exception:
        return False
    
    # if jnp.all(a == 0) or jnp.all(b == 0):
    #     return True

    ra = jax.tree_util.tree_map(lambda x: _round_sig(x, n), a)
    rb = jax.tree_util.tree_map(lambda x: _round_sig(x, n), b)

    leaves_a = jax.tree_util.tree_leaves(ra)
    leaves_b = jax.tree_util.tree_leaves(rb)

    for xa, xb in zip(leaves_a, leaves_b):
        if not bool(jnp.all(xa == xb)):
            return False
    return True


def loss_pred(pred, label, mask=None):
    if mask is None:
        return 0.5 * jnp.mean((pred - label) ** 2), jnp.zeros_like(pred)
    else:
        # mean loss over output axis:
        loss = jnp.mean((pred - label) ** 2, -1)
        # mean over mask loss
        return 0.5 * jnp.sum(mask * loss), jnp.zeros_like(pred)


def check_grad_all(grad_1, grad_2, to_check=None, **kwargs):
    test_failed = False
    # Check that the size matches
    assert_trees_all_equal_shapes(grad_1, grad_2)

    # Create a list of paths to variables to check
    if to_check is not None:
        paths, _ = jax.tree_util.tree_flatten_with_path(to_check)
        paths_to_check = []
        for path in paths:
            paths_to_check.append(
                "/".join([a.key for a in path[0] if a.__class__ == jax.tree_util.DictKey])
                + "/"
                + path[1]
            )
    else:
        flatten_grad_1 = jax.tree_util.tree_flatten_with_path(grad_1)[0]
        paths_to_check = [
            "/".join([a.key for a in g1[0] if a.__class__ == jax.tree_util.DictKey])
            for g1 in flatten_grad_1
        ]

    # For all the parameters to check, verify that they are close to each other
    for path in paths_to_check:
        keys = path.split("/")
        val1, val2 = grad_1, grad_2
        for key in keys:
            val1 = val1[key]
            val2 = val2[key]

        n_sig = kwargs.get("n_sig_digits", 5)

        try:
            assert equal_up_to_n_sigfigs(val1, val2, n_sig), "Mismatch at %s\nval1: %s\nval2: %s" % (path, val1, val2)
        except AssertionError as e:
            print(e)
            test_failed = True
            continue

    if test_failed:
        raise AssertionError("Gradient check failed.")


def compute_grads(model, params_states, inputs, y, mask):
    rng = jax.random.PRNGKey(0)
    # Compute grad online
    _, online_grad = compute_grad(params_states, rng, inputs, y, mask, model)

    model.training_mode = "bptt"
    _, grad = compute_grad(params_states, rng, inputs, y, mask, model)
    return grad, online_grad
