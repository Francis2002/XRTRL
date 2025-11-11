import os
import unittest
from functools import partial
from online_lru.layers import SequenceLayer
from online_lru.rec import LRU
import jax
from utils import base_params, inputs, y, mask, compute_grads, check_grad_all
import flax.linen as nn
import jax.numpy as jnp
from online_lru.utils.util import _take_t
from tests.utils import equal_up_to_n_sigfigs

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


batched_SequenceLayer = nn.vmap(
    SequenceLayer,
    in_axes=0,
    out_axes=0,
    variable_axes={"params": None, "dropout": None, "traces": 0, "perturbations": 0, "cache": 0},
    methods=["__call__", "update_gradients", "pre_seq", "run_seq", "post_seq_with_cached_dropout_at_t", "post_seq_with_cached_dropout", "post_skip"],
    split_rngs={"params": False, "dropout": True},
)


class TestLayers(unittest.TestCase):
    def test_online(self):
        """
        Check that the parameter update computed online by an SequenceLayer is
        the correct one, that is the gradient for all parameters that appear
        within or after the LRU layer.
        """
        lru = partial(LRU, training_mode="online_full", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.1,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_full",
            "training": True,
            "activation": "full_glu",
        }
        batched_seqlayer = batched_SequenceLayer(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        params_states = batched_seqlayer.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)},
            inputs,
        )
        _, online_grad = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_SequenceLayer(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "seq": [
                "B_re",
                "B_im",
                "C_re",
                "C_im",
                "D",
                "gamma_log",
                "nu",
                "theta",
            ],
            "out1": ["bias", "kernel"],
            "out2": ["bias", "kernel"],
        }
        check_grad_all(grad, online_grad, to_check=dict_check, atol=1e-1)

    def test_layer_replay(self):
        """
        Check that the cached dropout implementation is equivalent to standard
        dropout when using the same masks. And that both are equivalent to the
        initial forward pass.
        """

        test_failed = False
        
        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.1,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "full_glu",
        }
        batched_seqlayer = batched_SequenceLayer(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        params_states = batched_seqlayer.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)},
            inputs,
        )
        
        # 1) Forward to populate masks in “cache”
        out1, mod_vars = batched_seqlayer.apply(params_states, inputs, rngs={"dropout": jax.random.PRNGKey(0)}, mutable=["cache"])

        # Recompute the pieces via public methods (no nested submodule poking)
        h_pre = batched_seqlayer.apply(
            params_states, inputs, method=batched_seqlayer.pre_seq
        )

        # LRU.__call__(h_pre) returns y directly (it computes hidden + to_output internally)
        y = batched_seqlayer.apply(
            params_states, h_pre, method=batched_seqlayer.run_seq
        )

        # 2) Replay with sequence cached dropout
        post_seq_cached = batched_seqlayer.apply(
            params_states, y, method=batched_seqlayer.post_seq_with_cached_dropout
        )
        post_skip_cached = inputs + post_seq_cached
        after_post_skip_cached = batched_seqlayer.apply(
            params_states, post_skip_cached, method=batched_seqlayer.post_skip
        )

        # Slice at t for all ts
        after_post_skip_t_cacheds = []
        for t in range(base_params["seq_length"]):
            t = jnp.int32(t)
            batch_size = inputs.shape[0]
            t_batched = jnp.full((batch_size,), t, dtype=jnp.int32)
            # _take_t expects a sequence of shape (T, ...) and a single scalar t, so vectorize it across the batch dimension
            take_t_batched = jax.vmap(_take_t, in_axes=(0, 0))
            y_t = take_t_batched(y, t_batched)

            # 2) Replay with cached dropout at t
            post_seq_t_cached = batched_seqlayer.apply(
                params_states, y_t, t_batched, method=batched_seqlayer.post_seq_with_cached_dropout_at_t
            )

            post_skip_t_cached = take_t_batched(inputs, t_batched) + post_seq_t_cached
            after_post_skip_t_cached = batched_seqlayer.apply(
                params_states, post_skip_t_cached, method=batched_seqlayer.post_skip            # First slice then compute path
            )

            after_post_skip_t_cacheds.append(after_post_skip_t_cached)

        # Stack back the time dimension
        after_post_skip_t_cacheds = jnp.stack(after_post_skip_t_cacheds, axis=1)  # shape (B, T, H)
        
        # Checks
        try:
            assert equal_up_to_n_sigfigs(after_post_skip_t_cacheds, after_post_skip_cached, 5), "Mismatch in cached dropout outputs at time t"
        except AssertionError as e:
            print(e)
            test_failed = True
        try :
            assert equal_up_to_n_sigfigs(after_post_skip_t_cacheds, out1, 5), "Mismatch in cached dropout vs initial outputs at time t"
        except AssertionError as e:
            print(e)
            test_failed = True
        try:
            assert equal_up_to_n_sigfigs(after_post_skip_cached, out1, 5), "Mismatch in sequence cached dropout vs initial outputs at time t"
        except AssertionError as e:
            print(e)
            test_failed = True

        if test_failed:
            raise AssertionError("Cached dropout test failed.")

        
