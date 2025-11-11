import os
import unittest
from functools import partial
from online_lru.seq_model import StackedEncoder
from online_lru.rec import LRU
import jax
from utils import base_params, inputs, y, mask, compute_grads, check_grad_all
import flax.linen as nn

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


batched_StackedEncoder = nn.vmap(
    StackedEncoder,
    in_axes=0,
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "traces": 0,
        "perturbations": 0,
        "cache": 0,
    },
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False, "dropout": True},
)


class TestStackedEncoder(unittest.TestCase):
    def test_online(self):
        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "gelu",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"
        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )
        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8)

    def test_online_prenorm(self):
        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "gelu",
            "n_layers": 2,
            "prenorm": True,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"
        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )
        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8) 

    def test_online_no_act(self):
        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "none",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"
        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )
        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8)

import flax
import jax.numpy as jnp

class TestStackedEncoderXRTRLInternals(unittest.TestCase):
    def test_tiny_net_no_norm(self):
        base_params = {
            "d_hidden": 3,
            "d_model": 2,
            "seq_length": 1,
        }

        inputs = jax.random.normal(
            jax.random.PRNGKey(0), (1, base_params["seq_length"], base_params["d_model"])
        )

        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "gelu",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"

        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )

        # Make all params identity except gamma_log, nu, theta, B_re, B_im
        params_states = flax.core.frozen_dict.unfreeze(params_states)
        for layer_key in params_states["params"].keys():
            if layer_key.startswith("layers_"):
                layer_params = params_states["params"][layer_key]["seq"]
                for param_key in layer_params.keys():
                    if param_key not in ["gamma_log", "nu", "theta", "B_re", "B_im"]:
                        layer_params[param_key] = jnp.ones_like(layer_params[param_key])
                params_states["params"][layer_key]["norm"]["bias"] = jnp.zeros_like(
                    params_states["params"][layer_key]["norm"]["bias"]
                )
                params_states["params"][layer_key]["norm"]["scale"] = jnp.ones_like(
                    params_states["params"][layer_key]["norm"]["scale"]
                )

        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8)

    def test_tiny_net_no_norm_no_act(self):
        base_params = {
            "d_hidden": 3,
            "d_model": 2,
            "seq_length": 1,
        }

        inputs = jax.random.normal(
            jax.random.PRNGKey(0), (1, base_params["seq_length"], base_params["d_model"])
        )

        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "none",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"

        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )

        # Make all params identity except gamma_log, nu, theta, B_re, B_im
        params_states = flax.core.frozen_dict.unfreeze(params_states)
        for layer_key in params_states["params"].keys():
            if layer_key.startswith("layers_"):
                layer_params = params_states["params"][layer_key]["seq"]
                for param_key in layer_params.keys():
                    if param_key not in ["gamma_log", "nu", "theta", "B_re", "B_im"]:
                        layer_params[param_key] = jnp.ones_like(layer_params[param_key])
                params_states["params"][layer_key]["norm"]["bias"] = jnp.zeros_like(
                    params_states["params"][layer_key]["norm"]["bias"]
                )
                params_states["params"][layer_key]["norm"]["scale"] = jnp.ones_like(
                    params_states["params"][layer_key]["norm"]["scale"]
                )

        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8)

    def test_tiny_net_no_B_or_no_C(self):
        base_params = {
            "d_hidden": 3,
            "d_model": 2,
            "seq_length": 1,
        }

        inputs = jax.random.normal(
            jax.random.PRNGKey(0), (1, base_params["seq_length"], base_params["d_model"])
        )

        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "gelu",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"

        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )

        # Zero C of layer_0 => h of layer_1 does not depend on h of layer_0 => cross-layer contribs are 0
        params_states_no_C = flax.core.frozen_dict.unfreeze(params_states)
        for layer_key in params_states_no_C["params"].keys():
            if layer_key.startswith("layers_"):
                layer_params = params_states_no_C["params"][layer_key]["seq"]
                for param_key in layer_params.keys():
                    if param_key in ["C_re", "C_im"] and layer_key == "layers_0":
                        layer_params[param_key] = jnp.zeros_like(layer_params[param_key])

        # Zero B of layer_1 => h of layer_1 does not depend on h of layer_0 => cross-layer contribs are 0
        params_states_no_B = flax.core.frozen_dict.unfreeze(params_states)
        for layer_key in params_states_no_B["params"].keys():
            if layer_key.startswith("layers_"):
                layer_params = params_states_no_B["params"][layer_key]["seq"]
                for param_key in layer_params.keys():
                    if param_key in ["B_re", "B_im"] and layer_key == "layers_1":
                        layer_params[param_key] = jnp.zeros_like(layer_params[param_key])

        _, online_grad_no_C = compute_grads(batched_stacked_encoder, params_states_no_C, inputs, y, mask)
        _, online_grad_no_B = compute_grads(batched_stacked_encoder, params_states_no_B, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad_no_C, _ = compute_grads(batched_seqlayer, params_states_no_C, inputs, y, mask)
        grad_no_B, _ = compute_grads(batched_seqlayer, params_states_no_B, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad_no_C, online_grad_no_C, to_check=dict_check, n_sig_digits=5, atol=1e-8)
        check_grad_all(grad_no_B, online_grad_no_B, to_check=dict_check, n_sig_digits=5, atol=1e-8)

    def test_tiny_net_no_act(self):
        base_params = {
            "d_hidden": 3,
            "d_model": 2,
            "seq_length": 1,
        }

        inputs = jax.random.normal(
            jax.random.PRNGKey(0), (1, base_params["seq_length"], base_params["d_model"])
        )

        lru = partial(LRU, training_mode="online_xrtrl", **base_params)
        seq_layer_params = {
            "rec": lru,
            "dropout": 0.0,
            "d_input": 2,
            "d_model": 2,
            "seq_length": base_params["seq_length"],
            "training_mode": "online_xrtrl",
            "training": True,
            "activation": "none",
            "n_layers": 2,
        }
        batched_stacked_encoder = batched_StackedEncoder(**seq_layer_params)
        batched_stacked_encoder.rec_type = "LRU"

        params_states = batched_stacked_encoder.init(
            {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(0)}, inputs
        )

        _, online_grad = compute_grads(batched_stacked_encoder, params_states, inputs, y, mask)

        lru = partial(LRU, training_mode="bptt", **base_params)
        seq_layer_params["training_mode"] = "bptt"
        seq_layer_params["rec"] = lru
        batched_seqlayer = batched_StackedEncoder(**seq_layer_params)
        batched_seqlayer.rec_type = "LRU"
        grad, _ = compute_grads(batched_seqlayer, params_states, inputs, y, mask)

        dict_check = {
            "layers_1": {
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
            },
             "layers_0": {
                "seq": [
                    "gamma_log",
                    "nu",
                    "theta",
                ],
            }
        }
        check_grad_all(grad, online_grad, to_check=dict_check, n_sig_digits=5, atol=1e-8)

