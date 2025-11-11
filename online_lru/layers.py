from flax import linen as nn
import jax
import jax.numpy as jnp

class SequenceLayer(nn.Module):
    """
    Defines a single layer, with one of the rec module, nonlinearity, dropout, batch/layer norm,
        etc.

    Args:
        rec             (nn.Module):    the recurrent layer to use
        dropout         (float32):      dropout rate
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        seq_length      (int32):        length of the time sequence considered
                                        we usually refer to this size as T
        activation      (string):       type of activation function to use
        training_mode   (string):       type of training
        online          (bool):         whether gradient calculation is done online
        prenorm         (bool):         apply prenorm if true or postnorm if false
    """

    rec: nn.Module
    dropout: float
    d_model: int
    seq_length: int
    activation: str = "gelu"
    training: bool = True
    training_mode: str = "bptt"
    prenorm: bool = False

    def setup(self):
        """Initializes the rec, layer norm and dropout"""
        self.seq = self.rec()

        if self.activation in ["full_glu"]:
            self.out1 = nn.Dense(self.d_model)
            self.out2 = nn.Dense(self.d_model)
        elif self.activation in ["half_glu1", "half_glu2"]:
            self.out2 = nn.Dense(self.d_model)

        self.norm = nn.LayerNorm()

        self.drop = nn.Dropout(
            self.dropout,
            broadcast_dims=[0],
            deterministic=not self.training,
        )

        if self.training_mode in ['online_xrtrl']:
            self.dropout_mask_1 = self.variable(
                    "cache", "dropout_mask_1", jnp.zeros, (self.seq_length, self.d_model)
                )
            self.dropout_mask_2 = self.variable(
                "cache", "dropout_mask_2", jnp.zeros, (self.seq_length, self.d_model)
            )

    def run_seq(self, z):
        """
        Run the recurrent module over the whole sequence.
        Used to expose the recurrent core for testing purposes.
        """
        return self.seq(z)

    def apply_dropout(self, x, rate: float, name: str):
        if rate == 0.0:
            return x
        
        if self.training_mode not in ['online_xrtrl']:
            return self.drop(x)

        # online xrtrl mode must cache dropout masks!
        key = self.make_rng("dropout")
        keep = 1.0 - rate
        mask = (jax.random.bernoulli(key, keep, shape=x.shape).astype(x.dtype) / keep)
        # overwrite each forward → no stale or wrong-shape masks
        if name == "dropout_mask_1":
            var = self.dropout_mask_1
        else:
            var = self.dropout_mask_2
        var.value = mask
        return x * mask

    def apply_cached_dropout(self, x, rate: float, var):
        if rate == 0.0:
            return x

        return x if var is None else (x * var.value)
    
    # time-indexed cached dropout (for single-time-step replay)
    def apply_cached_dropout_at_t(self, x_t, t: int, rate: float, var):
        if rate == 0.0:
            return x_t

        if var is None:
            return x_t
        
        mask = var.value  # shape (T, H)
        mask_t = jax.lax.dynamic_index_in_dim(mask, t, keepdims=False)  # shape (H,)
        return x_t * mask_t

    def pre_seq(self, x):
        """
        Processing done to the input before calling the recurrent module.
        """
        if self.prenorm:
            x = self.norm(x)
        return x

    def post_seq(self, x):
        """
        Processing done after the recurrent layer, inside the main branch, before the
        merge with the skip connection.
        """
        if self.activation in ["full_glu", "half_glu1", "half_glu2"]:
            out2 = self.out2
        if self.activation in ["full_glu"]:
            out1 = self.out1

        if self.activation in ["full_glu"]:
            x = self.apply_dropout(nn.gelu(x), self.dropout, "dropout_mask_1")
            x = out1(x) * jax.nn.sigmoid(out2(x))
            x = self.apply_dropout(x, self.dropout, "dropout_mask_2")
        elif self.activation in ["half_glu1"]:
            x = self.apply_dropout(nn.gelu(x), self.dropout, "dropout_mask_1")
            x = x * jax.nn.sigmoid(out2(x))
            x = self.apply_dropout(x, self.dropout, "dropout_mask_2")
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.apply_dropout(nn.gelu(x), self.dropout, "dropout_mask_1")
            x = x * jax.nn.sigmoid(out2(x1))
            x = self.apply_dropout(x, self.dropout, "dropout_mask_2")
        elif self.activation in ["gelu"]:
            x = self.apply_dropout(nn.gelu(x), self.dropout, "dropout_mask_1")
        elif self.activation in ["none"]:
            x = x
        else:
            raise NotImplementedError("Activation: {} not implemented".format(self.activation))
        return x

    def post_seq_with_cached_dropout(self, x):
        """
        Same as post_seq but using cached dropout masks. For online xrtrl training mode eligibility trace calculation.
        """
        if self.activation in ["full_glu", "half_glu1", "half_glu2"]:
            out2 = self.out2
        if self.activation in ["full_glu"]:
            out1 = self.out1

        if self.activation in ["full_glu"]:
            x = self.apply_cached_dropout(nn.gelu(x), self.dropout, self.dropout_mask_1)
            x = out1(x) * jax.nn.sigmoid(out2(x))
            x = self.apply_cached_dropout(x, self.dropout, self.dropout_mask_2)
        elif self.activation in ["half_glu1"]:
            x = self.apply_cached_dropout(nn.gelu(x), self.dropout, self.dropout_mask_1)
            x = x * jax.nn.sigmoid(out2(x))
            x = self.apply_cached_dropout(x, self.dropout, self.dropout_mask_2)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.apply_cached_dropout(nn.gelu(x), self.dropout, self.dropout_mask_1)
            x = x * jax.nn.sigmoid(out2(x1))
            x = self.apply_cached_dropout(x, self.dropout, self.dropout_mask_2)
        elif self.activation in ["gelu"]:
            x = self.apply_cached_dropout(nn.gelu(x), self.dropout, self.dropout_mask_1)
        elif self.activation in ["none"]:
            x = x
        else:
            raise NotImplementedError("Activation: {} not implemented".format(self.activation))
        return x
    
    def post_seq_with_cached_dropout_at_t(self, x, t):
        """
        Same as post_seq but using cached dropout masks. For online xrtrl training mode eligibility trace calculation.
        """
        if self.activation in ["full_glu", "half_glu1", "half_glu2"]:
            out2 = self.out2
        if self.activation in ["full_glu"]:
            out1 = self.out1

        if self.activation in ["full_glu"]:
            x = self.apply_cached_dropout_at_t(nn.gelu(x), t, self.dropout, self.dropout_mask_1)
            x = out1(x) * jax.nn.sigmoid(out2(x))
            x = self.apply_cached_dropout_at_t(x, t, self.dropout, self.dropout_mask_2)
        elif self.activation in ["half_glu1"]:
            x = self.apply_cached_dropout_at_t(nn.gelu(x), t, self.dropout, self.dropout_mask_1)
            x = x * jax.nn.sigmoid(out2(x))
            x = self.apply_cached_dropout_at_t(x, t, self.dropout, self.dropout_mask_2)
        elif self.activation in ["half_glu2"]:
            # Only apply GELU to the gate input
            x1 = self.apply_cached_dropout_at_t(nn.gelu(x), t, self.dropout, self.dropout_mask_1)
            x = x * jax.nn.sigmoid(out2(x1))
            x = self.apply_cached_dropout_at_t(x, t, self.dropout, self.dropout_mask_2)
        elif self.activation in ["gelu"]:
            x = self.apply_cached_dropout_at_t(nn.gelu(x), t, self.dropout, self.dropout_mask_1)
        elif self.activation in ["none"]:
            x = x
        else:
            raise NotImplementedError("Activation: {} not implemented".format(self.activation))
        return x

    def post_skip(self, x):
        """
        Processing done after the skip and main connections have been added together.
        """
        if not self.prenorm:
            x = self.norm(x)
        return x

    def __call__(self, x):
        inputs = x
        hiddens_pre_seq = self.pre_seq(inputs)
        hiddens_post_seq = self.seq(hiddens_pre_seq)
        hiddens_post_skip = inputs + self.post_seq(hiddens_post_seq)
        return self.post_skip(hiddens_post_skip)

    def update_gradients(self, grad, inputs):
        # Update the gradients of the recurrent module
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        grad["seq"] = self.seq.update_gradients(grad["seq"], inputs)
        return grad
