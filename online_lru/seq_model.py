import jax
import jax.numpy as jnp
from flax import linen as nn
from .layers import SequenceLayer


class StackedEncoder(nn.Module):
    """
    Defines a stack of SequenceLayer to be used as an encoder.

    Args:
        rec             (nn.Module):    the recurrent module to use
        n_layers        (int32):        the number of SequenceLayer to stack
        dropout         (float32):      dropout rate
        d_input         (int32):        this is the feature size of the encoder inputs
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        seq_length      (int32):        length of the time sequence considered
                                        we usually refer to this size as T
        activation      (string):       type of activation function to use
        training        (bool):         whether in training mode or not
        training_mode   (string):       type of training
        prenorm         (bool):         apply prenorm if true or postnorm if false
    """

    rec: nn.Module
    n_layers: int
    d_input: int
    d_model: int
    seq_length: int
    activation: str = "gelu"
    readout: int = 0
    dropout: float = 0.0
    training: bool = True
    training_mode: str = "bptt"
    prenorm: bool = False

    def _c2r_vec(z):   # (N,) complex -> (2N,) real
        return jnp.concatenate([jnp.real(z), jnp.imag(z)], axis=-1)

    def _r2c_vec(x):   # (2N,) real -> (N,) complex
        n = x.shape[-1] // 2
        return x[:n] + 1j * x[n:]

    def _c2r_cols(Z):  # (N, M) complex columns -> (M, 2N) real
        return jnp.concatenate([jnp.real(Z).T, jnp.imag(Z).T], axis=-1)

    def _r2c_cols(Y):  # (M, 2H) real -> (H, M) complex (columns)
        m, twoh = Y.shape
        h = twoh // 2
        return (Y[:, :h] + 1j * Y[:, h:]).T

    def setup(self):
        """
        Initializes a linear encoder and the stack of SequenceLayer.
        """
        self.encoder = nn.Dense(self.d_model)
        self.layers = [
            SequenceLayer(
                rec=self.rec,
                dropout=self.dropout,
                d_model=self.d_model,
                seq_length=self.seq_length,
                activation=self.activation,
                training=self.training,
                training_mode=self.training_mode,
                prenorm=self.prenorm,
            )
            for _ in range(self.n_layers)
        ]
        if self.readout > 0:
            self.mlp = nn.Dense(self.readout)

    def __call__(self, x):
        """
        Compute the TxH output of the stacked encoder given an Txd_input
        input sequence.
        Args:
             x (float32): input sequence (T, d_input)
        Returns:
            output sequence (float32): (T, d_model)
        """
        x = self.encoder(x)
        for i, layer in enumerate(self.layers):
            x = layer(x)

        if self.training_mode == "online_reservoir":
            x = jax.lax.stop_gradient(x)

        if self.readout > 0:
            x = nn.relu(self.mlp(x))
        return x

    def update_gradients(self, grad, inputs):
        # Update the gradients of encoder
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        if self.training_mode not in ["online_xrtrl"]:
            for i, layer in enumerate(self.layers[::-1]):
                name_layer = "layers_%d" % (self.n_layers - i - 1)
                grad[name_layer] = layer.update_gradients(grad[name_layer])

            return grad

        # 1) re-run forward per layer to collect h_k[t], x_k[t]
        #    we reuse our existing modules so weights/masks etc. are identical.
        inputs = self.encoder(inputs)               # inputs is the model input
        xs = [inputs]                               # x_0 is the input to the first layer. x is an array of the inputs to the different layers
        hs = []                                     # list of hidden sequences per layer
        for i, layer in enumerate(self.layers):
            inputs_after_pre_seq = layer.pre_seq(xs[-1])            # (T, N)
            h = layer.seq.get_hidden_states(inputs_after_pre_seq)   # (T, H)
            y = layer.seq.to_output(inputs_after_pre_seq, h)        # (T, N)
            x = layer.post_seq_with_cached_dropout(y) + xs[-1]      # (T, N)
            x = layer.post_skip(x)                                  # (T, N)
            hs.append(h)
            xs.append(x)

        deltas = [layer.seq.pert_hidden_states.value for layer in self.layers]  # list of (T, H)

        L = len(self.layers)
        H = self.layers[0].seq.d_hidden

        # accumulators per layer-ell
        g_lambda = [jnp.zeros((H,), dtype=jnp.complex64) for _ in range(L)]
        g_gamma  = [jnp.zeros((H,), dtype=jnp.complex64) for _ in range(L)]

        # 4) for each source ell, stream over time and propagate E^k_{theta_ell}
        for ell in range(L):
            # init E^k_t across k=ell..L as zeros at t=0
            E_lambda = [None]*L
            E_gamma  = [None]*L
            for k in range(ell, L):
                E_lambda[k] = jnp.zeros((H, H), dtype=jnp.complex64)
                E_gamma[k]  = jnp.zeros((H, H), dtype=jnp.complex64)

            # define time step update
            def time_step(carry, t):
                E_lambda, E_gamma, gL, gG = carry
                # injection at k=ell
                seq_ell = self.layers[ell].seq
                Lam_ell = jnp.diag(seq_ell.get_diag_lambda())
                b_lam = jnp.diag(hs[ell][t-1])                    # Diag(h_ell,t-1)
                b_gam = jnp.diag(seq_ell.get_B() @ xs[ell][t])  # Diag((B x)_t)

                # k = ell
                E_lambda[ell] = Lam_ell @ E_lambda[ell] + b_lam
                E_gamma[ell]  = Lam_ell @ E_gamma[ell]  + b_gam

                # propagate upwards
                for k in range(ell+1, L):
                    seq_k = self.layers[k].seq
                    Lam_k = jnp.diag(seq_k.get_diag_lambda())
                    E_lambda[k] = Lam_k @ E_lambda[k] + get_cross_layer_contribs(k, t, E_lambda)
                    E_gamma[k]  = Lam_k @ E_gamma[k]  + get_cross_layer_contribs(k, t, E_gamma)
                
                # accumulate sum_{k} {δ_k[t]^T E^k_{theta_ell,t}}
                acc_lam = 0
                acc_gam = 0
                for k in range(ell, L):
                    acc_lam = acc_lam + deltas[k][t] @ E_lambda[k]
                    acc_gam = acc_gam + deltas[k][t] @ E_gamma[k]
                gL = gL + acc_lam
                gG = gG + acc_gam
                return (E_lambda, E_gamma, gL, gG), None

            # scan over t=1..T-1 (adapt indices to how you index in your traces)
            T = xs[0].shape[0]
            carry = (E_lambda, E_gamma, g_lambda[ell], g_gamma[ell])
            (E_lambda, E_gamma, g_lambda[ell], g_gamma[ell]), _ = jax.lax.scan(time_step, carry, jnp.arange(1, T))

        # 5) map g_lambda/g_gamma into the actual parameter grads (nu, theta, gamma_log) per layer
        for ell, layer in enumerate(self.layers):
            seq = layer.seq
            # lambda path: VJP through lambda(nu, theta)
            _, pull = jax.vjp(lambda nu, th: seq.get_diag_lambda(nu=nu, theta=th), seq.nu, seq.theta)
            dnu, dtheta = pull(g_lambda[ell])
            grad["encoder"][f"layers_{ell}"]["seq"]["nu"]    = dnu
            grad["encoder"][f"layers_{ell}"]["seq"]["theta"] = dtheta

            # gamma path
            if seq.gamma_norm:
                grad["encoder"][f"layers_{ell}"]["seq"]["gamma_log"] = (g_gamma[ell] * seq.get_diag_gamma()).real

        return grad

class ClassificationModel(nn.Module):
    """
    Classificaton sequence model. This consists of the stacked encoder, pooling across the
    sequence length, a linear decoder, and a softmax operation.

    Args:
        rec             (nn.Module):    the recurrent module to use
        n_layers        (int32):        the number of SequenceLayer to stack
        dropout         (float32):      dropout rate
        d_input         (int32):        this is the feature size of the encoder inputs
        d_model         (int32):        this is the feature size of the layer inputs and outputs
                                        we usually refer to this size as H
        d_output        (int32):        the output dimension, i.e. the number of classes
        padded:         (bool):         if true: padding was used
        mode            (str):          Options: [
                                            pool: use mean pooling,
                                            last: just take the last state,
                                            none: no pooling
                                        ]
        activation      (string):       type of activation function to use
        training        (bool):         whether in training mode or not
        training_mode   (string):       type of training
        prenorm         (bool):         apply prenorm if true or postnorm if false
        multidim        (int):          number of outputs (default: 1). Greater than 1 when
                                        several classificaitons are done per timestep
    """

    rec: nn.Module
    rec_type: str
    d_input: int
    d_output: int
    d_model: int
    n_layers: int
    seq_length: int
    padded: bool
    activation: str = "gelu"
    readout: int = 0
    dropout: float = 0.2
    training: bool = True
    training_mode: str = "bptt"
    mode: str = "pool"
    prenorm: bool = False
    multidim: int = 1

    def setup(self):
        """
        Initializes the stacked encoder and a linear decoder.
        """
        self.encoder = StackedEncoder(
            rec=self.rec,
            d_input=self.d_input,
            d_model=self.d_model,
            n_layers=self.n_layers,
            seq_length=self.seq_length,
            activation=self.activation,
            readout=self.readout,
            dropout=self.dropout,
            training=self.training,
            training_mode=self.training_mode,
            prenorm=self.prenorm,
        )
        self.decoder = nn.Dense(self.d_output * self.multidim)

    def decode(self, x, var=None):
        if var is None:
            x = self.decoder(x)
        else:
            x = self.decoder.apply(var, x)
        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)
        return nn.log_softmax(x, axis=-1)

    def __call__(self, x):
        """
        Compute the size d_output log softmax output given a Txd_input input sequence.
        Args:
             x (float32): input sequence (T, d_input)
        Returns:
            output (float32): (d_output)
        """
        if self.padded:
            # x, length = x  # input consists of data and prepadded seq lens
            x = x  # removed the length of the sequence for now

        x = self.encoder(x)
        if self.mode in ["pool"]:
            # Perform mean pooling across time
            if self.padded:
                raise (
                    NotImplementedError,
                    "removed the length from the inputs, doesn't work anymore",
                )
                # x = masked_meanpool(x, length)
            else:
                x = jnp.mean(x, axis=0)

        elif self.mode in ["last"]:
            # Just take the last state
            if self.padded:
                raise NotImplementedError(
                    "Mode must be in ['pool'] for self.padded=True (for now...)"
                )
            else:
                x = x[-1]
        elif self.mode in ["none"]:
            # Do not pool at all
            # if self.padded:
            #     raise NotImplementedError(
            #         "Mode must be in ['pool'] for self.padded=True (for now...)"
            #     )
            # else:
            # HACK: This includes padded parts of the input but proper masking requires some
            # rewriting
            x = x
        elif self.mode in ["pool_st"]:

            def cumulative_mean(x):
                return jnp.cumsum(x, axis=0) / jnp.arange(1, x.shape[0] + 1)[:, None]

            x = jax.lax.stop_gradient(cumulative_mean(x) - x) + x

        else:
            raise NotImplementedError("Mode must be in ['pool', 'pool_st', 'last', 'none']")

        return self.decode(x)

    def update_gradients(self, grad, inputs):
        # Update the gradients of encoder
        # NOTE no need to update other gradients as they will be updated through spatial backprop
        if self.training_mode in ["bptt", "online_spatial", "online_reservoir"]:
            raise ValueError("Upgrade gradient should not be called for this training mode")

        grad["encoder"] = self.encoder.update_gradients(grad["encoder"], inputs)
        return grad


# Here we call vmap to parallelize across a batch of input sequences
BatchClassificationModel = nn.vmap(
    ClassificationModel,
    in_axes=0,
    out_axes=0,
    variable_axes={
        "params": None,
        "dropout": None,
        "traces": 0,
        "cache": 0,
        "perturbations": 0,
    },
    methods=["__call__", "update_gradients"],
    split_rngs={"params": False, "dropout": True},
)


def masked_meanpool(x, lengths):
    """
    Helper function to perform mean pooling across the sequence length
    when sequences have variable lengths. We only want to pool across
    the prepadded sequence length.
    Args:
         x (float32): input sequence (L, d_model)
         lengths (int32):   the original length of the sequence before padding
    Returns:
        mean pooled output sequence (float32): (d_model)
    """
    L = x.shape[0]
    mask = jnp.arange(L) < lengths
    return jnp.sum(mask[..., None] * x, axis=0) / lengths


# Here we call vmap to parallelize across a batch of input sequences
batch_masked_meanpool = jax.vmap(masked_meanpool)


# For Document matching task (e.g. AAN)
class RetrievalDecoder(nn.Module):
    """
    Defines the decoder to be used for document matching tasks,
    e.g. the AAN task. This is defined as in the S4 paper where we apply
    an MLP to a set of 4 features. The features are computed as described in
    Tay et al 2020 https://arxiv.org/pdf/2011.04006.pdf.
    Args:
        d_output    (int32):    the output dimension, i.e. the number of classes
        d_model     (int32):    this is the feature size of the layer inputs and outputs
                    we usually refer to this size as H
    """

    d_model: int
    d_output: int

    def setup(self):
        """
        Initializes 2 dense layers to be used for the MLP.
        """
        self.layer1 = nn.Dense(self.d_model)
        self.layer2 = nn.Dense(self.d_output)

    def __call__(self, x):
        """
        Computes the input to be used for the softmax function given a set of
        4 features. Note this function operates directly on the batch size.
        Args:
             x (float32): features (bsz, 4*d_model)
        Returns:
            output (float32): (bsz, d_output)
        """
        x = self.layer1(x)
        x = nn.gelu(x)
        return self.layer2(x)
