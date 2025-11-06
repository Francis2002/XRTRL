import jax.numpy as jnp

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

