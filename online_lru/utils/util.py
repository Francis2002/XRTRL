from typing import Sequence
import argparse
from jax import lax
import jax.numpy as jnp

def is_list(x):
	"""
	From AG:  this is usually used in a pattern where it's turned into a list, so can just do that here
	:param x:
	:return:
	"""
	return isinstance(x, Sequence) and not isinstance(x, str)


def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')
	
# ------------------- for single timestep replay processing ------------------- #

def _take_t(x, t):
    # x: (T, ...), t: int32[] tracer
    return lax.dynamic_index_in_dim(x, t, keepdims=False)

def _take_t_minus_1_or_zero(vecT, t, length_one, dtype):
    # vecT: (T, H) -> returns (H,)
    # returns vecT[t-1] if t>0 else zeros(H,)
    return lax.cond(
        t > 0,
        lambda tt: lax.dynamic_index_in_dim(vecT, tt - 1, keepdims=False),
        lambda tt: jnp.zeros((length_one,), dtype=dtype),
        t,
    )
