"""Network building functions and other utilities."""

import haiku as hk
import jax
import jax.numpy as jnp
from absl import flags

FLAGS = flags.FLAGS


def mlp_fn(batch,
           hidden_sizes=[300, 100],
           hidden_activation=jax.nn.relu,
           output_size=None,
           output_activation=None):
    """Standard MLP network."""

    if output_size is None:
        output_size = FLAGS.num_classes

    # Normalize input data.
    x = batch['image'].astype(jnp.float32) / 255.

    # Build body.
    layers = [hk.Flatten()]
    for size in hidden_sizes:
        layers += [hk.Linear(size), hidden_activation]

    # Build head.
    layers.append(hk.Linear(output_size))
    if output_activation is not None:
        layers.append(output_activation)

    mlp = hk.Sequential(layers)
    return mlp(x)


def get_num_params(params):
    """Returns number of parameters is the network."""
    num_params = 0
    for p in jax.tree_leaves(params):
        num_params = num_params + jnp.prod(p.shape)
    return num_params
