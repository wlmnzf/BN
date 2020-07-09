"""Metric calculation functions."""

import jax
import jax.numpy as jnp
from absl import flags
from absl import logging

FLAGS = flags.FLAGS


@jax.jit
def accuracy(logits, targets):
    """Returns classification accuracy."""
    # Return accuracy = how many predictions match the ground truth
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)


@jax.jit
def softmax_cross_entropy_with_logits(logits, targets, num_classes=None):
    """Computes mean softmax cross entropy with logits over the batch."""
    if num_classes is None:
        num_classes = FLAGS.num_classes

    # Generate one_hot labels from index classes
    labels = jax.nn.one_hot(targets, num_classes)

    softmax_xent = -jnp.sum(labels * jax.nn.log_softmax(logits))
    softmax_xent /= labels.shape[0]

    return softmax_xent


@jax.jit
def l2_loss(params):
    """Computes the weight decay loss by penalising the norm of parameters."""
    return 0.5 * sum(jnp.sum(jnp.square(p)) for p in jax.tree_leaves(params))


def log_metrics(prefix, step, metrics):
    """Logs dictionary of metrics."""
    for name, value in metrics.items():
        logging.info('%6d | %-32s%9.3f', step, prefix + '/' + name, value)
