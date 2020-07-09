"""Simple MLP classifier on MNIST."""

import haiku as hk
import jax
from jax.experimental import optix
from absl import app
from absl import flags
from absl import logging

from utils import data as data_utils
from utils import metric as metric_utils
from utils import network as network_utils
from utils import train as train_utils

FLAGS = flags.FLAGS
FLAGS.showprefixforinfo = False

flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-4, 'Weight decay to apply to parameters.')
flags.DEFINE_integer('num_classes', 10, 'Number of classes.')
flags.DEFINE_integer('n_train_steps', 5000, 'Number of training steps.')
flags.DEFINE_integer('log_interval', 1000, 'Training logging interval.')


def main(argv):
    del argv

    # Make datasets for train and test.
    train_dataset = data_utils.load_dataset(
        'train', is_training=True, batch_size=1000)
    train_eval_dataset = data_utils.load_dataset(
        'train', is_training=False, batch_size=10000)
    test_eval_dataset = data_utils.load_dataset(
        'test', is_training=False, batch_size=10000)

    # Draw a data batch and log shapes.
    batch = next(train_dataset)
    logging.info("Image batch shape: %s", batch['image'].shape)
    logging.info("Label batch shape: %s", batch['label'].shape)

    # Since we don't store additional state statistics, e.g. needed in
    # batch norm, we use `hk.transform`. When we use batch_norm, we will use
    # `hk.transform_with_state`.
    net = hk.without_apply_rng(hk.transform(
        network_utils.mlp_fn,
        apply_rng=True
    ))

    # Initialize model
    params = net.init(jax.random.PRNGKey(42), batch)
    logging.info('Total number of parameters: %d',
                 network_utils.get_num_params(params))

    # Define and initialize optimizer.
    opt = optix.adam(FLAGS.lr)
    opt_state = opt.init(params)

    def loss(params, batch):
        """Compute the loss of the network, including L2 for regularization."""
        # Get network predictions
        logits = net.apply(params, batch)
        # Compute mean softmax cross entropy over the batch
        softmax_xent = metric_utils.softmax_cross_entropy_with_logits(
            logits, batch)
        # Compute the weight decay loss by penalising the norm of parameters
        l2_loss = metric_utils.l2_loss(params)
        return softmax_xent + FLAGS.weight_decay * l2_loss

    @jax.jit
    def sgd_update(params, opt_state, batch):
        """Learning rule (stochastic gradient descent)."""
        # Use jax transformation `grad` to compute gradients;
        # it expects the prameters of the model and the input batch
        grads = jax.grad(loss)(params, batch)
        # Compute parameters updates based on gradients and optimiser state
        updates, opt_state = opt.update(grads, opt_state)
        # Apply updates to parameters
        new_params = optix.apply_updates(params, updates)
        return new_params, opt_state

    def calculate_metrics(params, batch):
        """Calculates accuracy."""
        logits = net.apply(params, batch)
        return {
            'accuracy': metric_utils.accuracy(logits, batch),
            'loss': loss(params, batch),
        }

    # Train!
    train_utils.loop(params,
                     opt_state,
                     calculate_metrics,
                     sgd_update,
                     train_dataset,
                     train_eval_dataset,
                     test_eval_dataset,
                     n_steps=FLAGS.n_train_steps,
                     log_interval=FLAGS.log_interval)


if __name__ == '__main__':
    app.run(main)
