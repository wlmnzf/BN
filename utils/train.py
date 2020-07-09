"""Training utilities."""

import jax
from absl import flags

from utils import metric as metric_utils

FLAGS = flags.FLAGS


def loop(params,
         opt_state,
         calculate_metrics,
         sgd_update,
         train_dataset,
         train_eval_dataset,
         test_eval_dataset,
         n_steps,
         log_interval):
    """Train/eval loop."""
    step = 0
    while True:
        if step % log_interval == 0:
            # Periodically evaluate classification accuracy on train/test sets.
            train_metrics = calculate_metrics(params, next(train_eval_dataset))
            test_metrics = calculate_metrics(params, next(test_eval_dataset))
            train_metrics, test_metrics = jax.device_get(
                (train_metrics, test_metrics))
            metric_utils.log_metrics('train', step, train_metrics)
            metric_utils.log_metrics('test', step, test_metrics)

        if step == n_steps:
            break

        # Do SGD on a batch of training examples.
        params, opt_state = sgd_update(params, opt_state, next(train_dataset))

        step += 1

    return params, opt_state
