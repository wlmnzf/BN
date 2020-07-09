"""Training utilities."""

import jax

from humblesl import metric as metric_utils


def loop(params,
         opt_state,
         train_dataset,
         sgd_update,
         n_steps=None,
         log_interval=None,
         train_eval_dataset=None,
         test_eval_dataset=None,
         calculate_metrics=None):
    """Train/eval loop."""
    step = 0
    while True:
        if log_interval is not None and step % log_interval == 0:
            # Periodically evaluate classification accuracy on train/test sets.
            metrics = dict()
            if train_eval_dataset is not None:
                metrics['train'] = calculate_metrics(
                    params, next(train_eval_dataset))
            if test_eval_dataset is not None:
                metrics['test'] = calculate_metrics(
                    params, next(test_eval_dataset))

            metrics = jax.device_get(metrics)
            for prefix, metric in metrics.items():
                metric_utils.log_metrics(prefix, step, metric)

        if n_steps is not None and step == n_steps:
            break

        # Do SGD on a batch of training examples.
        params, opt_state = sgd_update(params, opt_state, next(train_dataset))

        step += 1

    return params, opt_state
