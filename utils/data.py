"""Dataset utilities."""

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


def load_dataset(split, *, is_training, batch_size):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load('mnist:3.*.*', split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)
