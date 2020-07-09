"""Dataset utilities."""

import tensorflow_datasets as tfds


def load_dataset(name, split, *, is_training, batch_size):
    """Loads the dataset as a generator of batches."""
    ds = tfds.load(name, split=split).cache().repeat()
    if is_training:
        ds = ds.shuffle(10 * batch_size, seed=0)
    ds = ds.batch(batch_size)
    return tfds.as_numpy(ds)
