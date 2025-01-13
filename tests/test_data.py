from torch.utils.data import Dataset
import torch
import pytest

from mnist.data import corrupt_mnist

import warnings
warnings.filterwarnings("ignore")

import os.path
@pytest.mark.skipif(not os.path.exists("data/raw"), reason="Data files not found")
def test_my_dataset():
    """Test the MyDataset class."""
    train_set, test_set = corrupt_mnist()
    assert len(train_set) == 30000, "Train set is supposed to have 30000 examples"
    assert len(test_set) == 5000, "Test set is supposed to have 5000 examples"

    for dataset in [train_set, test_set]:
        for x, y in dataset:
            assert x.shape == (1, 28, 28), "Train data is not the correct shape (1,28,28)"
            assert y in range(10), "Test data is not the correct shape (1,28,28)"

    train_labels = torch.unique(train_set.tensors[1])
    assert (train_labels == torch.arange(0,10)).all(), "Not all labels are represented in train data"
    test_labels = torch.unique(test_set.tensors[1])
    assert (test_labels == torch.arange(0,10)).all(), "Not all labels are represented in test data"

