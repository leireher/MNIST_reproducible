from mnist.model import MyModel
import torch
import pytest

import warnings
warnings.filterwarnings("ignore")

def test_model_output():
    dummy_data = torch.rand(1,1,28,28)
    model = MyModel()
    out = model(dummy_data)

    assert out.shape == (1, 10), "Output of the model is not of shape (1,10)"

@pytest.mark.parametrize("test_input,expected", [(2,2), (1,1), (3,3)])
def test_dummy(test_input, expected):
    assert test_input == expected

@pytest.mark.parametrize("batch_size", [32, 64])
def test_batch(batch_size):
    model = MyModel()
    x = torch.rand(batch_size, 1, 28, 28)
    y = model(x)
    assert y.shape == (batch_size, 10)