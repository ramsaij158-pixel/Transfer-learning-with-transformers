import pytest

from madewithml.data import CustomPreprocessor


@pytest.fixture
def dataset_loc():
    return "datasets/dataset.csv"


@pytest.fixture
def preprocessor():
    return CustomPreprocessor()
