import pytest
from functional_dacc import is_subset


def test_dummy():
    assert True


def test_is_subset():
    iter1 = ("a", "b", "c")
    iter2 = "bc"
    iter3 = ["b", "c", "d"]
    assert not is_subset(iter1, iter3)
