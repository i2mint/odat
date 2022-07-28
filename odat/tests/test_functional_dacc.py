import pytest
from pytest import fixture
from odat.utils.functional_dacc import (
    is_subset,
    get_intermediate_nodes,
)
from meshed import DAG, code_to_dag


@code_to_dag
def example_dag():
    wfs = load(source)
    fvs = chunker(wfs, chk_size)


def test_dummy():
    assert True


def test_is_subset():
    iter1 = ("a", "b", "c")
    iter2 = "bc"
    iter3 = ["b", "c", "d"]
    assert not is_subset(iter1, iter3)
    assert is_subset(iter2, iter3)


def test_get_intermediate_nodes():

    expected = ["wfs", "fvs"]
    result = get_intermediate_nodes(example_dag)

    assert set(expected) == set(result)
