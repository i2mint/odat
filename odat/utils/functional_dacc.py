"""
Utils to create daccs from dags
"""
from meshed import DAG
from typing import Iterable


def get_intermediate_nodes(dag: DAG):
    vnodes = set(dag.var_nodes)
    roots = set(dag.roots)
    result = vnodes - roots

    return result


def is_subset(iterable1: Iterable, iterable2: Iterable):
    return all(item in iterable2 for item in iterable1)


def validate_input_dacc_gen(input_tuple, intermediate_nodes):
    return is_subset(input_tuple, intermediate_nodes)
