"""
Utils to create daccs from dags
"""


from meshed import DAG, code_to_dag
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

#mall of inputs with keys that are the roots of the dag
Dacc(dag, input_mall)
# check gurgle
dacc.gen('chks', 'tags')-> calls dag[:chks]


#def tag_wf(wf: WaveForm, tag: Tag):
    pass
# check simple.py in examples
# dans dropdown
input: WaveForm 