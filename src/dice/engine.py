from __future__ import annotations
from collections import defaultdict
from itertools import product as cartesian_product
from dice.ast_nodes import Expr
from dice.evaluator import eval_expr

Domain = list[tuple[int, float]]
PMF = dict[int, float]


def die_domain(sides: int) -> Domain:
    probability = 1.0 / sides
    return [(face, probability) for face in range(1, sides + 1)]


def build_pmf(expr: Expr, var_domains: dict[str, Domain]) -> PMF:
    if not var_domains:
        return {int(eval_expr(expr, {})): 1.0}

    variable_names = list(var_domains.keys())
    domain_lists = [var_domains[variable_name] for variable_name in variable_names]

    outcome_probs: dict[int, float] = defaultdict(float)
    for outcome in cartesian_product(*domain_lists):
        variable_values = {variable_name: face for variable_name, (face, _) in zip(variable_names, outcome)}
        joint_probability = 1.0
        for _, probability in outcome:
            joint_probability *= probability
        outcome_probs[int(eval_expr(expr, variable_values))] += joint_probability

    return dict(outcome_probs)
