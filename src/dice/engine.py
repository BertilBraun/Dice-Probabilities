from __future__ import annotations
from collections import defaultdict
from itertools import product as cartesian_product
from dice.ast_nodes import Expr
from dice.evaluator import eval_expr


def die_domain(sides: int) -> list[tuple[int, float]]:
    p = 1.0 / sides
    return [(face, p) for face in range(1, sides + 1)]


def build_pmf(
    expr: Expr,
    var_domains: dict[str, list[tuple[int, float]]],
) -> dict[int, float]:
    if not var_domains:
        return {int(eval_expr(expr, {})): 1.0}

    names = list(var_domains.keys())
    domains = [var_domains[n] for n in names]

    result: dict[int, float] = defaultdict(float)
    for combo in cartesian_product(*domains):
        env = {name: face for name, (face, _) in zip(names, combo)}
        joint_p = 1.0
        for _, p in combo:
            joint_p *= p
        result[int(eval_expr(expr, env))] += joint_p

    return dict(result)
