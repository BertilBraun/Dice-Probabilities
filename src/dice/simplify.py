from __future__ import annotations

from collections import defaultdict
from typing import Callable

from dice.ast_nodes import And, BinaryNode, Const, Expr, IfElse, Or, Var
from dice.engine import Domain, PMF, build_pmf


def vars_of(expr: Expr) -> frozenset[str]:
    """Return the set of all variable names referenced anywhere in expr."""
    match expr:
        case Const():
            return frozenset()
        case Var(name=name):
            return frozenset({name})
        case BinaryNode(left=left, right=right):
            return vars_of(left) | vars_of(right)
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return vars_of(condition) | vars_of(then_branch) | vars_of(else_branch)
        case _:
            assert False, f"Unexpected node type: {expr}"


def _pairwise(a: PMF, b: PMF, combine: Callable[[int, int], int]) -> PMF:
    """
    Apply combine(x, y) to every pair of outcomes from two independent PMFs,
    accumulating probability mass at each result value.

    For addition this is the standard discrete convolution; for other operators
    it generalises to an arbitrary output function over the joint distribution.
    """
    result: dict[int, float] = defaultdict(float)
    for value_a, prob_a in a.items():
        for value_b, prob_b in b.items():
            result[combine(value_a, value_b)] += prob_a * prob_b
    return dict(result)


def _and_pmf(a: PMF, b: PMF) -> PMF:
    prob = sum(p for v, p in a.items() if v) * sum(p for v, p in b.items() if v)
    return {1: prob, 0: 1.0 - prob}


def _or_pmf(a: PMF, b: PMF) -> PMF:
    prob = 1.0 - sum(p for v, p in a.items() if not v) * sum(p for v, p in b.items() if not v)
    return {1: prob, 0: 1.0 - prob}


def _mix(cond_pmf: PMF, then_pmf: PMF, else_pmf: PMF) -> PMF:
    """Weight then_pmf and else_pmf by the probability that cond is truthy."""
    prob_true = sum(p for v, p in cond_pmf.items() if v)
    result: dict[int, float] = defaultdict(float)
    for value, prob in then_pmf.items():
        result[value] += prob_true * prob
    for value, prob in else_pmf.items():
        result[value] += (1.0 - prob_true) * prob
    return dict(result)


def _combine_pmf(node: BinaryNode, left_pmf: PMF, right_pmf: PMF) -> PMF:
    match node:
        case And():
            return _and_pmf(left_pmf, right_pmf)
        case Or():
            return _or_pmf(left_pmf, right_pmf)
        case _:
            return _pairwise(left_pmf, right_pmf, node.apply)


def simplify(expr: Expr, domains: dict[str, Domain]) -> tuple[Expr, dict[str, Domain]]:
    """
    Walk the AST bottom-up. Wherever two subtrees have disjoint variable sets
    and neither is contested by an outer sibling, collapse them into a single
    virtual variable whose domain is the combined PMF.

    Independent sums like 50d20 are collapsed via pairwise convolution
    (O(N * outcomes^2)), not Cartesian enumeration. The remaining expression
    passed to build_pmf contains only genuinely entangled variables.
    """
    new_domains = dict(domains)
    counter = 0

    # Pre-annotate every original node with its variable set, keyed by id().
    # We need the *original* var sets when checking disjointness — by the time
    # a child has been simplified it may have been replaced by a virtual Var
    # whose id() is different from the node we started with.
    orig: dict[int, frozenset[str]] = {}

    def _annotate(node: Expr) -> None:
        if id(node) in orig:
            return
        match node:
            case BinaryNode(left=left, right=right):
                _annotate(left)
                _annotate(right)
            case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
                _annotate(condition)
                _annotate(then_branch)
                _annotate(else_branch)
        orig[id(node)] = vars_of(node)

    _annotate(expr)

    def _pmf_of(node: Expr, var_set: frozenset[str]) -> PMF:
        match node:
            case Const(value=value):
                return {value: 1.0}
            case Var(name=name):
                return dict(new_domains[name])
            case _:
                return build_pmf(node, {v: new_domains[v] for v in var_set})

    def _register(pmf: PMF, consume: frozenset[str]) -> tuple[Var, frozenset[str]]:
        nonlocal counter
        # Remove the consumed variables — they are fully absorbed into the new
        # virtual variable and must not appear in the Cartesian product.
        for variable in consume:
            new_domains.pop(variable, None)
        name = f"_v{counter}"
        counter += 1
        new_domains[name] = list(pmf.items())
        return Var(name=name), frozenset({name})

    def _try_binary(
        node: BinaryNode,
        original_vars: frozenset[str],
        forbidden: frozenset[str],
    ) -> tuple[Expr, frozenset[str]]:
        left, right = node.left, node.right
        left_vars, right_vars = orig[id(left)], orig[id(right)]
        simplified_left, simplified_left_vars = go(left, forbidden | right_vars)
        simplified_right, simplified_right_vars = go(right, forbidden | left_vars)
        if not (left_vars & right_vars) and not (original_vars & forbidden):
            # Both children are independent and not contested by outer siblings:
            # combine their PMFs directly, replacing both with one virtual variable.
            combined = _combine_pmf(
                node,
                _pmf_of(simplified_left, simplified_left_vars),
                _pmf_of(simplified_right, simplified_right_vars),
            )
            return _register(combined, simplified_left_vars | simplified_right_vars)
        return node.with_children(simplified_left, simplified_right), simplified_left_vars | simplified_right_vars

    def go(node: Expr, forbidden: frozenset[str]) -> tuple[Expr, frozenset[str]]:
        """
        Bottom-up simplification pass.

        `forbidden` is the union of variable sets of all sibling subtrees at
        every ancestor level. A subtree whose variables intersect `forbidden`
        cannot be collapsed: those variables are also needed elsewhere in the
        expression, so computing the subtree's PMF in isolation would discard
        their correlation with the rest and produce wrong results.
        """
        original_vars = orig[id(node)]

        match node:
            case Const() | Var():
                return node, original_vars

            case BinaryNode():
                return _try_binary(node, original_vars, forbidden)

            case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
                cond_vars = orig[id(condition)]
                then_vars = orig[id(then_branch)]
                else_vars = orig[id(else_branch)]
                simplified_cond, simplified_cond_vars = go(condition, forbidden | then_vars | else_vars)
                simplified_then, simplified_then_vars = go(then_branch, forbidden | cond_vars | else_vars)
                simplified_else, simplified_else_vars = go(else_branch, forbidden | cond_vars | then_vars)
                if not (cond_vars & (then_vars | else_vars)) and not (original_vars & forbidden):
                    # Condition is independent of both branches: compute each
                    # PMF separately and mix by P(condition is truthy).
                    return _register(
                        _mix(
                            _pmf_of(simplified_cond, simplified_cond_vars),
                            _pmf_of(simplified_then, simplified_then_vars),
                            _pmf_of(simplified_else, simplified_else_vars),
                        ),
                        simplified_cond_vars | simplified_then_vars | simplified_else_vars,
                    )
                return (
                    IfElse(simplified_cond, simplified_then, simplified_else),
                    simplified_cond_vars | simplified_then_vars | simplified_else_vars,
                )

            case _:
                return node, original_vars

    simplified, _ = go(expr, frozenset())
    return simplified, new_domains
