from __future__ import annotations
from collections import defaultdict
import operator as _op
from typing import Callable

from dice.ast_nodes import (
    Expr,
    Const,
    Var,
    Add,
    Sub,
    Mul,
    Compare,
    IfElse,
    And,
    Or,
)
from dice.engine import Domain, PMF, build_pmf

_CMP_FNS: dict[str, Callable[[int, int], bool]] = {
    '<': _op.lt,
    '>': _op.gt,
    '<=': _op.le,
    '>=': _op.ge,
    '==': _op.eq,
    '!=': _op.ne,
}


def vars_of(expr: Expr) -> frozenset[str]:
    """Return the set of all variable names referenced anywhere in expr."""
    match expr:
        case Const():
            return frozenset()
        case Var(name=n):
            return frozenset({n})
        case (
            Add(left=left, right=right)
            | Sub(left=left, right=right)
            | Mul(left=left, right=right)
            | Compare(left=left, right=right)
            | And(left=left, right=right)
            | Or(left=left, right=right)
        ):
            return vars_of(left) | vars_of(right)
        case IfElse(condition=cond, then_branch=then_, else_branch=else_):
            return vars_of(cond) | vars_of(then_) | vars_of(else_)
        case _:
            assert False, f'Unexpected node type: {expr}'


def _pairwise(a: PMF, b: PMF, combine: Callable[[int, int], int]) -> PMF:
    """
    Apply combine(x, y) to every pair of outcomes from two independent PMFs,
    accumulating probability mass at each result value.

    For addition this is the standard discrete convolution; for other operators
    it generalises to an arbitrary output function over the joint distribution.
    """
    result: dict[int, float] = defaultdict(float)
    for va, pa in a.items():
        for vb, pb in b.items():
            result[combine(va, vb)] += pa * pb
    return dict(result)


def _and_pmf(a: PMF, b: PMF) -> PMF:
    p = sum(p for v, p in a.items() if v) * sum(p for v, p in b.items() if v)
    return {1: p, 0: 1.0 - p}


def _or_pmf(a: PMF, b: PMF) -> PMF:
    p = 1.0 - sum(p for v, p in a.items() if not v) * sum(p for v, p in b.items() if not v)
    return {1: p, 0: 1.0 - p}


def _mix(cond_pmf: PMF, then_pmf: PMF, else_pmf: PMF) -> PMF:
    """Weight then_pmf and else_pmf by the probability that cond is truthy."""
    p_true = sum(p for v, p in cond_pmf.items() if v)
    result: dict[int, float] = defaultdict(float)
    for v, p in then_pmf.items():
        result[v] += p_true * p
    for v, p in else_pmf.items():
        result[v] += (1.0 - p_true) * p
    return dict(result)


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

    def _annotate(e: Expr) -> None:
        if id(e) in orig:
            return
        match e:
            case (
                Add(left=left, right=right)
                | Sub(left=left, right=right)
                | Mul(left=left, right=right)
                | Compare(left=left, right=right)
                | And(left=left, right=right)
                | Or(left=left, right=right)
            ):
                _annotate(left)
                _annotate(right)
            case IfElse(condition=c, then_branch=t, else_branch=eb):
                _annotate(c)
                _annotate(t)
                _annotate(eb)
        orig[id(e)] = vars_of(e)

    _annotate(expr)

    def _pmf_of(e: Expr, var_set: frozenset[str]) -> PMF:
        match e:
            case Const(value=v):
                return {v: 1.0}
            case Var(name=n):
                return dict(new_domains[n])
            case _:
                return build_pmf(e, {v: new_domains[v] for v in var_set})

    def _register(pmf: PMF, consume: frozenset[str]) -> tuple[Var, frozenset[str]]:
        nonlocal counter
        # Remove the consumed variables — they are fully absorbed into the new
        # virtual variable and must not appear in the Cartesian product.
        for v in consume:
            new_domains.pop(v, None)
        name = f'_v{counter}'
        counter += 1
        new_domains[name] = list(pmf.items())
        return Var(name=name), frozenset({name})

    def _try_binary(
        ov: frozenset[str],
        left: Expr,
        right: Expr,
        forbidden: frozenset[str],
        combine: Callable[[PMF, PMF], PMF],
        rebuild: Callable[[Expr, Expr], Expr],
    ) -> tuple[Expr, frozenset[str]]:
        lv, rv = orig[id(left)], orig[id(right)]
        sl, slv = go(left,  forbidden | rv)
        sr, srv = go(right, forbidden | lv)
        if not (lv & rv) and not (ov & forbidden):
            # Both children are independent and not contested by outer siblings:
            # combine their PMFs directly, replacing both with one virtual variable.
            return _register(combine(_pmf_of(sl, slv), _pmf_of(sr, srv)), slv | srv)
        return rebuild(sl, sr), slv | srv

    def go(e: Expr, forbidden: frozenset[str]) -> tuple[Expr, frozenset[str]]:
        """
        Bottom-up simplification pass.

        `forbidden` is the union of variable sets of all sibling subtrees at
        every ancestor level. A subtree whose variables intersect `forbidden`
        cannot be collapsed: those variables are also needed elsewhere in the
        expression, so computing the subtree's PMF in isolation would discard
        their correlation with the rest and produce wrong results.
        """
        ov = orig[id(e)]

        match e:
            case Const() | Var():
                return e, ov

            case Add(left=left, right=right):
                return _try_binary(ov, left, right, forbidden,
                    combine=lambda a, b: _pairwise(a, b, _op.add),
                    rebuild=Add)

            case Sub(left=left, right=right):
                return _try_binary(ov, left, right, forbidden,
                    combine=lambda a, b: _pairwise(a, b, _op.sub),
                    rebuild=Sub)

            case Mul(left=left, right=right):
                return _try_binary(ov, left, right, forbidden,
                    combine=lambda a, b: _pairwise(a, b, _op.mul),
                    rebuild=Mul)

            case Compare(left=left, op=op, right=right):
                fn = _CMP_FNS[op]
                return _try_binary(ov, left, right, forbidden,
                    combine=lambda a, b: _pairwise(a, b, lambda x, y: int(fn(x, y))),
                    rebuild=lambda sl, sr: Compare(sl, op, sr))

            case And(left=left, right=right):
                return _try_binary(ov, left, right, forbidden,
                    combine=_and_pmf,
                    rebuild=And)

            case Or(left=left, right=right):
                return _try_binary(ov, left, right, forbidden,
                    combine=_or_pmf,
                    rebuild=Or)

            case IfElse(condition=cond, then_branch=then_, else_branch=else_):
                cv, tv, ev = orig[id(cond)], orig[id(then_)], orig[id(else_)]
                sc, scv = go(cond,  forbidden | tv | ev)
                st, stv = go(then_, forbidden | cv | ev)
                se, sev = go(else_, forbidden | cv | tv)
                if not (cv & (tv | ev)) and not (ov & forbidden):
                    # Condition is independent of both branches: compute each
                    # PMF separately and mix by P(condition is truthy).
                    return _register(
                        _mix(_pmf_of(sc, scv), _pmf_of(st, stv), _pmf_of(se, sev)),
                        scv | stv | sev,
                    )
                return IfElse(sc, st, se), scv | stv | sev

            case _:
                return e, ov

    simplified, _ = go(expr, frozenset())
    return simplified, new_domains
