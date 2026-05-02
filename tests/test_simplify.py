"""
Tests for simplify.py: vars_of extraction and the simplification pass.

vars_of tests verify the variable-set annotation on raw AST nodes.
Simplify tests verify both structural properties (how many variables
remain after simplification) and semantic correctness (same PMF as
the unoptimised build_pmf path).
"""

import pytest
from dice.ast_nodes import Add, Sub, Mul, Compare, IfElse, And, Or, Const, Var
from dice.engine import build_pmf, die_domain
from dice.parser import parse
from dice.simplify import simplify, vars_of


# ── Helpers ───────────────────────────────────────────────────────────────────

def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def pmf_eq(p: dict, q: dict, tol: float = 1e-9) -> bool:
    return set(p) == set(q) and all(approx(p[k], q[k], tol) for k in p)


def run(text: str) -> dict[int, float]:
    expr, domains = parse(text)
    return build_pmf(expr, domains)


def run_simplified(text: str) -> tuple[dict[int, float], dict]:
    expr, domains = parse(text)
    expr, domains = simplify(expr, domains)
    return build_pmf(expr, domains), domains


# ── vars_of ───────────────────────────────────────────────────────────────────

class TestVarsOf:
    def test_const(self):
        assert vars_of(Const(5)) == frozenset()

    def test_var(self):
        assert vars_of(Var('x')) == frozenset({'x'})

    def test_add_two_vars(self):
        assert vars_of(Add(Var('x'), Var('y'))) == frozenset({'x', 'y'})

    def test_add_same_var_deduped(self):
        assert vars_of(Add(Var('x'), Var('x'))) == frozenset({'x'})

    def test_sub(self):
        assert vars_of(Sub(Var('a'), Const(3))) == frozenset({'a'})

    def test_mul(self):
        assert vars_of(Mul(Var('a'), Var('b'))) == frozenset({'a', 'b'})

    def test_compare_with_const(self):
        assert vars_of(Compare(Var('x'), '>', Const(3))) == frozenset({'x'})

    def test_compare_two_vars(self):
        assert vars_of(Compare(Var('x'), '<', Var('y'))) == frozenset({'x', 'y'})

    def test_and(self):
        assert vars_of(And(Var('a'), Var('b'))) == frozenset({'a', 'b'})

    def test_or(self):
        assert vars_of(Or(Var('a'), Var('b'))) == frozenset({'a', 'b'})

    def test_ifelse_all_different(self):
        node = IfElse(Var('c'), Var('t'), Var('e'))
        assert vars_of(node) == frozenset({'c', 't', 'e'})

    def test_ifelse_shared_condition_and_branch(self):
        node = IfElse(Compare(Var('x'), '>', Const(3)), Var('x'), Const(0))
        assert vars_of(node) == frozenset({'x'})

    def test_nested(self):
        # (x + y) > (z - 1)
        node = Compare(Add(Var('x'), Var('y')), '>', Sub(Var('z'), Const(1)))
        assert vars_of(node) == frozenset({'x', 'y', 'z'})

    def test_deeply_nested(self):
        # (a + b) -> (c * d) | e
        node = IfElse(Var('a'), Mul(Var('c'), Var('d')), Var('e'))
        assert vars_of(node) == frozenset({'a', 'c', 'd', 'e'})


# ── Simplify: structural (domain count after simplification) ──────────────────

class TestSimplifyStructure:
    def test_two_independent_dice_collapse_to_one(self):
        expr, domains = parse('d6 + d6')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1

    def test_three_independent_dice_collapse_to_one(self):
        expr, domains = parse('d6 + d6 + d6')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1

    def test_ten_independent_dice_collapse_to_one(self):
        expr, domains = parse(' + '.join('d6' for _ in range(10)))
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1

    def test_shared_variable_not_collapsed(self):
        # X appears twice — cannot be treated as two independent variables.
        # After simplification the only variable should still be _d0 (X's die).
        expr, domains = parse('f(X): X + X; f(d6)')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1
        # The remaining domain must be X's original d6 range, not a virtual PMF
        # with values reaching 12 (which would only appear if X were duplicated).
        values = {v for v, _ in list(domains.values())[0]}
        assert max(values) == 6  # single d6 face, not 2–12

    def test_independent_conditional_collapses_to_one(self):
        # Condition die and branch dice are all different variables.
        expr, domains = parse('d6 > 5 -> d20 + d20 | 0')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1

    def test_shared_condition_and_branch_not_collapsed(self):
        # X appears in both condition and then-branch — must enumerate.
        expr, domains = parse('f(X): X > 3 -> X | 0; f(d6)')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1
        values = {v for v, _ in list(domains.values())[0]}
        assert max(values) == 6  # X's d6 domain, not a collapsed PMF

    def test_independent_halves_collapse_to_two_then_one(self):
        # (3d6) + (3d6): each half collapses independently, then outer + collapses.
        expr, domains = parse('(d6 + d6 + d6) + (d6 + d6 + d6)')
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1

    def test_fifty_d20_collapse_to_one(self):
        expr, domains = parse(' + '.join('d20' for _ in range(50)))
        expr, domains = simplify(expr, domains)
        assert len(domains) == 1


# ── Simplify: semantic (PMF correctness) ──────────────────────────────────────
# Every expression here is also checked against the unoptimised path.

class TestSimplifyCorrectness:
    @pytest.mark.parametrize('text', [
        'd6 + d6',
        'd6 - d6',
        'd6 * d4',
        'd6 > 3 -> d4 | 0',
        'd6 > 5 -> d20 + d20 | 0',
        'f(X, Y): X + Y > 6 -> 2*X | X - Y; f(d6, d6)',
        'f(X): X > 3 -> X | 0; f(d6)',         # shared variable — no collapse
        'f(X): X + X; f(d6)',                   # shared variable — no collapse
        'd6 + d6 + d6 + d6',
        '(d6 + d6) + (d6 + d6)',
        'd6 > 3 && d6 > 3',
        'd6 > 3 || d6 > 3',
    ])
    def test_pmf_matches_unoptimised(self, text):
        expected = run(text)
        got, _ = run_simplified(text)
        assert pmf_eq(got, expected), f'PMF mismatch for {text!r}'

    def test_independent_conditional_p_zero(self):
        # d6 > 5 → only face 6 triggers; P(0) = 5/6
        pmf, _ = run_simplified('d6 > 5 -> d20 | 0')
        assert approx(pmf.get(0, 0.0), 5 / 6)

    def test_fifty_d20_p_zero(self):
        text = 'd6 > 5 -> ' + ' + '.join('d20' for _ in range(50)) + ' | 0'
        pmf, domains = run_simplified(text)
        assert len(domains) == 1
        assert approx(pmf.get(0, 0.0), 5 / 6)

    def test_sum_axioms(self):
        for n in range(2, 8):
            pmf, _ = run_simplified(' + '.join('d6' for _ in range(n)))
            assert approx(sum(pmf.values()), 1.0)
            assert all(p >= 0 for p in pmf.values())
            assert min(pmf) == n
            assert max(pmf) == 6 * n
