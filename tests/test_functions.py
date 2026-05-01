import pytest
from dice.ast_nodes import Call, Const, Var, Die, Add, Compare
from dice.parser import parse, parse_expr
from dice.engine import build_pmf


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def pmf_sum(pmf: dict) -> float:
    return sum(pmf.values())


def _run(text: str) -> dict[int, float]:
    expr, domains = parse(text)
    return build_pmf(expr, domains)


# ── AST: Call node ────────────────────────────────────────────────────────────

class TestCallNode:
    def test_call_stores_name_and_args(self):
        node = Call("f", (Const(1), Var("X")))
        assert node.name == "f"
        assert node.args == (Const(1), Var("X"))

    def test_call_equality(self):
        assert Call("f", (Const(1),)) == Call("f", (Const(1),))
        assert Call("f", (Const(1),)) != Call("g", (Const(1),))

    def test_call_zero_args(self):
        node = Call("f", ())
        assert node.args == ()

    def test_parse_expr_produces_call_node(self):
        node = parse_expr("f(X, 3)")
        assert isinstance(node, Call)
        assert node.name == "f"
        assert node.args == (Var("X"), Const(3))


# ── Basic function definitions ────────────────────────────────────────────────

class TestBasicFunctions:
    def test_identity_function(self):
        pmf = _run("f(X): X\nf(d6)")
        assert set(pmf.keys()) == {1, 2, 3, 4, 5, 6}
        assert approx(pmf_sum(pmf), 1.0)

    def test_constant_offset_function(self):
        pmf = _run("f(X): X + 1\nf(d6)")
        assert set(pmf.keys()) == {2, 3, 4, 5, 6, 7}
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 4.5)

    def test_two_parameter_function(self):
        pmf = _run("add(X, Y): X + Y\nadd(d6, d6)")
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)

    def test_semicolon_as_separator(self):
        pmf = _run("f(X): X + 1; f(d6)")
        assert set(pmf.keys()) == {2, 3, 4, 5, 6, 7}

    def test_two_definitions_one_line(self):
        pmf = _run("double(X): X * 2; triple(X): X * 3; double(d6)")
        assert set(pmf.keys()) == {2, 4, 6, 8, 10, 12}

    def test_function_with_conditional(self):
        pmf = _run("threshold(X): X > 3 -> X | 0\nthreshold(d6)")
        assert approx(pmf[0], 0.5)
        assert approx(pmf[4], 1 / 6)

    def test_function_composition(self):
        pmf = _run("add(X, Y): X + Y\nsum3(X, Y, Z): add(add(X, Y), Z)\nsum3(d6, d6, d6)")
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 10.5)


# ── Inline dice in function bodies ───────────────────────────────────────────

class TestInlineDiceInFunctions:
    def test_function_with_inline_die(self):
        # a(X): X + d6 — each call gets a fresh d6
        pmf = _run("a(X): X + d6\na(d6)")
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)   # E[d6] + E[d6]
        assert approx(pmf_sum(pmf), 1.0)

    def test_inline_die_is_independent_per_call(self):
        # b(X): a(X) + a(X) where a(X): X + d6
        # Expands to: (_d0 + _d1) + (_d0 + _d2) = 2*_d0 + _d1 + _d2  (3 dice)
        pmf = _run("a(X): X + d6\nb(X): a(X) + a(X)\nb(d6)")
        assert approx(pmf_sum(pmf), 1.0)
        # 2*d6 + d6 + d6: min = 4, max = 24
        assert min(pmf.keys()) == 4
        assert max(pmf.keys()) == 24
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 14.0)  # 2*3.5 + 3.5 + 3.5


# ── Parameter bound once per call site ───────────────────────────────────────

class TestParameterBinding:
    def test_parameter_used_twice_is_same_roll(self):
        # double(X): X + X — X is one die roll, doubled; NOT two independent rolls
        pmf = _run("double(X): X + X\ndouble(d6)")
        # Must be even numbers 2,4,...,12
        assert set(pmf.keys()) == {2, 4, 6, 8, 10, 12}
        # Each with probability 1/6 (one roll, doubled)
        for outcome in pmf:
            assert approx(pmf[outcome], 1 / 6)

    def test_parameter_bound_once_across_function_calls(self):
        # a(X): X > 3 -> 1 | 0
        # b(X): a(X) + a(X)
        # b(d6): X is bound once; both a(X) calls see the same roll
        # → outcome is always 0 or 2, never 1
        pmf = _run("a(X): X > 3 -> 1 | 0\nb(X): a(X) + a(X)\nb(d6)")
        assert approx(pmf.get(0, 0.0), 0.5)
        assert approx(pmf.get(2, 0.0), 0.5)
        assert pmf.get(1, 0.0) == 0.0

    def test_two_independent_argument_dice(self):
        # f(X, Y): X + Y with f(d6, d6) — two independent dice
        pmf = _run("f(X, Y): X + Y\nf(d6, d6)")
        assert min(pmf.keys()) == 2
        assert max(pmf.keys()) == 12
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)


# ── Multi-function programs ───────────────────────────────────────────────────

class TestMultiFunctionPrograms:
    def test_user_example_a_b(self):
        # From the spec: a(X): X + d6; b(X): a(X) + a(X)
        # Two independent d6s inside a(), one shared argument d6
        pmf = _run("a(X): X + d6\nb(X): a(X) + a(X)\nb(d6)")
        assert approx(pmf_sum(pmf), 1.0)
        assert all(p >= 0 for p in pmf.values())
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 14.0)

    def test_three_level_composition(self):
        pmf = _run(
            "bonus(X): X + d6\n"
            "attack(X): bonus(X) + bonus(X)\n"
            "advantage(X, Y): X > Y -> X | Y\n"
            "advantage(attack(d6), attack(d6))"
        )
        assert approx(pmf_sum(pmf), 1.0)
        assert all(p >= 0 for p in pmf.values())


# ── Error cases ───────────────────────────────────────────────────────────────

class TestFunctionErrors:
    def test_undefined_function_raises(self):
        with pytest.raises(NameError, match="Undefined function"):
            parse("f(d6)")

    def test_wrong_argument_count_raises(self):
        with pytest.raises(ValueError, match="argument"):
            parse("f(X, Y): X + Y\nf(d6)")

    def test_recursive_function_raises(self):
        with pytest.raises(RecursionError):
            parse("f(X): f(X)\nf(d6)")

    def test_mutually_recursive_functions_raise(self):
        with pytest.raises(RecursionError):
            parse("a(X): b(X)\nb(X): a(X)\na(d6)")

    def test_duplicate_function_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            parse("f(X): X\nf(X): X + 1\nf(d6)")

    def test_multiple_final_expressions_raises(self):
        with pytest.raises(SyntaxError):
            parse("d6\nd6")
