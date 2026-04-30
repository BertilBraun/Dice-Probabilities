import pytest
from dice.ast_nodes import Const, Var, Die, Add, Mul, Sub, Compare, IfElse, And, Or
from dice.parser import parse, parse_expr
from dice.engine import die_domain


class TestParseExprAtoms:
    def test_integer_constant(self):
        assert parse_expr("42") == Const(42)

    def test_zero(self):
        assert parse_expr("0") == Const(0)

    def test_single_variable(self):
        assert parse_expr("X") == Var("X")

    def test_multi_char_variable(self):
        assert parse_expr("Roll") == Var("Roll")

    def test_parenthesised_const(self):
        assert parse_expr("(7)") == Const(7)

    def test_parenthesised_var(self):
        assert parse_expr("(X)") == Var("X")


class TestParseExprArithmetic:
    def test_add_two_consts(self):
        assert parse_expr("1 + 2") == Add(Const(1), Const(2))

    def test_sub_two_consts(self):
        assert parse_expr("5 - 3") == Sub(Const(5), Const(3))

    def test_mul_two_consts(self):
        assert parse_expr("3 * 4") == Mul(Const(3), Const(4))

    def test_add_is_left_associative(self):
        assert parse_expr("1 + 2 + 3") == Add(Add(Const(1), Const(2)), Const(3))

    def test_sub_is_left_associative(self):
        assert parse_expr("5 - 2 - 1") == Sub(Sub(Const(5), Const(2)), Const(1))

    def test_mul_binds_tighter_than_add(self):
        assert parse_expr("2*X + 1") == Add(Mul(Const(2), Var("X")), Const(1))

    def test_mul_binds_tighter_than_add_right(self):
        assert parse_expr("1 + 2*X") == Add(Const(1), Mul(Const(2), Var("X")))

    def test_parens_override_precedence(self):
        assert parse_expr("2*(X + 1)") == Mul(Const(2), Add(Var("X"), Const(1)))

    def test_no_spaces(self):
        assert parse_expr("X+Y") == Add(Var("X"), Var("Y"))


class TestParseExprCompare:
    @pytest.mark.parametrize("op", [">", "<", ">=", "<=", "==", "!="])
    def test_all_comparison_operators(self, op):
        node = parse_expr(f"X {op} 3")
        assert isinstance(node, Compare)
        assert node.op == op
        assert node.left == Var("X")
        assert node.right == Const(3)

    def test_compare_with_arithmetic_operands(self):
        node = parse_expr("X + Y > 6")
        assert isinstance(node, Compare)
        assert node.left == Add(Var("X"), Var("Y"))
        assert node.right == Const(6)
        assert node.op == ">"

    def test_compare_with_mul_operands(self):
        node = parse_expr("2*X >= Y")
        assert isinstance(node, Compare)
        assert node.left == Mul(Const(2), Var("X"))
        assert node.right == Var("Y")


class TestParseExprIfElse:
    def test_simple_ifelse(self):
        node = parse_expr("X > 3 -> X | 0")
        assert isinstance(node, IfElse)
        assert node.condition == Compare(Var("X"), ">", Const(3))
        assert node.then_branch == Var("X")
        assert node.else_branch == Const(0)

    def test_ifelse_with_arithmetic_branches(self):
        node = parse_expr("X > 3 -> X*2 | 0")
        assert node.then_branch == Mul(Var("X"), Const(2))
        assert node.else_branch == Const(0)

    def test_complex_two_var_ifelse(self):
        node = parse_expr("X + Y > 6 -> X*2 + Y | X + 2*Y")
        assert isinstance(node, IfElse)
        assert node.condition == Compare(Add(Var("X"), Var("Y")), ">", Const(6))
        assert node.then_branch == Add(Mul(Var("X"), Const(2)), Var("Y"))
        assert node.else_branch == Add(Var("X"), Mul(Const(2), Var("Y")))

    def test_arrow_does_not_conflict_with_subtraction(self):
        # "X - 1 -> X | 0" should not be confused by "->" tokenisation
        node = parse_expr("X - 1 > 0 -> X | 0")
        assert isinstance(node, IfElse)
        assert node.condition == Compare(Sub(Var("X"), Const(1)), ">", Const(0))


class TestParseFullExpression:
    def test_single_var_identity(self):
        expr, domains = parse("e(X): X; e(d6)")
        assert expr == Var("X")
        assert list(domains.keys()) == ["X"]
        assert len(domains["X"]) == 6

    def test_single_var_threshold(self):
        expr, domains = parse("e(X): X > 3 -> X | 0; e(d6)")
        assert isinstance(expr, IfElse)
        assert "X" in domains
        assert len(domains["X"]) == 6

    def test_two_vars_bound_to_d6(self):
        expr, domains = parse("e(X, Y): X + Y > 6 -> X*2 + Y | X + 2*Y; e(d6, d6)")
        assert set(domains.keys()) == {"X", "Y"}
        assert len(domains["X"]) == 6
        assert len(domains["Y"]) == 6

    def test_variable_order_preserved(self):
        _, domains = parse("e(X, Y): X + Y; e(d6, d6)")
        assert list(domains.keys()) == ["X", "Y"]

    def test_mixed_die_sizes(self):
        _, domains = parse("e(A, B): A + B; e(d4, d20)")
        assert len(domains["A"]) == 4
        assert len(domains["B"]) == 20

    def test_domain_probabilities_sum_to_1(self):
        _, domains = parse("e(X): X; e(d6)")
        total = sum(p for _, p in domains["X"])
        assert abs(total - 1.0) < 1e-9

    def test_domain_faces_are_correct(self):
        _, domains = parse("e(X): X; e(d6)")
        faces = [v for v, _ in domains["X"]]
        assert faces == [1, 2, 3, 4, 5, 6]

    def test_d20_domain(self):
        _, domains = parse("e(X): X; e(d20)")
        faces = [v for v, _ in domains["X"]]
        assert faces == list(range(1, 21))


class TestParseErrors:
    def test_mismatched_var_die_count_raises(self):
        with pytest.raises(ValueError, match="(?i)var"):
            parse("e(X, Y): X + Y; e(d6)")

    def test_unmatched_paren_raises(self):
        with pytest.raises(SyntaxError):
            parse_expr("(X + 3")

    def test_empty_expression_raises(self):
        with pytest.raises((SyntaxError, ValueError)):
            parse_expr("")


# ── Boolean operators ─────────────────────────────────────────────────────────

class TestParseAnd:
    def test_simple_and(self):
        node = parse_expr("X > 2 && Y > 3")
        assert node == And(Compare(Var("X"), ">", Const(2)), Compare(Var("Y"), ">", Const(3)))

    def test_and_left_associative(self):
        node = parse_expr("X > 0 && Y > 0 && Z > 0")
        expected = And(And(Compare(Var("X"), ">", Const(0)),
                           Compare(Var("Y"), ">", Const(0))),
                       Compare(Var("Z"), ">", Const(0)))
        assert node == expected

    def test_and_in_ifelse_condition(self):
        node = parse_expr("X > 2 && Y > 3 -> X + Y | 0")
        assert isinstance(node, IfElse)
        assert node.condition == And(
            Compare(Var("X"), ">", Const(2)),
            Compare(Var("Y"), ">", Const(3)),
        )
        assert node.then_branch == Add(Var("X"), Var("Y"))
        assert node.else_branch == Const(0)


class TestParseOr:
    def test_simple_or(self):
        node = parse_expr("X > 2 || Y > 3")
        assert node == Or(Compare(Var("X"), ">", Const(2)), Compare(Var("Y"), ">", Const(3)))

    def test_or_left_associative(self):
        node = parse_expr("X > 0 || Y > 0 || Z > 0")
        expected = Or(Or(Compare(Var("X"), ">", Const(0)),
                         Compare(Var("Y"), ">", Const(0))),
                      Compare(Var("Z"), ">", Const(0)))
        assert node == expected

    def test_or_in_ifelse_condition(self):
        node = parse_expr("X > 4 || Y > 4 -> X + Y | 0")
        assert isinstance(node, IfElse)
        assert node.condition == Or(
            Compare(Var("X"), ">", Const(4)),
            Compare(Var("Y"), ">", Const(4)),
        )


class TestBooleanPrecedence:
    def test_and_binds_tighter_than_or(self):
        # A || B && C  →  A || (B && C)
        node = parse_expr("X > 0 || Y > 0 && Z > 0")
        assert isinstance(node, Or)
        assert node.left == Compare(Var("X"), ">", Const(0))
        assert node.right == And(Compare(Var("Y"), ">", Const(0)),
                                 Compare(Var("Z"), ">", Const(0)))

    def test_compare_binds_tighter_than_and(self):
        # X + 1 > 2 && Y < 5  →  (X+1>2) && (Y<5)
        node = parse_expr("X + 1 > 2 && Y < 5")
        assert isinstance(node, And)
        assert node.left == Compare(Add(Var("X"), Const(1)), ">", Const(2))
        assert node.right == Compare(Var("Y"), "<", Const(5))

    def test_pipe_in_ifelse_not_confused_with_or(self):
        # X > 0 -> Y || Z > 0 | 0  →  IfElse(X>0, Y||Z>0, 0)
        node = parse_expr("X > 0 -> Y > 0 || Z > 0 | 0")
        assert isinstance(node, IfElse)
        assert node.then_branch == Or(Compare(Var("Y"), ">", Const(0)),
                                      Compare(Var("Z"), ">", Const(0)))
        assert node.else_branch == Const(0)


# ── Inline dice in parse_expr (returns raw Die nodes) ────────────────────────

class TestParseInlineDiceRaw:
    def test_bare_die_is_die_node(self):
        assert parse_expr("d6") == Die(6)

    def test_d20(self):
        assert parse_expr("d20") == Die(20)

    def test_die_in_add(self):
        assert parse_expr("d6 + 3") == Add(Die(6), Const(3))

    def test_two_dice_add(self):
        # Two structurally equal Die(6) nodes — distinct occurrences
        node = parse_expr("d6 + d6")
        assert node == Add(Die(6), Die(6))

    def test_die_in_condition(self):
        node = parse_expr("d6 > 3")
        assert node == Compare(Die(6), ">", Const(3))

    def test_die_in_ifelse(self):
        node = parse_expr("d6 > 3 -> d6 | 0")
        assert isinstance(node, IfElse)
        assert node.condition == Compare(Die(6), ">", Const(3))
        assert node.then_branch == Die(6)
        assert node.else_branch == Const(0)

    def test_mixed_named_and_inline(self):
        node = parse_expr("X + d6")
        assert node == Add(Var("X"), Die(6))


# ── parse() with inline dice (Die nodes extracted to Vars) ───────────────────

class TestParseWithDiceExtraction:
    def test_bare_d6_gives_one_domain(self):
        expr, domains = parse("d6")
        assert len(domains) == 1
        d = list(domains.values())[0]
        assert d == die_domain(6)

    def test_two_d6_give_two_domains(self):
        expr, domains = parse("d6 + d6")
        assert len(domains) == 2
        for d in domains.values():
            assert d == die_domain(6)

    def test_two_d6_become_distinct_vars(self):
        expr, domains = parse("d6 + d6")
        names = list(domains.keys())
        assert names[0] != names[1]

    def test_mixed_dice_sizes(self):
        _, domains = parse("d6 + d20")
        sizes = sorted(len(d) for d in domains.values())
        assert sizes == [6, 20]

    def test_die_in_condition_gives_two_domains(self):
        # d6 > 3 -> d4 | 0  — two dice, independent
        _, domains = parse("d6 > 3 -> d4 | 0")
        assert len(domains) == 2
        sizes = sorted(len(d) for d in domains.values())
        assert sizes == [4, 6]

    def test_named_form_with_inline_die(self):
        # e(X): X + d6; e(d6)  — named X plus anonymous die
        expr, domains = parse("e(X): X + d6; e(d6)")
        assert len(domains) == 2
        assert "X" in domains

    def test_domain_names_are_auto_generated(self):
        _, domains = parse("d6 + d6")
        for name in domains:
            assert name.startswith("_d")


# ── parse() bare form (no e() wrapper) ───────────────────────────────────────

class TestParseBareForms:
    def test_bare_constant(self):
        expr, domains = parse("42")
        assert expr == Const(42)
        assert domains == {}

    def test_bare_arithmetic(self):
        expr, domains = parse("d6 + d6")
        assert len(domains) == 2

    def test_bare_conditional(self):
        expr, domains = parse("d6 > 3 -> d6 | 0")
        assert isinstance(expr, IfElse)
        assert len(domains) == 2

    def test_named_form_still_works(self):
        expr, domains = parse("e(X): X; e(d6)")
        assert expr == Var("X")
        assert len(domains) == 1
