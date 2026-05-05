import pytest
from dice.ast_nodes import Const, Var, Die, Add, Mul, Sub, Compare, IfElse, And, Or
from dice.evaluator import eval_expr


class TestConst:
    def test_returns_value(self):
        assert eval_expr(Const(7), {}) == 7

    def test_zero(self):
        assert eval_expr(Const(0), {}) == 0

    def test_negative(self):
        assert eval_expr(Const(-3), {}) == -3


class TestVar:
    def test_looks_up_env(self):
        assert eval_expr(Var('X'), {'X': 4}) == 4

    def test_multiple_vars(self):
        env = {'X': 3, 'Y': 5}
        assert eval_expr(Var('X'), env) == 3
        assert eval_expr(Var('Y'), env) == 5

    def test_missing_var_raises(self):
        with pytest.raises(KeyError):
            eval_expr(Var('Z'), {})


class TestAdd:
    def test_adds_two_consts(self):
        assert eval_expr(Add(Const(3), Const(4)), {}) == 7

    def test_adds_var_and_const(self):
        assert eval_expr(Add(Var('X'), Const(2)), {'X': 5}) == 7

    def test_left_associative_nest(self):
        expr = Add(Add(Const(1), Const(2)), Const(3))
        assert eval_expr(expr, {}) == 6


class TestMul:
    def test_multiplies_consts(self):
        assert eval_expr(Mul(Const(3), Const(4)), {}) == 12

    def test_mul_by_zero(self):
        assert eval_expr(Mul(Var('X'), Const(0)), {'X': 99}) == 0

    def test_mul_var_by_const(self):
        assert eval_expr(Mul(Const(2), Var('X')), {'X': 6}) == 12


class TestSub:
    def test_subtracts_consts(self):
        assert eval_expr(Sub(Const(10), Const(3)), {}) == 7

    def test_result_can_be_negative(self):
        assert eval_expr(Sub(Const(3), Const(10)), {}) == -7


class TestCompare:
    @pytest.mark.parametrize(
        'left,op,right,expected',
        [
            (5, '>', 3, True),
            (3, '>', 5, False),
            (3, '>', 3, False),
            (3, '<', 5, True),
            (5, '<', 3, False),
            (3, '<=', 3, True),
            (4, '<=', 3, False),
            (3, '>=', 3, True),
            (2, '>=', 3, False),
            (3, '==', 3, True),
            (3, '==', 4, False),
            (3, '!=', 4, True),
            (3, '!=', 3, False),
        ],
    )
    def test_all_operators(self, left, op, right, expected):
        expr = Compare(Const(left), Const(right), op)
        assert eval_expr(expr, {}) == expected


class TestIfElse:
    def test_takes_then_branch_when_true(self):
        condition = Compare(Var('X'), Const(3), '>')
        expr = IfElse(condition, Mul(Var('X'), Const(2)), Const(0))
        assert eval_expr(expr, {'X': 5}) == 10

    def test_takes_else_branch_when_false(self):
        condition = Compare(Var('X'), Const(3), '>')
        expr = IfElse(condition, Mul(Var('X'), Const(2)), Const(0))
        assert eval_expr(expr, {'X': 2}) == 0

    def test_boundary_at_exactly_threshold(self):
        condition = Compare(Var('X'), Const(3), '>')
        expr = IfElse(condition, Mul(Var('X'), Const(2)), Const(0))
        assert eval_expr(expr, {'X': 3}) == 0

    def test_nested_ifelse(self):
        inner = IfElse(Compare(Var('X'), Const(2), '>'), Const(1), Const(0))
        outer = IfElse(Compare(Var('X'), Const(4), '>'), Const(2), inner)
        assert eval_expr(outer, {'X': 3}) == 1
        assert eval_expr(outer, {'X': 5}) == 2
        assert eval_expr(outer, {'X': 1}) == 0


class TestDieRaisesTypeError:
    def test_die_node_is_not_evaluatable(self):
        with pytest.raises(TypeError):
            eval_expr(Die(6), {})


class TestComplexExpressions:
    def test_two_var_conditional_then_branch(self):
        # X+Y>6 -> X*2+Y | X+2*Y ; X=4, Y=5 → 13
        condition = Compare(Add(Var('X'), Var('Y')), Const(6), '>')
        then_branch = Add(Mul(Var('X'), Const(2)), Var('Y'))
        else_branch = Add(Var('X'), Mul(Const(2), Var('Y')))
        expr = IfElse(condition, then_branch, else_branch)
        assert eval_expr(expr, {'X': 4, 'Y': 5}) == 13

    def test_two_var_conditional_else_branch(self):
        # X=2, Y=3 → 5 not > 6 → else: 2 + 2*3 = 8
        condition = Compare(Add(Var('X'), Var('Y')), Const(6), '>')
        then_branch = Add(Mul(Var('X'), Const(2)), Var('Y'))
        else_branch = Add(Var('X'), Mul(Const(2), Var('Y')))
        expr = IfElse(condition, then_branch, else_branch)
        assert eval_expr(expr, {'X': 2, 'Y': 3}) == 8


class TestAnd:
    def test_true_and_true(self):
        expr = And(Compare(Var('X'), Const(1), '>'), Compare(Var('Y'), Const(1), '>'))
        assert eval_expr(expr, {'X': 2, 'Y': 2})

    def test_false_and_true(self):
        expr = And(Compare(Var('X'), Const(5), '>'), Compare(Var('Y'), Const(1), '>'))
        assert not eval_expr(expr, {'X': 2, 'Y': 2})

    def test_true_and_false(self):
        expr = And(Compare(Var('X'), Const(1), '>'), Compare(Var('Y'), Const(5), '>'))
        assert not eval_expr(expr, {'X': 2, 'Y': 2})

    def test_false_and_false(self):
        expr = And(Compare(Var('X'), Const(5), '>'), Compare(Var('Y'), Const(5), '>'))
        assert not eval_expr(expr, {'X': 2, 'Y': 2})

    def test_and_in_ifelse_condition(self):
        condition = And(Compare(Var('X'), Const(3), '>'), Compare(Var('Y'), Const(3), '>'))
        expr = IfElse(condition, Add(Var('X'), Var('Y')), Const(0))
        assert eval_expr(expr, {'X': 4, 'Y': 5}) == 9
        assert eval_expr(expr, {'X': 4, 'Y': 2}) == 0
        assert eval_expr(expr, {'X': 2, 'Y': 5}) == 0


class TestOr:
    def test_true_or_true(self):
        expr = Or(Compare(Var('X'), Const(1), '>'), Compare(Var('Y'), Const(1), '>'))
        assert eval_expr(expr, {'X': 2, 'Y': 2})

    def test_true_or_false(self):
        expr = Or(Compare(Var('X'), Const(1), '>'), Compare(Var('Y'), Const(5), '>'))
        assert eval_expr(expr, {'X': 2, 'Y': 2})

    def test_false_or_true(self):
        expr = Or(Compare(Var('X'), Const(5), '>'), Compare(Var('Y'), Const(1), '>'))
        assert eval_expr(expr, {'X': 2, 'Y': 2})

    def test_false_or_false(self):
        expr = Or(Compare(Var('X'), Const(5), '>'), Compare(Var('Y'), Const(5), '>'))
        assert not eval_expr(expr, {'X': 2, 'Y': 2})

    def test_or_in_ifelse_condition(self):
        condition = Or(Compare(Var('X'), Const(4), '>'), Compare(Var('Y'), Const(4), '>'))
        expr = IfElse(condition, Add(Var('X'), Var('Y')), Const(0))
        assert eval_expr(expr, {'X': 5, 'Y': 1}) == 6  # X passes
        assert eval_expr(expr, {'X': 1, 'Y': 5}) == 6  # Y passes
        assert eval_expr(expr, {'X': 2, 'Y': 2}) == 0  # neither passes


class TestAndOrPrecedence:
    def test_and_binds_tighter_than_or(self):
        inner_and = And(Compare(Var('X'), Const(9), '>'), Compare(Var('X'), Const(9), '>'))
        outer_or = Or(Compare(Var('X'), Const(0), '>'), inner_and)
        assert eval_expr(outer_or, {'X': 1})

    def test_and_or_chained(self):
        expr = And(
            And(Compare(Var('X'), Const(0), '>'), Compare(Var('Y'), Const(0), '>')), Compare(Var('Z'), Const(0), '>')
        )
        assert eval_expr(expr, {'X': 1, 'Y': 1, 'Z': 1})
        assert not eval_expr(expr, {'X': 1, 'Y': 1, 'Z': 0})
