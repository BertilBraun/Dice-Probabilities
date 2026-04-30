import pytest
from dice.ast_nodes import (
    Const, Var, Die, Add, Mul, Sub, Compare, IfElse, And, Or
)


class TestConst:
    def test_stores_value(self):
        assert Const(5).value == 5

    def test_equality(self):
        assert Const(3) == Const(3)
        assert Const(3) != Const(4)

    def test_repr_contains_value(self):
        assert "5" in repr(Const(5))


class TestVar:
    def test_stores_name(self):
        assert Var("X").name == "X"

    def test_equality(self):
        assert Var("X") == Var("X")
        assert Var("X") != Var("Y")


class TestDie:
    def test_stores_sides(self):
        assert Die(6).sides == 6

    def test_d20(self):
        assert Die(20).sides == 20


class TestAdd:
    def test_stores_children(self):
        node = Add(Const(1), Const(2))
        assert node.left == Const(1)
        assert node.right == Const(2)

    def test_nested(self):
        node = Add(Add(Const(1), Const(2)), Const(3))
        assert node.left == Add(Const(1), Const(2))


class TestMul:
    def test_stores_children(self):
        node = Mul(Const(2), Var("X"))
        assert node.left == Const(2)
        assert node.right == Var("X")


class TestSub:
    def test_stores_children(self):
        node = Sub(Var("X"), Const(1))
        assert node.left == Var("X")
        assert node.right == Const(1)


class TestCompare:
    @pytest.mark.parametrize("op", ["<", ">", "<=", ">=", "==", "!="])
    def test_valid_ops(self, op):
        node = Compare(Var("X"), op, Const(3))
        assert node.op == op

    def test_stores_children(self):
        node = Compare(Var("X"), ">", Const(3))
        assert node.left == Var("X")
        assert node.right == Const(3)


class TestIfElse:
    def test_stores_all_branches(self):
        cond = Compare(Var("X"), ">", Const(3))
        node = IfElse(cond, Var("X"), Const(0))
        assert node.condition == cond
        assert node.then_branch == Var("X")
        assert node.else_branch == Const(0)


class TestEquality:
    def test_nested_add_equality(self):
        a = Add(Const(1), Const(2))
        b = Add(Const(1), Const(2))
        assert a == b

    def test_nested_add_inequality(self):
        a = Add(Const(1), Const(2))
        b = Add(Const(1), Const(3))
        assert a != b

    def test_different_types_not_equal(self):
        assert Const(1) != Var("X")
        assert Add(Const(1), Const(2)) != Mul(Const(1), Const(2))


class TestAnd:
    def test_stores_children(self):
        node = And(Const(1), Const(2))
        assert node.left == Const(1)
        assert node.right == Const(2)

    def test_equality(self):
        assert And(Const(1), Const(2)) == And(Const(1), Const(2))
        assert And(Const(1), Const(2)) != And(Const(1), Const(3))

    def test_repr_contains_children(self):
        r = repr(And(Const(1), Const(2)))
        assert "1" in r and "2" in r


class TestOr:
    def test_stores_children(self):
        node = Or(Const(1), Const(2))
        assert node.left == Const(1)
        assert node.right == Const(2)

    def test_equality(self):
        assert Or(Const(1), Const(2)) == Or(Const(1), Const(2))
        assert Or(Const(1), Const(2)) != And(Const(1), Const(2))

    def test_nested(self):
        node = And(Or(Const(1), Const(2)), Const(3))
        assert node.left == Or(Const(1), Const(2))
