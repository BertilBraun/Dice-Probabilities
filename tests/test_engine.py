import pytest
from dice.ast_nodes import Const, Var, Add, Mul, Compare, IfElse
from dice.engine import build_pmf, die_domain


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def pmf_sum(pmf: dict) -> float:
    return sum(pmf.values())


D6 = die_domain(6)
D4 = die_domain(4)
D20 = die_domain(20)


class TestDieDomain:
    def test_d6_has_six_faces(self):
        assert len(die_domain(6)) == 6

    def test_d6_faces_are_1_through_6(self):
        assert [v for v, _ in die_domain(6)] == [1, 2, 3, 4, 5, 6]

    def test_d6_uniform_probability(self):
        for _, p in die_domain(6):
            assert approx(p, 1 / 6)

    def test_d1_is_certain(self):
        assert die_domain(1) == [(1, 1.0)]

    def test_d20_has_twenty_faces(self):
        assert len(die_domain(20)) == 20

    def test_d20_faces_start_at_1(self):
        assert die_domain(20)[0] == (1, pytest.approx(1 / 20))


class TestSingleDie:
    def test_d6_identity_pmf(self):
        pmf = build_pmf(Var("X"), {"X": D6})
        assert set(pmf.keys()) == {1, 2, 3, 4, 5, 6}
        for v in range(1, 7):
            assert approx(pmf[v], 1 / 6)

    def test_d6_sums_to_1(self):
        pmf = build_pmf(Var("X"), {"X": D6})
        assert approx(pmf_sum(pmf), 1.0)

    def test_d6_expected_value_3_5(self):
        pmf = build_pmf(Var("X"), {"X": D6})
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 3.5)

    def test_d6_no_negative_probs(self):
        pmf = build_pmf(Var("X"), {"X": D6})
        assert all(p >= 0 for p in pmf.values())

    def test_const_expr_is_certain(self):
        pmf = build_pmf(Const(42), {})
        assert pmf == {42: 1.0}

    def test_pmf_keys_are_ints_not_bools(self):
        pmf = build_pmf(Const(1), {})
        assert type(list(pmf.keys())[0]) is int


class TestTwoDice:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = build_pmf(Add(Var("X"), Var("Y")), {"X": D6, "Y": D6})

    def test_2d6_min_is_2(self):
        assert min(self.pmf.keys()) == 2

    def test_2d6_max_is_12(self):
        assert max(self.pmf.keys()) == 12

    def test_2d6_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_2d6_expected_value_7(self):
        ev = sum(k * p for k, p in self.pmf.items())
        assert approx(ev, 7.0)

    def test_2d6_mode_is_7(self):
        assert approx(self.pmf[7], 6 / 36)

    def test_2d6_prob_of_2_is_1_36(self):
        assert approx(self.pmf[2], 1 / 36)

    def test_2d6_prob_of_12_is_1_36(self):
        assert approx(self.pmf[12], 1 / 36)

    def test_2d6_symmetric(self):
        for k in range(11):
            assert approx(self.pmf[2 + k], self.pmf[12 - k])

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())


class TestScalarTransform:
    def test_2x_d6_expected_value_7(self):
        pmf = build_pmf(Mul(Const(2), Var("X")), {"X": D6})
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)

    def test_2x_d6_range_is_even_numbers(self):
        pmf = build_pmf(Mul(Const(2), Var("X")), {"X": D6})
        assert set(pmf.keys()) == {2, 4, 6, 8, 10, 12}


class TestConditionalThreshold:
    @pytest.fixture(autouse=True)
    def compute(self):
        cond = Compare(Var("X"), ">", Const(3))
        self.pmf = build_pmf(IfElse(cond, Var("X"), Const(0)), {"X": D6})

    def test_keys_are_correct(self):
        assert set(self.pmf.keys()) == {0, 4, 5, 6}

    def test_prob_of_zero_is_half(self):
        assert approx(self.pmf[0], 0.5)

    def test_prob_of_4_is_1_6(self):
        assert approx(self.pmf[4], 1 / 6)

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_expected_value_2_5(self):
        ev = sum(k * p for k, p in self.pmf.items())
        assert approx(ev, 2.5)


class TestTwoVarConditional:
    @pytest.fixture(autouse=True)
    def compute(self):
        cond  = Compare(Add(Var("X"), Var("Y")), ">", Const(6))
        then_ = Add(Mul(Var("X"), Const(2)), Var("Y"))
        else_ = Add(Var("X"), Mul(Const(2), Var("Y")))
        self.pmf = build_pmf(IfElse(cond, then_, else_), {"X": D6, "Y": D6})

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_min_value(self):
        assert min(self.pmf.keys()) == 3

    def test_max_value(self):
        assert max(self.pmf.keys()) == 18

    def test_known_probability_of_3(self):
        # X=1,Y=1: else branch 1+2*1=3, only assignment → P=1/36
        assert approx(self.pmf.get(3, 0), 1 / 36)


class TestMixedDice:
    def test_d4_plus_d20_expected_value(self):
        pmf = build_pmf(Add(Var("A"), Var("B")), {"A": D4, "B": D20})
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 13.0)

    def test_d4_plus_d20_sums_to_1(self):
        pmf = build_pmf(Add(Var("A"), Var("B")), {"A": D4, "B": D20})
        assert approx(pmf_sum(pmf), 1.0)
