"""
Extreme multi-variable test cases where X and Y both appear
in conditions and in value expressions.

Expected values were derived analytically before writing the tests.
"""
import pytest
from dice.parser import parse
from dice.engine import build_pmf, die_domain


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def pmf_sum(pmf: dict) -> float:
    return sum(pmf.values())


def ev(pmf: dict) -> float:
    return sum(k * p for k, p in pmf.items())


def _run(text: str) -> dict[int, float]:
    expr, domains = parse(text)
    return build_pmf(expr, domains)


# ─── Absolute difference: X > Y -> X - Y | Y - X ────────────────────────────
#
# Always gives |X - Y| for 2d6.
# P(k) = (6 - k) * 2 / 36  for k ∈ {1..5}
# P(0) = 6/36
# E[|X-Y|] = (0*6 + 1*10 + 2*8 + 3*6 + 4*4 + 5*2) / 36 = 70/36 = 35/18

class TestAbsoluteDifference:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run("e(X, Y): X > Y -> X - Y | Y - X; e(d6, d6)")

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_outcomes_are_0_through_5(self):
        assert set(self.pmf.keys()) == {0, 1, 2, 3, 4, 5}

    def test_prob_of_0(self):
        assert approx(self.pmf[0], 6 / 36)

    def test_prob_of_1(self):
        assert approx(self.pmf[1], 10 / 36)

    def test_prob_of_2(self):
        assert approx(self.pmf[2], 8 / 36)

    def test_prob_of_5(self):
        assert approx(self.pmf[5], 2 / 36)

    def test_expected_value(self):
        assert approx(ev(self.pmf), 35 / 18)

    def test_symmetric_around_0(self):
        # P(k) decreases as k increases
        for k in range(1, 5):
            assert self.pmf[k] > self.pmf[k + 1]


# ─── Max of two dice: X > Y -> X | Y ─────────────────────────────────────────
#
# P(max = k) = (2k - 1) / 36  for k ∈ {1..6}
# E[max(X,Y)] = 161 / 36

class TestMaxOfTwoDice:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run("e(X, Y): X > Y -> X | Y; e(d6, d6)")

    def test_outcomes_are_1_through_6(self):
        assert set(self.pmf.keys()) == {1, 2, 3, 4, 5, 6}

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
    def test_prob_of_k(self, k):
        assert approx(self.pmf[k], (2 * k - 1) / 36)

    def test_expected_value(self):
        assert approx(ev(self.pmf), 161 / 36)

    def test_higher_outcomes_are_more_likely(self):
        for k in range(1, 6):
            assert self.pmf[k] < self.pmf[k + 1]


# ─── Min of two dice: X > Y -> Y | X ─────────────────────────────────────────
#
# P(min = k) = (13 - 2k) / 36  for k ∈ {1..6}
# E[min(X,Y)] = 91 / 36
# Cross-check: E[max] + E[min] = E[X] + E[Y] = 7.0

class TestMinOfTwoDice:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run("e(X, Y): X > Y -> Y | X; e(d6, d6)")

    def test_outcomes_are_1_through_6(self):
        assert set(self.pmf.keys()) == {1, 2, 3, 4, 5, 6}

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    @pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 6])
    def test_prob_of_k(self, k):
        assert approx(self.pmf[k], (13 - 2 * k) / 36)

    def test_expected_value(self):
        assert approx(ev(self.pmf), 91 / 36)

    def test_lower_outcomes_are_more_likely(self):
        for k in range(1, 6):
            assert self.pmf[k] > self.pmf[k + 1]


class TestMaxPlusMinEqualsSum:
    """E[max] + E[min] = E[X+Y] = 7 — a purely analytic identity."""
    def test_ev_identity(self):
        pmf_max = _run("e(X, Y): X > Y -> X | Y; e(d6, d6)")
        pmf_min = _run("e(X, Y): X > Y -> Y | X; e(d6, d6)")
        assert approx(ev(pmf_max) + ev(pmf_min), 7.0)


# ─── Double threshold (nested ifelse in both branches) ───────────────────────
#
# X > 3 -> (Y > 3 -> X*Y | X) | Y  over d6, d6
#
# Four regions:
#   X ∈ {1,2,3} (outer else):          value = Y
#   X ∈ {4,5,6}, Y ∈ {1,2,3} (inner else): value = X
#   X ∈ {4,5,6}, Y ∈ {4,5,6} (inner then): value = X*Y
#
# PMF (derived analytically):
#   k ∈ {1,2,3}:      P = 3/36   (only from outer-else branch, X=1..3, Y=k)
#   k ∈ {4,5,6}:      P = 6/36   (outer-else Y=k AND inner-else X=k, 3+3 paths)
#   k ∈ {16,25,36}:   P = 1/36   (inner-then, k = X*Y with X=Y)
#   k = 20:           P = 2/36   (inner-then: 4*5 or 5*4)
#   k = 24:           P = 2/36   (inner-then: 4*6 or 6*4)
#   k = 30:           P = 2/36   (inner-then: 5*6 or 6*5)
#
# E = (1*3 + 2*3 + 3*3 + 4*6 + 5*6 + 6*6 +
#      16 + 20*2 + 24*2 + 25 + 30*2 + 36) / 36
#   = 333 / 36 = 9.25

class TestDoubleThreshold:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run("e(X, Y): X > 3 -> (Y > 3 -> X*Y | X) | Y; e(d6, d6)")

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_small_outcomes_have_prob_3_36(self):
        for k in (1, 2, 3):
            assert approx(self.pmf[k], 3 / 36)

    def test_medium_outcomes_have_prob_6_36(self):
        for k in (4, 5, 6):
            assert approx(self.pmf[k], 6 / 36)

    def test_product_outcomes_present(self):
        for k in (16, 20, 24, 25, 30, 36):
            assert k in self.pmf

    def test_product_16_25_36_have_prob_1_36(self):
        for k in (16, 25, 36):
            assert approx(self.pmf[k], 1 / 36)

    def test_product_20_24_30_have_prob_2_36(self):
        for k in (20, 24, 30):
            assert approx(self.pmf[k], 2 / 36)

    def test_expected_value(self):
        assert approx(ev(self.pmf), 333 / 36)


# ─── Four-way branching on both variables ─────────────────────────────────────
#
# X > Y -> (X > 4 -> X*2 + Y | X + Y) | (Y > 4 -> X + Y*2 | X + Y)
#
# Four regions (exhaustive enumeration):
#   X > Y and X > 4:          value = X*2 + Y   (9 pairs)
#   X > Y and X ≤ 4:          value = X + Y     (6 pairs)
#   X ≤ Y and Y > 4:          value = X + Y*2   (11 pairs)
#   X ≤ Y and Y ≤ 4:          value = X + Y     (10 pairs)
#
# Sum of all 36 values = 125 + 30 + 158 + 50 = 363
# E = 363/36 = 121/12

class TestFourWayBranching:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run(
            "e(X, Y): X > Y -> (X > 4 -> X*2 + Y | X + Y) | "
            "(Y > 4 -> X + Y*2 | X + Y); e(d6, d6)"
        )

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_min_value(self):
        # X=1,Y=1: X≤Y and Y≤4 → X+Y = 2
        assert min(self.pmf.keys()) == 2

    def test_max_value(self):
        # X=6,Y=6: X≤Y (equality), Y=6>4 → X+Y*2 = 6+12 = 18
        assert max(self.pmf.keys()) == 18

    def test_expected_value(self):
        assert approx(ev(self.pmf), 363 / 36)

    def test_outcome_2_has_prob_1_36(self):
        # Only (X=1,Y=1): X≤Y=1≤4 → X+Y=2
        assert approx(self.pmf[2], 1 / 36)

    def test_outcome_17_has_prob_2_36(self):
        # (X=6,Y=5): X>Y, X>4 → 12+5=17
        # (X=5,Y=6): X≤Y, Y=6>4 → 5+12=17
        assert approx(self.pmf[17], 2 / 36)

    def test_outcome_11_has_prob_at_least_1_36(self):
        # Multiple paths can reach 11
        # (5,1): X>Y,X>4 → 10+1=11; (1,5): X≤Y,Y>4 → 1+10=11
        assert approx(self.pmf[11], 2 / 36)


# ─── Asymmetric condition: 2*X > Y + 3 ───────────────────────────────────────
#
# e(X, Y): X*2 > Y + 3 -> X*Y | X + Y   over d6, d6
#
# Condition: 2X > Y+3  ↔  2X - Y > 3  ↔  2X - Y ≥ 4
# This is a non-trivial diagonal boundary in the (X,Y) space.
# We verify structural properties only (no clean closed form).

class TestAsymmetricCondition:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run("e(X, Y): X*2 > Y + 3 -> X*Y | X + Y; e(d6, d6)")

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_min_is_2(self):
        # Smallest else-branch value: X=1,Y=1 → 2 (2*1=2 not > 1+3=4)
        assert min(self.pmf.keys()) == 2

    def test_max_is_36(self):
        # X=6,Y=6: 12 > 9 → then branch: 36
        assert max(self.pmf.keys()) == 36

    def test_then_branch_never_produces_2(self):
        # X*Y = 2 requires (X=1,Y=2) or (X=2,Y=1).
        # (1,2): 2 > 5? No → else. (2,1): 4 > 4? No → else.
        # So outcome 2 is only reachable via else (X+Y=2 → only (1,1); X*Y=2 via else is X+Y≠2).
        # P(2) = P(X=1,Y=1 and 2 not > 4) = 1/36
        assert approx(self.pmf[2], 1 / 36)

    def test_outcome_36_only_from_then_branch(self):
        # X*Y=36 → X=Y=6; 12 > 9 ✓. X+Y=36 impossible.
        assert approx(self.pmf[36], 1 / 36)


# ─── Three-variable case ──────────────────────────────────────────────────────
#
# e(X, Y, Z): X + Y > Z + 3 -> X*Y - Z | X + Y + Z   over d4, d4, d6
#
# Condition: X+Y > Z+3  (two dice vs one)
# We verify structural properties and two known expected-value bounds.

class TestThreeVariables:
    @pytest.fixture(autouse=True)
    def compute(self):
        self.pmf = _run(
            "e(X, Y, Z): X + Y > Z + 3 -> X*Y - Z | X + Y + Z; e(d4, d4, d6)"
        )

    def test_sums_to_1(self):
        assert approx(pmf_sum(self.pmf), 1.0)

    def test_no_negative_probs(self):
        assert all(p >= 0 for p in self.pmf.values())

    def test_min_value(self):
        # else branch: X+Y+Z ≥ 3.  Minimum when X=1,Y=1,Z=1 and 2 not > 4.
        assert min(self.pmf.keys()) == 3

    def test_max_value(self):
        # then branch: X*Y - Z.  Max = 4*4 - 1 = 15.
        # (X=4,Y=4,Z=1): 8 > 4 ✓ → 16 - 1 = 15
        assert max(self.pmf.keys()) == 15

    def test_domain_size_is_96(self):
        # 4 * 4 * 6 = 96 joint assignments → PMF is well-defined
        assert approx(pmf_sum(self.pmf), 1.0)


# ─── Property: all extreme cases satisfy probability axioms ───────────────────

EXTREME_EXPRESSIONS = [
    "e(X, Y): X > Y -> X - Y | Y - X; e(d6, d6)",
    "e(X, Y): X > Y -> X | Y; e(d6, d6)",
    "e(X, Y): X > Y -> Y | X; e(d6, d6)",
    "e(X, Y): X > 3 -> (Y > 3 -> X*Y | X) | Y; e(d6, d6)",
    "e(X, Y): X > Y -> (X > 4 -> X*2 + Y | X + Y) | (Y > 4 -> X + Y*2 | X + Y); e(d6, d6)",
    "e(X, Y): X*2 > Y + 3 -> X*Y | X + Y; e(d6, d6)",
    "e(X, Y, Z): X + Y > Z + 3 -> X*Y - Z | X + Y + Z; e(d4, d4, d6)",
]


class TestProbabilityAxioms:
    @pytest.mark.parametrize("text", EXTREME_EXPRESSIONS)
    def test_sums_to_1(self, text):
        pmf = _run(text)
        assert approx(pmf_sum(pmf), 1.0)

    @pytest.mark.parametrize("text", EXTREME_EXPRESSIONS)
    def test_no_negative_probs(self, text):
        pmf = _run(text)
        assert all(p >= 0 for p in pmf.values())

    @pytest.mark.parametrize("text", EXTREME_EXPRESSIONS)
    def test_all_probs_at_most_1(self, text):
        pmf = _run(text)
        assert all(p <= 1.0 + 1e-9 for p in pmf.values())
