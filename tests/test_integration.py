import subprocess
import sys
import pytest
from dice.parser import parse
from dice.engine import build_pmf


def approx(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) < tol


def pmf_sum(pmf: dict) -> float:
    return sum(pmf.values())


def _run(text: str) -> dict[int, float]:
    expr, domains = parse(text)
    return build_pmf(expr, domains)


class TestFullPipeline:
    def test_identity_d6(self):
        pmf = _run('e(X): X; e(d6)')
        assert set(pmf.keys()) == {1, 2, 3, 4, 5, 6}
        assert approx(pmf_sum(pmf), 1.0)

    def test_2d6_expected_value(self):
        pmf = _run('e(X, Y): X + Y; e(d6, d6)')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)

    def test_conditional_threshold_d6(self):
        pmf = _run('e(X): X > 3 -> X | 0; e(d6)')
        assert approx(pmf[0], 0.5)
        assert approx(pmf[4], 1 / 6)
        assert approx(pmf[5], 1 / 6)
        assert approx(pmf[6], 1 / 6)

    def test_conditional_expected_value(self):
        pmf = _run('e(X): X > 3 -> X | 0; e(d6)')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 2.5)

    def test_two_var_complex(self):
        pmf = _run('e(X, Y): X + Y > 6 -> X*2 + Y | X + 2*Y; e(d6, d6)')
        assert approx(pmf_sum(pmf), 1.0)
        assert all(p >= 0 for p in pmf.values())
        assert max(pmf.keys()) == 18
        assert min(pmf.keys()) == 3

    def test_d20_expected_value(self):
        pmf = _run('e(X): X; e(d20)')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 10.5)


EXPRESSIONS = [
    'e(X): X; e(d6)',
    'e(X): X; e(d20)',
    'e(X): X; e(d4)',
    'e(X, Y): X + Y; e(d6, d6)',
    'e(X, Y): X + Y; e(d4, d6)',
    'e(X): X > 3 -> X | 0; e(d6)',
    'e(X): 2*X; e(d6)',
    'e(X, Y): X + Y > 6 -> X*2 + Y | X + 2*Y; e(d6, d6)',
]


class TestPropertyChecks:
    @pytest.mark.parametrize('text', EXPRESSIONS)
    def test_sums_to_1(self, text):
        pmf = _run(text)
        assert approx(pmf_sum(pmf), 1.0), f"PMF for '{text}' sums to {pmf_sum(pmf)}"

    @pytest.mark.parametrize('text', EXPRESSIONS)
    def test_no_negative_probs(self, text):
        pmf = _run(text)
        for k, p in pmf.items():
            assert p >= 0, f"Negative probability {p} for outcome {k} in '{text}'"

    @pytest.mark.parametrize('text', EXPRESSIONS)
    def test_all_probs_at_most_1(self, text):
        pmf = _run(text)
        for k, p in pmf.items():
            assert p <= 1.0 + 1e-9, f"Probability {p} > 1 for outcome {k} in '{text}'"


class TestKnownExpectedValues:
    @pytest.mark.parametrize(
        'text,expected_ev',
        [
            ('e(X): X; e(d4)', 2.5),
            ('e(X): X; e(d6)', 3.5),
            ('e(X): X; e(d8)', 4.5),
            ('e(X): X; e(d10)', 5.5),
            ('e(X): X; e(d12)', 6.5),
            ('e(X): X; e(d20)', 10.5),
            ('e(X, Y): X + Y; e(d6, d6)', 7.0),
            ('e(X): X > 3 -> X | 0; e(d6)', 2.5),
        ],
    )
    def test_expected_value(self, text, expected_ev):
        pmf = _run(text)
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, expected_ev), f'E[{text!r}] = {ev}, expected {expected_ev}'


class TestCLIOutput:
    def _cli(self, text: str) -> tuple[str, int]:
        result = subprocess.run([sys.executable, '-m', 'dice', text], capture_output=True, text=True)
        return result.stdout, result.returncode

    def test_exit_code_success(self):
        _, code = self._cli('e(X): X; e(d6)')
        assert code == 0

    def test_output_contains_expected_value_label(self):
        out, _ = self._cli('e(X): X; e(d6)')
        assert '(E):' in out

    def test_output_contains_ev_3_5(self):
        out, _ = self._cli('e(X): X; e(d6)')
        assert '3.5' in out

    def test_output_probabilities_are_percentages(self):
        out, _ = self._cli('e(X): X; e(d6)')
        assert '%' in out

    def test_bad_input_exits_nonzero(self):
        _, code = self._cli('not valid syntax at all')
        assert code != 0

    def test_threshold_output_contains_zero(self):
        out, _ = self._cli('e(X): X > 3 -> X | 0; e(d6)')
        assert '0:' in out

    def test_d6_output_has_six_outcome_lines(self):
        out, _ = self._cli('e(X): X; e(d6)')
        percent_lines = [ln for ln in out.splitlines() if '%' in ln]
        assert len(percent_lines) == 6

    def test_output_sorted_ascending(self):
        out, _ = self._cli('e(X): X; e(d6)')
        outcomes = [int(ln.split(':')[0]) for ln in out.splitlines() if '%' in ln]
        assert outcomes == sorted(outcomes)


# ── Anonymous dice (bare form) ────────────────────────────────────────────────


class TestAnonymousDice:
    def test_single_d6(self):
        pmf = _run('d6')
        assert set(pmf.keys()) == {1, 2, 3, 4, 5, 6}
        assert approx(pmf_sum(pmf), 1.0)

    def test_2d6_sum(self):
        pmf = _run('d6 + d6')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)

    def test_2d6_sum_range(self):
        pmf = _run('d6 + d6')
        assert min(pmf.keys()) == 2
        assert max(pmf.keys()) == 12

    def test_d6_plus_d4_expected_value(self):
        pmf = _run('d6 + d4')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 3.5 + 2.5)

    def test_anonymous_threshold(self):
        # d6 > 3 -> d6 | 0  — TWO independent dice
        # condition die and result die are different rolls
        pmf = _run('d6 > 3 -> d6 | 0')
        # 3 outcomes from condition die ≤ 3 → 0;  3 outcomes from condition die > 3 each paired with 6 result outcomes
        # P(0) = 3/6 = 1/2  (cond die ∈ {1,2,3})
        # P(k for k in 1..6) = (3/6)*(1/6) = 1/12  (cond die ∈ {4,5,6})
        assert approx(pmf[0], 0.5)
        for k in range(1, 7):
            assert approx(pmf[k], 1 / 12)

    def test_d20_identity(self):
        pmf = _run('d20')
        assert len(pmf) == 20
        assert approx(pmf_sum(pmf), 1.0)
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 10.5)

    def test_anonymous_and_named_mixed(self):
        # e(X): X + d6; e(d6)  — X is named, d6 is inline/anonymous
        pmf = _run('e(X): X + d6; e(d6)')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 7.0)  # E[d6] + E[d6] = 3.5 + 3.5


# ── Boolean AND / OR ──────────────────────────────────────────────────────────


class TestBooleanOperators:
    def test_and_condition_both_must_pass(self):
        # X > 3 && Y > 3 -> X + Y | 0  over d6, d6
        # Both > 3: X,Y ∈ {4,5,6}, 9 pairs → then branch
        # Otherwise: 27 pairs → else branch = 0
        pmf = _run('e(X, Y): X > 3 && Y > 3 -> X + Y | 0; e(d6, d6)')
        assert approx(pmf[0], 27 / 36)

    def test_and_condition_expected_value(self):
        # E[X+Y | both >3] * P(both >3) = E[X+Y for X,Y in 4..6] * 9/36
        # E[X for X in {4,5,6}] = 5.0, so E[X+Y] = 10.0 conditioned
        # E[result] = 10.0 * (9/36) + 0 * (27/36) = 90/36 = 2.5
        pmf = _run('e(X, Y): X > 3 && Y > 3 -> X + Y | 0; e(d6, d6)')
        ev = sum(k * p for k, p in pmf.items())
        assert approx(ev, 10.0 * 9 / 36)

    def test_or_condition_either_can_pass(self):
        # X > 4 || Y > 4 -> X + Y | 0  over d6, d6
        # Neither > 4: X,Y ∈ {1..4}, 16 pairs → else = 0
        # At least one > 4: 36 - 16 = 20 pairs → then
        pmf = _run('e(X, Y): X > 4 || Y > 4 -> X + Y | 0; e(d6, d6)')
        assert approx(pmf[0], 16 / 36)

    def test_or_condition_sums_to_1(self):
        pmf = _run('e(X, Y): X > 4 || Y > 4 -> X + Y | 0; e(d6, d6)')
        assert approx(pmf_sum(pmf), 1.0)

    def test_and_in_complex_expression(self):
        # X > 2 && Y > 2 -> X*Y | X + Y  over d4, d4
        pmf = _run('e(X, Y): X > 2 && Y > 2 -> X*Y | X + Y; e(d4, d4)')
        assert approx(pmf_sum(pmf), 1.0)
        assert all(p >= 0 for p in pmf.values())
        # X=3,Y=3: 3*3=9; X=4,Y=4: 16; etc.
        assert 9 in pmf

    def test_combined_and_or(self):
        # X > 4 || (Y > 3 && Z > 3) -> X + Y + Z | 0  over d6, d6, d6
        pmf = _run('e(X, Y, Z): X > 4 || Y > 3 && Z > 3 -> X + Y + Z | 0; e(d6, d6, d6)')
        assert approx(pmf_sum(pmf), 1.0)
        assert all(p >= 0 for p in pmf.values())

    def test_anonymous_dice_with_and(self):
        # d6 > 3 && d6 > 3 -> 1 | 0 — two independent dice, P(both >3) = (3/6)^2 = 1/4
        pmf = _run('d6 > 3 && d6 > 3 -> 1 | 0')
        assert approx(pmf.get(1, 0), 9 / 36)
        assert approx(pmf.get(0, 0), 27 / 36)


# ── Property checks on new expressions ───────────────────────────────────────

NEW_EXPRESSIONS = [
    'd6',
    'd6 + d6',
    'd6 + d4',
    'd6 > 3 -> d4 | 0',
    'd6 > 3 && d6 > 3 -> 1 | 0',
    'e(X, Y): X > 3 && Y > 3 -> X + Y | 0; e(d6, d6)',
    'e(X, Y): X > 4 || Y > 4 -> X + Y | 0; e(d6, d6)',
    'e(X, Y, Z): X > 4 || Y > 3 && Z > 3 -> X + Y + Z | 0; e(d6, d6, d6)',
]


class TestNewPropertyChecks:
    @pytest.mark.parametrize('text', NEW_EXPRESSIONS)
    def test_sums_to_1(self, text):
        pmf = _run(text)
        assert approx(pmf_sum(pmf), 1.0)

    @pytest.mark.parametrize('text', NEW_EXPRESSIONS)
    def test_no_negative_probs(self, text):
        pmf = _run(text)
        assert all(p >= 0 for p in pmf.values())
