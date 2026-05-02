from __future__ import annotations

import sys

from dice.engine import PMF

BAR_WIDTH = 30


def _block_character() -> str:
    try:
        "█".encode(sys.stdout.encoding or "utf-8")
        return "█"
    except (UnicodeEncodeError, LookupError):
        return "#"


def render_ascii(pmf: PMF, expected_value: float) -> str:
    block = _block_character()
    max_probability = max(pmf.values())
    label_width = max(len(str(outcome)) for outcome in pmf)
    lines = []
    for outcome in sorted(pmf):
        probability = pmf[outcome]
        bar_length = round(probability / max_probability * BAR_WIDTH)
        bar = block * bar_length
        lines.append(f"{outcome:>{label_width}} |{bar:<{BAR_WIDTH}}  {probability * 100:.2f}%")
    lines.append(f"(E): {expected_value:.4g}")
    return "\n".join(lines)


def render_matplotlib(pmf: PMF, expected_value: float) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "matplotlib is not installed — run: pip install matplotlib"
            "  or  pip install -e '.[plot]'"
        )

    outcomes = sorted(pmf)
    probabilities = [pmf[outcome] * 100 for outcome in outcomes]

    figure, axes = plt.subplots(figsize=(max(8, len(outcomes) * 0.6), 5))
    axes.bar(outcomes, probabilities, color="steelblue", edgecolor="white", linewidth=0.5)
    axes.set_xlabel("Outcome")
    axes.set_ylabel("Probability (%)")
    axes.set_title(f"Distribution  —  E[X] = {expected_value:.4g}")
    axes.set_xticks(outcomes)
    figure.tight_layout()
    plt.show()
