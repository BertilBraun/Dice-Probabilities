from __future__ import annotations

import argparse
import sys

from dice.engine import build_pmf
from dice.parser import parse
from dice.render import render_ascii, render_matplotlib
from dice.simplify import simplify


def _format_pmf(pmf: dict[int, float], expected_value: float) -> str:
    lines = [f'{k}: {pmf[k] * 100:.2f}%' for k in sorted(pmf)]
    lines.append(f'(E): {expected_value:.4g}')
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> None:
    argument_parser = argparse.ArgumentParser(
        prog='dice',
        description='Exact probability distributions for dice expressions.',
    )
    argument_parser.add_argument('expression', help='Dice expression to evaluate.')
    argument_parser.add_argument(
        '--render',
        choices=['ascii', 'mat', 'none'],
        default='none',
        help='Rendering mode: ascii bar chart, mat (matplotlib), or none (default: none).',
    )
    parsed = argument_parser.parse_args(argv if argv is not None else sys.argv[1:])

    try:
        expr, domains = parse(parsed.expression)
    except (SyntaxError, ValueError) as exc:
        print(f'Parse error: {exc}', file=sys.stderr)
        sys.exit(1)

    expr, domains = simplify(expr, domains)
    pmf = build_pmf(expr, domains)
    expected_value = sum(k * p for k, p in pmf.items())

    if parsed.render == 'ascii':
        print(render_ascii(pmf, expected_value))
    elif parsed.render == 'mat':
        render_matplotlib(pmf, expected_value)
    else:
        print(_format_pmf(pmf, expected_value))
