from __future__ import annotations
import sys
from dice.parser import parse
from dice.engine import build_pmf
from dice.simplify import simplify


def format_pmf(pmf: dict[int, float], ev: float) -> str:
    lines = [f"{k}: {pmf[k] * 100:.2f}%" for k in sorted(pmf)]
    lines.append(f"(E): {ev:.4g}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    args = argv if argv is not None else sys.argv[1:]
    if not args:
        print('Usage: dice "e(X): EXPR; e(dN)"', file=sys.stderr)
        sys.exit(1)
    text = args[0]
    try:
        expr, domains = parse(text)
    except (SyntaxError, ValueError) as exc:
        print(f"Parse error: {exc}", file=sys.stderr)
        sys.exit(1)
    expr, domains = simplify(expr, domains)
    pmf = build_pmf(expr, domains)
    ev = sum(k * p for k, p in pmf.items())
    print(format_pmf(pmf, ev))
