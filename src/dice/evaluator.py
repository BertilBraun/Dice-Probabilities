from __future__ import annotations
import operator as _op
from dice.ast_nodes import Expr, Const, Var, Add, Sub, Mul, Compare, IfElse, And, Or

_OPS = {
    "<":  _op.lt,
    ">":  _op.gt,
    "<=": _op.le,
    ">=": _op.ge,
    "==": _op.eq,
    "!=": _op.ne,
}


def eval_expr(expr: Expr, env: dict[str, int]) -> int:
    match expr:
        case Const(value=v):
            return v
        case Var(name=n):
            return env[n]
        case Add(left=a, right=b):
            return eval_expr(a, env) + eval_expr(b, env)
        case Sub(left=a, right=b):
            return eval_expr(a, env) - eval_expr(b, env)
        case Mul(left=a, right=b):
            return eval_expr(a, env) * eval_expr(b, env)
        case Compare(left=a, op=op, right=b):
            return _OPS[op](eval_expr(a, env), eval_expr(b, env))
        case IfElse(condition=c, then_branch=t, else_branch=f):
            return eval_expr(t, env) if eval_expr(c, env) else eval_expr(f, env)
        case And(left=a, right=b):
            return eval_expr(a, env) and eval_expr(b, env)
        case Or(left=a, right=b):
            return eval_expr(a, env) or eval_expr(b, env)
        case _:
            raise TypeError(f"Unknown expression node: {type(expr)}")
