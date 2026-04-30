from __future__ import annotations
import operator as _op
from dice.ast_nodes import Expr, Const, Var, Add, Sub, Mul, Compare, IfElse, And, Or

Env = dict[str, int]

_OPERATOR_FUNCTIONS = {
    "<":  _op.lt,
    ">":  _op.gt,
    "<=": _op.le,
    ">=": _op.ge,
    "==": _op.eq,
    "!=": _op.ne,
}


def eval_expr(expr: Expr, env: Env) -> int:
    match expr:
        case Const(value=value):
            return value
        case Var(name=name):
            return env[name]
        case Add(left=left, right=right):
            return eval_expr(left, env) + eval_expr(right, env)
        case Sub(left=left, right=right):
            return eval_expr(left, env) - eval_expr(right, env)
        case Mul(left=left, right=right):
            return eval_expr(left, env) * eval_expr(right, env)
        case Compare(left=left, op=operator, right=right):
            return _OPERATOR_FUNCTIONS[operator](eval_expr(left, env), eval_expr(right, env))
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return eval_expr(then_branch, env) if eval_expr(condition, env) else eval_expr(else_branch, env)
        case And(left=left, right=right):
            return eval_expr(left, env) and eval_expr(right, env)
        case Or(left=left, right=right):
            return eval_expr(left, env) or eval_expr(right, env)
        case _:
            raise TypeError(f"Unknown expression node: {type(expr)}")
