from __future__ import annotations

from dice.ast_nodes import BinaryNode, Call, Const, Expr, IfElse, Var

Env = dict[str, int]


def eval_expr(expr: Expr, env: Env) -> int:
    match expr:
        case Const(value=value):
            return value
        case Var(name=name):
            return env[name]
        case BinaryNode(left=left, right=right):
            return expr.apply(eval_expr(left, env), eval_expr(right, env))
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return eval_expr(then_branch, env) if eval_expr(condition, env) else eval_expr(else_branch, env)
        case Call():
            raise TypeError("Unexpanded function call reached the evaluator; use parse() to expand first")
        case _:
            raise TypeError(f"Unknown expression node: {type(expr)}")
