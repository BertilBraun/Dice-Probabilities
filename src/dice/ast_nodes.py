from __future__ import annotations
from dataclasses import dataclass


class Expr:
    pass


@dataclass(frozen=True)
class Const(Expr):
    value: int


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Die(Expr):
    sides: int


@dataclass(frozen=True)
class Add(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Sub(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Mul(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Compare(Expr):
    left: Expr
    op: str
    right: Expr


@dataclass(frozen=True)
class IfElse(Expr):
    condition: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass(frozen=True)
class And(Expr):
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Or(Expr):
    left: Expr
    right: Expr


VALID_OPS = {"<", ">", "<=", ">=", "==", "!="}
