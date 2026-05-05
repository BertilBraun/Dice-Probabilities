from __future__ import annotations

import operator as _op
from dataclasses import dataclass
from typing import Callable

_CMP_FNS: dict[str, Callable[[int, int], bool]] = {
    '<': _op.lt,
    '>': _op.gt,
    '<=': _op.le,
    '>=': _op.ge,
    '==': _op.eq,
    '!=': _op.ne,
}
VALID_OPS = set(_CMP_FNS.keys())


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
class BinaryNode(Expr):
    left: Expr
    right: Expr

    def apply(self, left_val: int, right_val: int) -> int:
        raise NotImplementedError

    def with_children(self, left: Expr, right: Expr) -> BinaryNode:
        return type(self)(left, right)


@dataclass(frozen=True)
class Add(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return left_val + right_val


@dataclass(frozen=True)
class Sub(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return left_val - right_val


@dataclass(frozen=True)
class Mul(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return left_val * right_val


@dataclass(frozen=True)
class Compare(BinaryNode):
    # Field order under inheritance is (left, right, op) — parent fields first.
    op: str

    def apply(self, left_val: int, right_val: int) -> int:
        return int(_CMP_FNS[self.op](left_val, right_val))

    def with_children(self, left: Expr, right: Expr) -> Compare:
        return Compare(left, right, self.op)


@dataclass(frozen=True)
class And(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return left_val and right_val


@dataclass(frozen=True)
class Or(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return left_val or right_val


@dataclass(frozen=True)
class IfElse(Expr):
    condition: Expr
    then_branch: Expr
    else_branch: Expr


@dataclass(frozen=True)
class Call(Expr):
    name: str
    args: tuple[Expr, ...]
