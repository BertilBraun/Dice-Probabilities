from __future__ import annotations
import re
from enum import auto, Enum
from typing import NamedTuple
from dice.ast_nodes import (
    Expr, Const, Var, Die, Add, Sub, Mul, Compare, IfElse, And, Or, VALID_OPS
)
from dice.engine import die_domain, Domain

# ── Token types ───────────────────────────────────────────────────────────────


class TokenKind(Enum):
    ARROW = auto()
    OR    = auto()
    AND   = auto()
    OP    = auto()
    INT   = auto()
    NAME  = auto()
    PUNCT = auto()
    EOF   = auto()


class Token(NamedTuple):
    kind: TokenKind
    value: str


# ── Lexer ─────────────────────────────────────────────────────────────────────

_TOKEN_PATTERN = re.compile(
    r"\s*(?:"
    r"(->|\|\||&&)|"               # ARROW / OR / AND  (two-char, must precede single-char)
    r"(>=|<=|==|!=)|"              # TWO-CHAR CMP OPS
    r"([><])|"                     # SINGLE-CHAR CMP
    r"(\d+)|"                      # INT
    r"([A-Za-z][A-Za-z0-9_]*)|"   # NAME
    r"([+\-*|(),;:])"              # PUNCTUATION (bare | is here, after ||)
    r")\s*"
)

_INLINE_DIE_PATTERN = re.compile(r"^d(\d+)$")


def _tokenise(text: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    while position < len(text):
        regex_match = _TOKEN_PATTERN.match(text, position)
        if not regex_match:
            raise SyntaxError(f"Unexpected character at position {position}: {text[position]!r}")
        multi_char_op, two_char_cmp, single_char_cmp, integer_str, identifier, punctuation = regex_match.groups()
        if multi_char_op:
            if multi_char_op == "->":
                tokens.append(Token(TokenKind.ARROW, "->"))
            elif multi_char_op == "||":
                tokens.append(Token(TokenKind.OR, "||"))
            else:
                tokens.append(Token(TokenKind.AND, "&&"))
        elif two_char_cmp:
            tokens.append(Token(TokenKind.OP, two_char_cmp))
        elif single_char_cmp:
            tokens.append(Token(TokenKind.OP, single_char_cmp))
        elif integer_str:
            tokens.append(Token(TokenKind.INT, integer_str))
        elif identifier:
            tokens.append(Token(TokenKind.NAME, identifier))
        elif punctuation:
            tokens.append(Token(TokenKind.PUNCT, punctuation))
        position = regex_match.end()
    tokens.append(Token(TokenKind.EOF, ""))
    return tokens


# ── Parser ────────────────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._position = 0

    def _peek(self) -> Token:
        return self._tokens[self._position]

    def _consume(self, kind: TokenKind | None = None, value: str | None = None) -> Token:
        token = self._tokens[self._position]
        if kind is not None and token.kind != kind:
            raise SyntaxError(
                f"Expected token kind {kind.name!r} but got {token.kind.name!r} ({token.value!r})"
            )
        if value is not None and token.value != value:
            raise SyntaxError(f"Expected {value!r} but got {token.value!r}")
        self._position += 1
        return token

    def _at_end(self) -> bool:
        return self._tokens[self._position].kind == TokenKind.EOF

    # ── expression levels ────────────────────────────────────────────────────

    def parse_expr(self) -> Expr:
        return self._parse_ifelse()

    def _parse_ifelse(self) -> Expr:
        condition = self._parse_orelse()
        if self._peek().kind == TokenKind.ARROW:
            self._consume()
            then_branch = self._parse_orelse()
            self._consume(TokenKind.PUNCT, "|")
            else_branch = self._parse_orelse()
            return IfElse(condition, then_branch, else_branch)
        return condition

    def _parse_orelse(self) -> Expr:
        node = self._parse_andexpr()
        while self._peek().kind == TokenKind.OR:
            self._consume()
            node = Or(node, self._parse_andexpr())
        return node

    def _parse_andexpr(self) -> Expr:
        node = self._parse_compare()
        while self._peek().kind == TokenKind.AND:
            self._consume()
            node = And(node, self._parse_compare())
        return node

    def _parse_compare(self) -> Expr:
        left = self._parse_add()
        token = self._peek()
        if token.kind == TokenKind.OP and token.value in VALID_OPS:
            self._consume()
            right = self._parse_add()
            return Compare(left, token.value, right)
        return left

    def _parse_add(self) -> Expr:
        node = self._parse_mul()
        while True:
            token = self._peek()
            if token.kind == TokenKind.PUNCT and token.value == "+":
                self._consume()
                node = Add(node, self._parse_mul())
            elif token.kind == TokenKind.PUNCT and token.value == "-":
                self._consume()
                node = Sub(node, self._parse_mul())
            else:
                break
        return node

    def _parse_mul(self) -> Expr:
        node = self._parse_atom()
        while self._peek() == Token(TokenKind.PUNCT, "*"):
            self._consume()
            node = Mul(node, self._parse_atom())
        return node

    def _parse_atom(self) -> Expr:
        token = self._peek()
        if token.kind == TokenKind.INT:
            self._consume()
            return Const(int(token.value))
        if token.kind == TokenKind.NAME:
            self._consume()
            die_match = _INLINE_DIE_PATTERN.match(token.value)
            if die_match:
                return Die(int(die_match.group(1)))
            return Var(token.value)
        if token.kind == TokenKind.PUNCT and token.value == "(":
            self._consume()
            node = self.parse_expr()
            self._consume(TokenKind.PUNCT, ")")
            return node
        raise SyntaxError(f"Unexpected token {token.kind.name!r} ({token.value!r})")

    # ── full e(...): ...; e(...) syntax ──────────────────────────────────────

    def parse_full(self) -> tuple[Expr, dict[str, Domain]]:
        # e( var_list ):
        self._consume(TokenKind.NAME, "e")
        self._consume(TokenKind.PUNCT, "(")
        var_names: list[str] = [self._consume(TokenKind.NAME).value]
        while self._peek() == Token(TokenKind.PUNCT, ","):
            self._consume()
            var_names.append(self._consume(TokenKind.NAME).value)
        self._consume(TokenKind.PUNCT, ")")
        self._consume(TokenKind.PUNCT, ":")

        # expression (may contain inline Die nodes)
        expr = self.parse_expr()

        # ; e( die_list )
        self._consume(TokenKind.PUNCT, ";")
        self._consume(TokenKind.NAME, "e")
        self._consume(TokenKind.PUNCT, "(")
        die_sizes: list[int] = [self._parse_die_ref()]
        while self._peek() == Token(TokenKind.PUNCT, ","):
            self._consume()
            die_sizes.append(self._parse_die_ref())
        self._consume(TokenKind.PUNCT, ")")

        if len(var_names) != len(die_sizes):
            raise ValueError(
                f"Number of variables ({len(var_names)}) does not match "
                f"number of dice ({len(die_sizes)})"
            )

        named_domains = {var_name: die_domain(sides) for var_name, sides in zip(var_names, die_sizes)}

        die_counter = [0]
        inline_domains: dict[str, Domain] = {}
        expr = _extract_dice(expr, die_counter, inline_domains)

        return expr, {**named_domains, **inline_domains}

    def _parse_die_ref(self) -> int:
        token = self._consume(TokenKind.NAME)
        if token.value == "d":
            sides_token = self._consume(TokenKind.INT)
            return int(sides_token.value)
        die_match = _INLINE_DIE_PATTERN.match(token.value)
        if die_match:
            return int(die_match.group(1))
        raise SyntaxError(f"Expected die reference like 'd6', got {token.value!r}")


# ── Die extraction ────────────────────────────────────────────────────────────

def _extract_dice(
    expr: Expr,
    die_counter: list[int],
    domains: dict[str, Domain],
) -> Expr:
    match expr:
        case Die(sides=sides):
            var_name = f"_d{die_counter[0]}"
            die_counter[0] += 1
            domains[var_name] = die_domain(sides)
            return Var(var_name)
        case Add(left=left, right=right):
            return Add(_extract_dice(left, die_counter, domains), _extract_dice(right, die_counter, domains))
        case Sub(left=left, right=right):
            return Sub(_extract_dice(left, die_counter, domains), _extract_dice(right, die_counter, domains))
        case Mul(left=left, right=right):
            return Mul(_extract_dice(left, die_counter, domains), _extract_dice(right, die_counter, domains))
        case Compare(left=left, op=operator, right=right):
            return Compare(_extract_dice(left, die_counter, domains), operator, _extract_dice(right, die_counter, domains))
        case And(left=left, right=right):
            return And(_extract_dice(left, die_counter, domains), _extract_dice(right, die_counter, domains))
        case Or(left=left, right=right):
            return Or(_extract_dice(left, die_counter, domains), _extract_dice(right, die_counter, domains))
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return IfElse(
                _extract_dice(condition, die_counter, domains),
                _extract_dice(then_branch, die_counter, domains),
                _extract_dice(else_branch, die_counter, domains),
            )
        case _:  # Const, Var — leaves
            return expr


# ── Public API ────────────────────────────────────────────────────────────────

def parse(text: str) -> tuple[Expr, dict[str, Domain]]:
    tokens = _tokenise(text)
    parser = _Parser(tokens)
    if text.strip().startswith("e("):
        return parser.parse_full()
    # Bare expression form
    expr = parser.parse_expr()
    if not parser._at_end():
        trailing_token = parser._peek()
        raise SyntaxError(f"Unexpected trailing token {trailing_token.kind.name!r} ({trailing_token.value!r})")
    die_counter = [0]
    domains: dict[str, Domain] = {}
    expr = _extract_dice(expr, die_counter, domains)
    return expr, domains


def parse_expr(text: str) -> Expr:
    if not text.strip():
        raise SyntaxError("Empty expression")
    tokens = _tokenise(text)
    parser = _Parser(tokens)
    node = parser.parse_expr()
    if not parser._at_end():
        trailing_token = parser._peek()
        raise SyntaxError(f"Unexpected trailing token {trailing_token.kind.name!r} ({trailing_token.value!r})")
    return node
