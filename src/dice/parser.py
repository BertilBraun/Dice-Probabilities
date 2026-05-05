from __future__ import annotations
import re
from enum import auto, Enum
from typing import NamedTuple
from dice.ast_nodes import (
    Add, BinaryNode, Call, Compare, Const, Die, Expr, IfElse, Mul, Or, And, Sub, Var, VALID_OPS
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
    r"[ \t]*(?:"
    r"(->|\|\||&&)|"               # ARROW / OR / AND  (two-char, must precede single-char)
    r"(>=|<=|==|!=)|"              # TWO-CHAR CMP OPS
    r"([><])|"                     # SINGLE-CHAR CMP
    r"(\d+)|"                      # INT
    r"([A-Za-z][A-Za-z0-9_]*)|"   # NAME
    r"([+\-*|(),;:])"              # PUNCTUATION (bare | is here, after ||)
    r")[ \t]*"
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
            return Compare(left, right, token.value)
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
            # Function call: NAME(arg, ...)
            if self._peek() == Token(TokenKind.PUNCT, "("):
                self._consume()
                args: list[Expr] = []
                if self._peek() != Token(TokenKind.PUNCT, ")"):
                    args.append(self.parse_expr())
                    while self._peek() == Token(TokenKind.PUNCT, ","):
                        self._consume()
                        args.append(self.parse_expr())
                self._consume(TokenKind.PUNCT, ")")
                return Call(token.value, tuple(args))
            return Var(token.value)
        if token.kind == TokenKind.PUNCT and token.value == "(":
            self._consume()
            node = self.parse_expr()
            self._consume(TokenKind.PUNCT, ")")
            return node
        raise SyntaxError(f"Unexpected token {token.kind.name!r} ({token.value!r})")


# ── Statement splitting ───────────────────────────────────────────────────────

def _split_statements(tokens: list[Token]) -> list[list[Token]]:
    """Split a token list into per-statement groups at PUNCT ';' boundaries."""
    statements: list[list[Token]] = []
    current: list[Token] = []
    for token in tokens:
        if token.kind == TokenKind.EOF:
            break
        elif token == Token(TokenKind.PUNCT, ";"):
            if current:
                statements.append(current)
                current = []
        else:
            current.append(token)
    if current:
        statements.append(current)
    return statements


def _is_function_definition(stmt_tokens: list[Token]) -> bool:
    """Return True if stmt_tokens starts with the pattern NAME(NAME,...):"""
    if len(stmt_tokens) < 3:
        return False
    if stmt_tokens[0].kind != TokenKind.NAME:
        return False
    if stmt_tokens[1] != Token(TokenKind.PUNCT, "("):
        return False
    pos = 2
    # Skip zero or more NAME tokens separated by commas
    if pos < len(stmt_tokens) and stmt_tokens[pos].kind == TokenKind.NAME:
        pos += 1
        while pos < len(stmt_tokens) and stmt_tokens[pos] == Token(TokenKind.PUNCT, ","):
            pos += 1
            if pos >= len(stmt_tokens) or stmt_tokens[pos].kind != TokenKind.NAME:
                return False
            pos += 1
    if pos >= len(stmt_tokens) or stmt_tokens[pos] != Token(TokenKind.PUNCT, ")"):
        return False
    pos += 1
    return pos < len(stmt_tokens) and stmt_tokens[pos] == Token(TokenKind.PUNCT, ":")


def _parse_function_def(stmt_tokens: list[Token]) -> tuple[str, list[str], Expr]:
    """Parse NAME(params): expr and return (name, param_names, body)."""
    parser = _Parser(stmt_tokens + [Token(TokenKind.EOF, "")])
    func_name = parser._consume(TokenKind.NAME).value
    parser._consume(TokenKind.PUNCT, "(")
    param_names: list[str] = []
    if parser._peek().kind == TokenKind.NAME:
        param_names.append(parser._consume(TokenKind.NAME).value)
        while parser._peek() == Token(TokenKind.PUNCT, ","):
            parser._consume()
            param_names.append(parser._consume(TokenKind.NAME).value)
    parser._consume(TokenKind.PUNCT, ")")
    parser._consume(TokenKind.PUNCT, ":")
    body = parser.parse_expr()
    if not parser._at_end():
        trailing = parser._peek()
        raise SyntaxError(
            f"Unexpected trailing token in function body: {trailing.kind.name!r} ({trailing.value!r})"
        )
    return func_name, param_names, body


def _parse_statement_expr(stmt_tokens: list[Token]) -> Expr:
    """Parse a statement as a bare expression."""
    parser = _Parser(stmt_tokens + [Token(TokenKind.EOF, "")])
    expr = parser.parse_expr()
    if not parser._at_end():
        trailing = parser._peek()
        raise SyntaxError(f"Unexpected trailing token: {trailing.kind.name!r} ({trailing.value!r})")
    return expr


# ── Substitution ──────────────────────────────────────────────────────────────

def _substitute(expr: Expr, substitution: dict[str, Expr]) -> Expr:
    """Replace Var nodes whose names appear in substitution."""
    match expr:
        case Var(name=name):
            return substitution.get(name, expr)
        case Call(name=name, args=args):
            return Call(name, tuple(_substitute(arg, substitution) for arg in args))
        case BinaryNode(left=left, right=right):
            return expr.with_children(_substitute(left, substitution), _substitute(right, substitution))
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return IfElse(
                _substitute(condition, substitution),
                _substitute(then_branch, substitution),
                _substitute(else_branch, substitution),
            )
        case _:  # Const, Die — no variables
            return expr


# ── Function expansion ────────────────────────────────────────────────────────

FunctionRegistry = dict[str, tuple[list[str], Expr]]


def _expand(
    expr: Expr,
    registry: FunctionRegistry,
    die_counter: list[int],
    domains: dict[str, Domain],
    call_stack: frozenset[str] = frozenset(),
) -> Expr:
    """Resolve all Call nodes via macro expansion with pre-extracted die arguments."""
    match expr:
        case Call(name=name, args=args):
            if name not in registry:
                raise NameError(f"Undefined function: {name!r}")
            if name in call_stack:
                raise RecursionError(f"Recursive function call: {name!r}")
            param_names, body = registry[name]
            if len(args) != len(param_names):
                raise ValueError(
                    f"Function {name!r} takes {len(param_names)} argument(s), got {len(args)}"
                )
            # Expand each argument, then pre-extract its dice so the parameter
            # maps to a single variable even when used multiple times in the body
            clean_args: list[Expr] = []
            for arg in args:
                expanded_arg = _expand(arg, registry, die_counter, domains, call_stack)
                clean_arg = _extract_dice(expanded_arg, die_counter, domains)
                clean_args.append(clean_arg)
            substitution = dict(zip(param_names, clean_args))
            substituted_body = _substitute(body, substitution)
            return _expand(substituted_body, registry, die_counter, domains, call_stack | {name})
        case BinaryNode(left=left, right=right):
            return expr.with_children(
                _expand(left, registry, die_counter, domains, call_stack),
                _expand(right, registry, die_counter, domains, call_stack),
            )
        case IfElse(condition=condition, then_branch=then_branch, else_branch=else_branch):
            return IfElse(
                _expand(condition, registry, die_counter, domains, call_stack),
                _expand(then_branch, registry, die_counter, domains, call_stack),
                _expand(else_branch, registry, die_counter, domains, call_stack),
            )
        case _:  # Const, Var, Die — leaves, nothing to expand
            return expr


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
        case Call():
            raise TypeError("Unexpanded Call node in _extract_dice; expand before extracting dice")
        case BinaryNode(left=left, right=right):
            return expr.with_children(
                _extract_dice(left, die_counter, domains),
                _extract_dice(right, die_counter, domains),
            )
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
    # Newlines are statement separators, equivalent to semicolons
    normalized = text.replace("\n", ";")
    tokens = _tokenise(normalized)
    statement_token_lists = _split_statements(tokens)

    if not statement_token_lists:
        raise SyntaxError("Empty input")

    registry: FunctionRegistry = {}
    for stmt_tokens in statement_token_lists[:-1]:
        if not _is_function_definition(stmt_tokens):
            raise SyntaxError(
                "Non-final statements must be function definitions of the form name(params): expr"
            )
        func_name, param_names, body = _parse_function_def(stmt_tokens)
        if func_name in registry:
            raise ValueError(f"Duplicate function definition: {func_name!r}")
        registry[func_name] = (param_names, body)

    final_expr = _parse_statement_expr(statement_token_lists[-1])

    die_counter = [0]
    domains: dict[str, Domain] = {}
    expanded = _expand(final_expr, registry, die_counter, domains)
    expanded = _extract_dice(expanded, die_counter, domains)
    return expanded, domains


def parse_expr(text: str) -> Expr:
    """Parse a single expression, returning raw AST (Die and Call nodes unexpanded)."""
    if not text.strip():
        raise SyntaxError("Empty expression")
    tokens = _tokenise(text.replace("\n", ";"))
    parser = _Parser(tokens)
    node = parser.parse_expr()
    if not parser._at_end():
        trailing_token = parser._peek()
        raise SyntaxError(f"Unexpected trailing token {trailing_token.kind.name!r} ({trailing_token.value!r})")
    return node
