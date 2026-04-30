from __future__ import annotations
import re
from dice.ast_nodes import (
    Expr, Const, Var, Die, Add, Sub, Mul, Compare, IfElse, And, Or, VALID_OPS
)
from dice.engine import die_domain

# ── Lexer ────────────────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(
    r"\s*(?:"
    r"(->|\|\||&&)|"               # ARROW / OR / AND  (two-char, must precede single-char)
    r"(>=|<=|==|!=)|"              # TWO-CHAR CMP OPS
    r"([><])|"                     # SINGLE-CHAR CMP
    r"(\d+)|"                      # INT
    r"([A-Za-z][A-Za-z0-9_]*)|"   # NAME
    r"([+\-*|(),;:])"              # PUNCTUATION (bare | is here, after ||)
    r")\s*"
)

_INLINE_DIE_RE = re.compile(r"^d(\d+)$")


def _tokenise(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    while pos < len(text):
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise SyntaxError(f"Unexpected character at position {pos}: {text[pos]!r}")
        multi, two_char, cmp1, integer, name, punct = m.groups()
        if multi:
            if multi == "->":
                tokens.append(("ARROW", "->"))
            elif multi == "||":
                tokens.append(("OR", "||"))
            else:
                tokens.append(("AND", "&&"))
        elif two_char:
            tokens.append(("OP", two_char))
        elif cmp1:
            tokens.append(("OP", cmp1))
        elif integer:
            tokens.append(("INT", integer))
        elif name:
            tokens.append(("NAME", name))
        elif punct:
            tokens.append(("PUNCT", punct))
        pos = m.end()
    tokens.append(("EOF", ""))
    return tokens


# ── Parser ───────────────────────────────────────────────────────────────────

class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> tuple[str, str]:
        return self._tokens[self._pos]

    def _consume(self, kind: str | None = None, value: str | None = None) -> tuple[str, str]:
        tok = self._tokens[self._pos]
        if kind is not None and tok[0] != kind:
            raise SyntaxError(
                f"Expected token type {kind!r} but got {tok[0]!r} ({tok[1]!r})"
            )
        if value is not None and tok[1] != value:
            raise SyntaxError(f"Expected {value!r} but got {tok[1]!r}")
        self._pos += 1
        return tok

    def _at_end(self) -> bool:
        return self._tokens[self._pos][0] == "EOF"

    # ── expression levels ────────────────────────────────────────────────────

    def parse_expr(self) -> Expr:
        return self._parse_ifelse()

    def _parse_ifelse(self) -> Expr:
        cond = self._parse_orelse()
        if self._peek() == ("ARROW", "->"):
            self._consume()
            then_ = self._parse_orelse()
            self._consume("PUNCT", "|")
            else_ = self._parse_orelse()
            return IfElse(cond, then_, else_)
        return cond

    def _parse_orelse(self) -> Expr:
        node = self._parse_andexpr()
        while self._peek()[0] == "OR":
            self._consume()
            node = Or(node, self._parse_andexpr())
        return node

    def _parse_andexpr(self) -> Expr:
        node = self._parse_compare()
        while self._peek()[0] == "AND":
            self._consume()
            node = And(node, self._parse_compare())
        return node

    def _parse_compare(self) -> Expr:
        left = self._parse_add()
        kind, val = self._peek()
        if kind == "OP" and val in VALID_OPS:
            self._consume()
            right = self._parse_add()
            return Compare(left, val, right)
        return left

    def _parse_add(self) -> Expr:
        node = self._parse_mul()
        while True:
            kind, val = self._peek()
            if kind == "PUNCT" and val == "+":
                self._consume()
                node = Add(node, self._parse_mul())
            elif kind == "PUNCT" and val == "-":
                self._consume()
                node = Sub(node, self._parse_mul())
            else:
                break
        return node

    def _parse_mul(self) -> Expr:
        node = self._parse_atom()
        while self._peek() == ("PUNCT", "*"):
            self._consume()
            node = Mul(node, self._parse_atom())
        return node

    def _parse_atom(self) -> Expr:
        kind, val = self._peek()
        if kind == "INT":
            self._consume()
            return Const(int(val))
        if kind == "NAME":
            self._consume()
            m = _INLINE_DIE_RE.match(val)
            if m:
                return Die(int(m.group(1)))
            return Var(val)
        if kind == "PUNCT" and val == "(":
            self._consume()
            node = self.parse_expr()
            self._consume("PUNCT", ")")
            return node
        raise SyntaxError(f"Unexpected token {kind!r} ({val!r})")

    # ── full e(...): ...; e(...) syntax ──────────────────────────────────────

    def parse_full(self) -> tuple[Expr, dict[str, list[tuple[int, float]]]]:
        # e( var_list ):
        self._consume("NAME", "e")
        self._consume("PUNCT", "(")
        var_names: list[str] = [self._consume("NAME")[1]]
        while self._peek() == ("PUNCT", ","):
            self._consume()
            var_names.append(self._consume("NAME")[1])
        self._consume("PUNCT", ")")
        self._consume("PUNCT", ":")

        # expression (may contain inline Die nodes)
        expr = self.parse_expr()

        # ; e( die_list )
        self._consume("PUNCT", ";")
        self._consume("NAME", "e")
        self._consume("PUNCT", "(")
        die_sizes: list[int] = [self._parse_die_ref()]
        while self._peek() == ("PUNCT", ","):
            self._consume()
            die_sizes.append(self._parse_die_ref())
        self._consume("PUNCT", ")")

        if len(var_names) != len(die_sizes):
            raise ValueError(
                f"Number of variables ({len(var_names)}) does not match "
                f"number of dice ({len(die_sizes)})"
            )

        named_domains = {name: die_domain(sides) for name, sides in zip(var_names, die_sizes)}

        # extract any inline dice from the expression
        counter = [0]
        inline_domains: dict[str, list[tuple[int, float]]] = {}
        expr = _extract_dice(expr, counter, inline_domains)

        return expr, {**named_domains, **inline_domains}

    def _parse_die_ref(self) -> int:
        tok_kind, tok_val = self._consume("NAME")
        if tok_val == "d":
            sides_tok = self._consume("INT")
            return int(sides_tok[1])
        m = _INLINE_DIE_RE.match(tok_val)
        if m:
            return int(m.group(1))
        raise SyntaxError(f"Expected die reference like 'd6', got {tok_val!r}")


# ── Die extraction ────────────────────────────────────────────────────────────

def _extract_dice(
    expr: Expr,
    counter: list[int],
    domains: dict[str, list[tuple[int, float]]],
) -> Expr:
    match expr:
        case Die(sides=n):
            name = f"_d{counter[0]}"
            counter[0] += 1
            domains[name] = die_domain(n)
            return Var(name)
        case Add(left=a, right=b):
            return Add(_extract_dice(a, counter, domains), _extract_dice(b, counter, domains))
        case Sub(left=a, right=b):
            return Sub(_extract_dice(a, counter, domains), _extract_dice(b, counter, domains))
        case Mul(left=a, right=b):
            return Mul(_extract_dice(a, counter, domains), _extract_dice(b, counter, domains))
        case Compare(left=a, op=op, right=b):
            return Compare(_extract_dice(a, counter, domains), op, _extract_dice(b, counter, domains))
        case And(left=a, right=b):
            return And(_extract_dice(a, counter, domains), _extract_dice(b, counter, domains))
        case Or(left=a, right=b):
            return Or(_extract_dice(a, counter, domains), _extract_dice(b, counter, domains))
        case IfElse(condition=c, then_branch=t, else_branch=f):
            return IfElse(
                _extract_dice(c, counter, domains),
                _extract_dice(t, counter, domains),
                _extract_dice(f, counter, domains),
            )
        case _:  # Const, Var — leaves
            return expr


# ── Public API ────────────────────────────────────────────────────────────────

def parse(text: str) -> tuple[Expr, dict[str, list[tuple[int, float]]]]:
    tokens = _tokenise(text)
    p = _Parser(tokens)
    if text.strip().startswith("e("):
        return p.parse_full()
    # Bare expression form
    expr = p.parse_expr()
    if not p._at_end():
        kind, val = p._peek()
        raise SyntaxError(f"Unexpected trailing token {kind!r} ({val!r})")
    counter = [0]
    domains: dict[str, list[tuple[int, float]]] = {}
    expr = _extract_dice(expr, counter, domains)
    return expr, domains


def parse_expr(text: str) -> Expr:
    if not text.strip():
        raise SyntaxError("Empty expression")
    tokens = _tokenise(text)
    p = _Parser(tokens)
    node = p.parse_expr()
    if not p._at_end():
        kind, val = p._peek()
        raise SyntaxError(f"Unexpected trailing token {kind!r} ({val!r})")
    return node
