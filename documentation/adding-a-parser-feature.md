# Adding a Parser-Only Feature: the `match` Expression

This document walks through designing a `match` expression for the dice language. It is an example of a **parser-only extension**: the new syntax desugars completely to existing AST nodes inside `_extract_dice`, so the evaluator, engine, and simplify pass require no changes.

---

## Motivation

The conditional `cond -> then | else` handles binary branches well. But expressing a multi-case dispatch — a damage table, a score bracket, a crit tier — requires deeply nested conditionals that are hard to read:

```text
d6 == 1 -> 0 | (d6 == 2 -> 1 | (d6 == 3 -> 2 | 3))
```

A `match` expression encodes the same logic linearly:

```text
match d6: 1 -> 0 | 2 -> 1 | 3 -> 2 | _ -> 3
```

Other natural uses:

```text
match d20: 20 -> 3 | 15..19 -> 2 | _ -> 1         # crit table
match d6: 1..2 -> 0 | 3..4 -> 1 | _ -> 2          # tiered damage
```

---

## Syntax

```text
match SCRUTINEE: ARM | ARM | ... | WILDCARD_ARM

ARM          := PATTERN -> RESULT
WILDCARD_ARM := _ -> RESULT

PATTERN      := INT              # exact match
             |  INT .. INT       # inclusive range
```

Rules:

- The wildcard arm `_ -> result` is **mandatory** and must be **last**.
- At least one non-wildcard arm is required.
- The scrutinee is any arithmetic expression (addition, multiplication, variable, die).
- Each result is any full expression (can itself contain dice, conditionals, etc.).

---

## Pipeline position

`match` is a **parse-time construct**. It never appears in the AST that reaches `build_pmf`:

```text
parse()
  _expand()           ← recurses into Match.scrutinee and each MatchCase.result
  _extract_dice()     ← extracts dice from sub-expressions, then desugars Match → IfElse
build_pmf()           ← never sees Match
```

---

## New token

The `..` range separator needs its own token type. Add to `TokenKind`:

```python
DOTDOT = auto()
```

Insert a new capture group in `_TOKEN_PATTERN` **before** the `INT` group (so `..` is never split into two `.` tokens):

```python
r'(\.\.)| '   # DOTDOT — must precede INT group
```

Update `_tokenise` to emit `Token(TokenKind.DOTDOT, '..')` for this group.

---

## New AST nodes

All five nodes are **transient** — they exist only between `_parse_atom` and the end of `_extract_dice`. They are not `Expr` subclasses (except `Match`) and never reach the evaluator, engine, or simplify pass.

```python
@dataclass(frozen=True)
class ExactPattern:
    value: int

@dataclass(frozen=True)
class RangePattern:
    low: int
    high: int

@dataclass(frozen=True)
class WildcardPattern:
    pass

MatchPattern = ExactPattern | RangePattern | WildcardPattern

@dataclass(frozen=True)
class MatchCase:
    pattern: MatchPattern
    result: Expr

@dataclass(frozen=True)
class Match(Expr):
    scrutinee: Expr
    cases: tuple[MatchCase, ...]
```

`Match` inherits from `Expr` so it can appear anywhere an expression is expected. The `case _` fallthrough in `_substitute`, `_expand`, and `_extract_dice` handles it until each walker adds an explicit arm.

---

## Parser additions

### `_parse_atom` — detect the `match` keyword

When `_parse_atom` sees a `NAME` token with value `match`, it calls `_parse_match()` instead of treating it as a variable or generic call:

```python
if token.kind == TokenKind.NAME:
    self._consume()
    if token.value == 'match':
        return self._parse_match()
    ...  # existing Die / Call / Var logic
```

### `_parse_match`

```python
def _parse_match(self) -> Match:
    # 'match' already consumed
    scrutinee = self._parse_add()          # arithmetic only — stops before ':'
    self._consume(TokenKind.PUNCT, ':')
    cases = [self._parse_match_arm()]
    while self._peek() == Token(TokenKind.PUNCT, '|'):
        self._consume()
        cases.append(self._parse_match_arm())
    for case in cases[:-1]:
        if isinstance(case.pattern, WildcardPattern):
            raise SyntaxError('Wildcard pattern _ must be the last match arm')
    if not isinstance(cases[-1].pattern, WildcardPattern):
        raise SyntaxError('Last match arm must be a wildcard _')
    return Match(scrutinee, tuple(cases))
```

### `_parse_match_arm`

```python
def _parse_match_arm(self) -> MatchCase:
    pattern = self._parse_pattern()
    self._consume(TokenKind.ARROW)
    result = self._parse_orelse()          # full expression; stops at '|' arm separator
    return MatchCase(pattern, result)
```

### `_parse_pattern`

```python
def _parse_pattern(self) -> MatchPattern:
    token = self._peek()
    if token == Token(TokenKind.NAME, '_'):
        self._consume()
        return WildcardPattern()
    if token.kind == TokenKind.INT:
        self._consume()
        low = int(token.value)
        if self._peek().kind == TokenKind.DOTDOT:
            self._consume()
            high = int(self._consume(TokenKind.INT).value)
            return RangePattern(low, high)
        return ExactPattern(low)
    raise SyntaxError(f'Expected pattern (integer, range, or _), got {token!r}')
```

---

## Tree-walker updates

`_substitute` and `_expand` each need one new case to recurse into a `Match` node. **Patterns are not expressions** — they hold only integer constants and are never substituted or expanded.

```python
# Same shape in both _substitute and _expand
case Match(scrutinee=scrutinee, cases=cases):
    return Match(
        <recurse>(scrutinee, ...),
        tuple(MatchCase(c.pattern, <recurse>(c.result, ...)) for c in cases),
    )
```

---

## Desugaring in `_extract_dice`

`_extract_dice` is where `Match` is converted to a nested `IfElse`. The scrutinee is extracted first, then each arm's result is extracted. Finally the arms are folded right-to-left into an `IfElse` chain — the wildcard arm becomes the innermost `else` branch.

```python
case Match(scrutinee=scrutinee, cases=cases):
    extracted_scrutinee = _extract_dice(scrutinee, die_counter, domains)
    extracted_cases = [
        MatchCase(c.pattern, _extract_dice(c.result, die_counter, domains))
        for c in cases
    ]
    result: Expr = extracted_cases[-1].result        # wildcard default
    for match_case in reversed(extracted_cases[:-1]):
        condition = _pattern_to_condition(match_case.pattern, extracted_scrutinee)
        result = IfElse(condition, match_case.result, result)
    return result
```

The scrutinee variable is **shared** across all pattern conditions — it represents a single roll, which is correct.

### `_pattern_to_condition`

```python
def _pattern_to_condition(pattern: MatchPattern, scrutinee: Expr) -> Expr:
    match pattern:
        case ExactPattern(value=value):
            return Compare(scrutinee, Const(value), '==')
        case RangePattern(low=low, high=high):
            return And(
                Compare(scrutinee, Const(low), '>='),
                Compare(scrutinee, Const(high), '<='),
            )
        case WildcardPattern():
            assert False, 'wildcard handled as default branch, not as condition'
```

---

## Files changed

| File | Change |
| --- | --- |
| `src/dice/ast_nodes.py` | Add `ExactPattern`, `RangePattern`, `WildcardPattern`, `MatchCase`, `Match` |
| `src/dice/parser.py` | Add `DOTDOT` token; add `_parse_match`, `_parse_match_arm`, `_parse_pattern`, `_pattern_to_condition`; update `_parse_atom`, `_substitute`, `_expand`, `_extract_dice` |
| `src/dice/evaluator.py` | No changes |
| `src/dice/engine.py` | No changes |
| `src/dice/simplify.py` | No changes |

---

## Test outline (`tests/test_match.py`)

**Parser / structural:**

- `test_exact_pattern_parses` — `parse_expr('match X: 1 -> A | _ -> B')` produces a `Match` node
- `test_range_pattern_parses` — `parse_expr('match X: 1..3 -> A | _ -> B')` produces a `RangePattern`
- `test_wildcard_required` — missing wildcard raises `SyntaxError`
- `test_wildcard_must_be_last` — `match X: _ -> A | 1 -> B` raises `SyntaxError`

**PMF correctness:**

- `test_exact_pmf` — `match d6: 1 -> 10 | _ -> 0` → P(10) == 1/6, P(0) == 5/6
- `test_range_pmf` — `match d6: 1..3 -> 1 | _ -> 0` → P(1) == 1/2
- `test_all_arms_covered` — PMF always sums to 1.0
- `test_scrutinee_with_dice` — `match d6 + d6: 2 -> 1 | _ -> 0` → P(1) == 1/36
- `test_match_in_function` — `f(X): match X: 1 -> 10 | _ -> 0; f(d6)`
- `test_simplify_compatible` — `run_simplified(expr)` matches `run(expr)` for all match expressions
