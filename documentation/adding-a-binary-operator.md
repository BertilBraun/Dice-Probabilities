# Adding a Full-Pipeline Feature: `min`/`max` Operators

This document walks through designing `min(expr, expr)` and `max(expr, expr)` as first-class binary operators, plus `adv(die)` and `dis(die)` as ergonomic sugar. It is an example of a **full-pipeline extension**: the new AST node types survive through the evaluator, engine, and simplify pass — and are handled in each layer with **zero new code**, because the `BinaryNode.apply` protocol already covers them.

Compare this to [adding-a-parser-feature.md](adding-a-parser-feature.md), where a feature desugars at parse time and nothing new reaches the runtime layers. Understanding both patterns makes clear where integration cost is paid.

---

## Motivation

Two common dice mechanics are outside reach of the current language:

- **Advantage/disadvantage** (D&D): roll a die twice, take the higher or lower result. `d6 > 3 -> 2*d20 | d20` is not the same thing — it rolls different numbers of dice.
- **Damage cap**: `d12` can never exceed 8 due to some effect. Without `min`, this needs a comparison expression that produces boolean 0/1, not the capped value.

`min` and `max` fill both gaps:

```text
max(d20, d20)          # advantage roll
min(d12, 8)            # d12 capped at 8
adv(d20) + 5           # shorthand for max(d20, d20) + 5
dis(d8)                # shorthand for min(d8, d8)
f(X): max(X, 4); f(d6) # reuse same roll, keep at least 4
```

---

## Syntax

```text
min(EXPR, EXPR)   # permanent binary operator: take the lower value
max(EXPR, EXPR)   # permanent binary operator: take the higher value
adv(DIE)          # sugar for max(DIE, DIE) with two independent rolls
dis(DIE)          # sugar for min(DIE, DIE) with two independent rolls
```

`min`/`max` take any two expressions. `adv`/`dis` are restricted to single die expressions because the semantic requires rolling that die twice independently — passing a complex expression would be ambiguous.

---

## Two kinds of new node

This design introduces **two different kinds** of node, which illustrates the distinction cleanly:

| Node | Kind | Lifetime |
| --- | --- | --- |
| `Min`, `Max` | Permanent `BinaryNode` subclasses | Survive to evaluator, engine, simplify |
| `Advantage`, `Disadvantage` | Transient `Expr` subclasses | Removed by `_extract_dice`, like `Die` |

`Min`/`Max` are permanent because "take the lower/higher of two independent values" is a meaningful operation at evaluation time — `eval_expr` needs to compute it. `Advantage`/`Disadvantage` are transient because their semantic ("roll this die twice, independently") is a **variable-binding** concern that only the parser stage can resolve.

---

## New AST nodes (`src/dice/ast_nodes.py`)

### Permanent nodes

```python
@dataclass(frozen=True)
class Min(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return min(left_val, right_val)

@dataclass(frozen=True)
class Max(BinaryNode):
    def apply(self, left_val: int, right_val: int) -> int:
        return max(left_val, right_val)
```

That is the entire implementation. Everything else follows from the `BinaryNode` protocol.

### Transient nodes

```python
@dataclass(frozen=True)
class Advantage(Expr):
    inner: Expr

@dataclass(frozen=True)
class Disadvantage(Expr):
    inner: Expr
```

---

## Parser changes (`src/dice/parser.py`)

### `_parse_atom` — intercept built-in names

Before the generic `Call` path in the `NAME` branch, intercept `min`, `max`, `adv`, `dis`:

```python
if token.value in ('min', 'max', 'adv', 'dis'):
    self._consume(TokenKind.PUNCT, '(')
    first = self.parse_expr()
    if token.value in ('min', 'max'):
        self._consume(TokenKind.PUNCT, ',')
        second = self.parse_expr()
        self._consume(TokenKind.PUNCT, ')')
        return Min(first, second) if token.value == 'min' else Max(first, second)
    else:
        self._consume(TokenKind.PUNCT, ')')
        return Advantage(first) if token.value == 'adv' else Disadvantage(first)
```

`Min`/`Max` are returned directly as permanent nodes. `Advantage`/`Disadvantage` are returned as transient nodes to be resolved in `_extract_dice`.

### `_substitute` and `_expand` — add `Advantage`/`Disadvantage` cases

`Min`/`Max` are `BinaryNode` subclasses, so the existing `case BinaryNode(left=left, right=right)` arm handles them in both tree-walkers with no new code.

`Advantage`/`Disadvantage` need one new case each (same shape in both walkers):

```python
case Advantage(inner=inner):
    return Advantage(<recurse>(inner, ...))
case Disadvantage(inner=inner):
    return Disadvantage(<recurse>(inner, ...))
```

### `_extract_dice` — desugar `Advantage`/`Disadvantage`

This is where the two independent rolls are created. `_extract_dice` is called **twice** on the same inner expression, which allocates two distinct variable names pointing to two independent domains:

```python
case Advantage(inner=inner):
    left = _extract_dice(inner, die_counter, domains)
    right = _extract_dice(inner, die_counter, domains)
    return Max(left, right)
case Disadvantage(inner=inner):
    left = _extract_dice(inner, die_counter, domains)
    right = _extract_dice(inner, die_counter, domains)
    return Min(left, right)
```

After this, `adv(d6)` has become `Max(Var('_d0'), Var('_d1'))` with two d6 entries in `domains`. From this point it is indistinguishable from `max(d6, d6)` written explicitly — which is the correct semantic.

---

## How the pipeline handles `Min`/`Max` without new code

### Evaluator (`eval_expr`)

The existing dispatch:

```python
case BinaryNode(left=left, right=right):
    return expr.apply(eval_expr(left, env), eval_expr(right, env))
```

calls `Min.apply` or `Max.apply`, which return `min(l, r)` or `max(l, r)`. No new case needed.

### Engine (`build_pmf`)

`build_pmf` iterates over the Cartesian product of all variable domains and calls `eval_expr` for each combination. Since `eval_expr` handles `Min`/`Max`, `build_pmf` handles them too with no changes.

For `max(d6, d6)` this produces 36 combinations. `eval_expr` returns `max(i, j)` for each pair — the resulting PMF has P(6) = 11/36, P(5) = 9/36, etc.

### Simplify

`_try_binary` handles any `BinaryNode`:

```python
case BinaryNode():
    return _try_binary(node, original_vars, forbidden)
```

If `Min`/`Max`'s two children are independent, `_try_binary` collapses them via:

```python
_combine_pmf(node, left_pmf, right_pmf)
```

which falls through to:

```python
case _:
    return _pairwise(left_pmf, right_pmf, node.apply)
```

`_pairwise` computes `P(min(A,B) = k)` by summing `P(A=a) * P(B=b)` over all pairs where `min(a,b) = k` — the correct formula for the minimum of two independent random variables. No new code needed.

So `adv(d20) + d6` simplifies: `Max(Var('_d0'), Var('_d1'))` collapses to a virtual variable (the advantage PMF), then that virtual variable and `d6` collapse to a single PMF. `build_pmf` sees one virtual variable.

---

## Files changed

| File | Change |
| --- | --- |
| `src/dice/ast_nodes.py` | Add `Min`, `Max`, `Advantage`, `Disadvantage` |
| `src/dice/parser.py` | Intercept `min`/`max`/`adv`/`dis` in `_parse_atom`; add `Advantage`/`Disadvantage` cases to `_substitute`, `_expand`, `_extract_dice` |
| `src/dice/evaluator.py` | No changes |
| `src/dice/engine.py` | No changes |
| `src/dice/simplify.py` | No changes |

---

## Test outline (`tests/test_min_max.py`)

**Parser / structural:**

- `test_min_parses` — `parse_expr('min(X, Y)')` returns `Min(Var('X'), Var('Y'))`
- `test_max_parses` — `parse_expr('max(X, 4)')` returns `Max(Var('X'), Const(4))`
- `test_adv_parses` — `parse_expr('adv(X)')` returns `Advantage(Var('X'))`
- `test_adv_extracts_two_independent_vars` — `parse('adv(d6)')` produces two distinct domains

**Evaluator:**

- `test_min_takes_lower` — `eval_expr(Min(Const(3), Const(5)), {})` returns `3`
- `test_max_takes_higher` — `eval_expr(Max(Const(3), Const(5)), {})` returns `5`

**PMF correctness:**

- `test_min_d6_d6_pmf` — `min(d6, d6)`: P(1) == 11/36, sums to 1.0
- `test_max_d6_d6_pmf` — `max(d6, d6)`: P(6) == 11/36, sums to 1.0
- `test_adv_matches_max` — `run('adv(d6)')` equals `run('max(d6, d6)')`
- `test_dis_matches_min` — `run('dis(d6)')` equals `run('min(d6, d6)')`
- `test_min_const_cap` — `min(d6, 4)`: outcomes are 1–4; P(4) == 3/6
- `test_simplify_compatible` — `run_simplified(expr)` matches `run(expr)` for all expressions above
