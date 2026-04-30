# Dice Probabilities

An exact probability evaluator for DnD-style dice expressions. Given an expression involving dice and variables, it computes the full **probability mass function (PMF)** and **expected value** by enumerating all possible outcomes.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
python -m dice "EXPRESSION"
```

### Examples

```bash
# Single die
python -m dice "d6"

# Sum of two dice
python -m dice "d6 + d6"

# Conditional threshold
python -m dice "e(X): X > 3 -> X | 0; e(d6)"

# Boolean condition
python -m dice "e(X, Y): X > 3 && Y > 3 -> X + Y | 0; e(d6, d6)"

# Nested conditional
python -m dice "e(X, Y): X > 3 -> (Y > 3 -> X*Y | X) | Y; e(d6, d6)"
```

**Output format** — one line per outcome sorted ascending, then the expected value:

```
0: 50.00%
4: 16.67%
5: 16.67%
6: 16.67%
(E): 2.5
```

---

## Expression Syntax

There are two forms: **named** and **bare**.

### Named form

```
e(VAR, ...): EXPR; e(dN, ...)
```

Declares named variables and binds each to a die.

```
e(X, Y): X + Y > 6 -> X*2 + Y | X + 2*Y; e(d6, d6)
```

### Bare form

An expression with inline dice — no wrapper needed.

```
d6 + d6
d6 > 3 -> d4 | 0
d6 > 3 && d6 > 3 -> 1 | 0
```

> **Important:** each `dN` occurrence is an independent roll. `d6 > 3 -> d6 | 0` uses *two separate dice* — one for the condition and one for the result. To reuse the same roll, use the named form: `e(X): X > 3 -> X | 0; e(d6)`.

---

## Grammar

```
full    := "e(" NAME,... ")" ":" expr ";" "e(" dN,... ")"
         | expr

expr    := ifelse
ifelse  := orelse ("->" orelse "|" orelse)?
orelse  := andexpr ("||" andexpr)*
andexpr := compare ("&&" compare)*
compare := add (("<" | ">" | "<=" | ">=" | "==" | "!=") add)?
add     := mul (("+" | "-") mul)*
mul     := atom ("*" atom)*
atom    := INT | NAME | dN | "(" expr ")"
```

### Operator precedence (tightest to loosest)

| Level | Operators |
|-------|-----------|
| Arithmetic | `*` then `+` `-` |
| Comparison | `<` `>` `<=` `>=` `==` `!=` |
| Boolean AND | `&&` |
| Boolean OR | `\|\|` |
| Conditional | `cond -> then \| else` |

Parentheses override precedence at any level, including nesting conditionals:

```
e(X, Y): X > 3 -> (Y > 3 -> X*Y | X) | Y; e(d6, d6)
```

---

## Syntax Reference

### Dice

| Syntax | Meaning |
|--------|---------|
| `d6` | A fair six-sided die (faces 1–6) |
| `d20` | A fair twenty-sided die (faces 1–20) |
| `dN` | Any positive integer N |

In the named form, dice appear in the `e(dN, ...)` binding list. In expressions (named or bare), `dN` tokens are also allowed inline as anonymous dice.

### Arithmetic

```
X + Y        # addition
X - Y        # subtraction
X * Y        # multiplication
2*X + 1      # precedence: multiplication before addition
2*(X + 1)    # parentheses override precedence
```

### Comparisons

```
X > 3        # greater than
X < Y        # less than
X >= 3       # greater than or equal
X <= Y       # less than or equal
X == Y       # equal
X != 3       # not equal
```

Results are boolean (True/False), usable as conditional tests.

### Boolean operators

```
X > 3 && Y > 3       # both must be true
X > 4 || Y > 4       # at least one must be true
```

`&&` binds tighter than `||`. Both bind tighter than `->`.

```
# A || B && C  parses as  A || (B && C)
e(X, Y, Z): X > 4 || Y > 3 && Z > 3 -> X + Y + Z | 0; e(d6, d6, d6)
```

### Conditional expression

```
cond -> then | else
```

Evaluates `cond`; if truthy, returns `then`, otherwise `else`. Both branches can be arbitrary expressions including nested conditionals (wrap inner ones in parentheses):

```
X > 3 -> (Y > 3 -> X*Y | X) | Y
```

---

## Architecture

The evaluator is built as a clean pipeline:

```
text
  ↓  parse()
(Expr AST, var_domains)
  ↓  build_pmf()
dict[int, float]   ← PMF
```

### Modules

| Module | Responsibility |
|--------|---------------|
| `ast_nodes.py` | Frozen dataclass AST nodes (`Const`, `Var`, `Die`, `Add`, `Sub`, `Mul`, `Compare`, `IfElse`, `And`, `Or`) |
| `evaluator.py` | `eval_expr(expr, env)` — pure evaluation given a concrete variable assignment |
| `engine.py` | `build_pmf(expr, domains)` — joint enumeration over all variable assignments |
| `parser.py` | Recursive-descent parser; `parse()` and `parse_expr()` |
| `cli.py` | Command-line interface |

### How it works

The engine enumerates the Cartesian product of all variable domains. For each assignment it evaluates the expression and accumulates the probability:

```python
for assignment in product(domains):
    p = product(probabilities)
    value = eval_expr(expr, assignment)
    result[value] += p
```

This is **exact** (no sampling) and **exponential** in the number of variables — `O(kⁿ)` where `k` is die size and `n` is variable count. For typical DnD expressions (2–5 dice) it is instant.

---

## Known Limitations

- No division or modulo (no AST nodes for them)
- No unary minus — use `0 - X` as a workaround
- No boolean short-circuit in the PMF sense: all branches are always evaluated structurally (though `eval_expr` itself does short-circuit `&&`/`||` correctly)
- Anonymous dice (`dN`) cannot be "the same roll" — each occurrence is an independent variable. Use the named form to reuse a roll across the expression.

---

## Running Tests

```bash
pytest tests/ -v
```

328 tests covering: AST nodes, evaluator, engine (PMF axioms, known expected values, symmetry), parser (precedence, all operators, error cases), integration (full pipeline, property checks, CLI output), and extreme multi-variable cases.
