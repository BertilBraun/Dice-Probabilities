# Dice Probabilities

When you roll two dice and the result of one die affects what the other die contributes, standard probability tables break down. This tool handles those cases exactly.

The core idea: bind dice rolls to named variables via functions, then write expressions over those variables. The engine enumerates every combination of outcomes, evaluates the expression for each, and accumulates exact probabilities.

For example:

```text
f(X, Y): X + Y > 6 -> 2*X | X - Y
f(d6, d6)
```

This asks: roll two d6s — if their sum exceeds 6, score double the first die; otherwise score the difference. The answer isn't a single number but a full distribution, because X and Y appear in both the condition and the result branches.

```text
-4: 0.56%    -3: 1.39%    -2: 2.50%    ...
 8: 2.78%    10: 5.56%    12: 2.78%
(E): 3.78
```

Functions compose, and each call is independent:

```text
bonus(X): X + d6
attack(X): bonus(X) + bonus(X)
attack(d6)
```

Here `attack(d6)` expands to `(arg + d6_1) + (arg + d6_2)` — the argument die is rolled once and shared, while the `d6` inside `bonus` is a fresh independent roll on each call. Every possible combination of rolls is considered; the result is always exact, never sampled.

## Installation

```bash
pip install -e .
```

Requires Python 3.10+.

## Usage

```bash
python -m dice "EXPRESSION"
```

Multi-line programs can be passed as a single quoted string:

```bash
python -m dice "bonus(X): X + d6
attack(X): bonus(X) + bonus(X)
attack(d6)"
```

**Output format** — one line per outcome sorted ascending, then the expected value:

```text
0: 50.00%
4: 16.67%
5: 16.67%
6: 16.67%
(E): 2.5
```

---

## Expression Syntax

A program is a sequence of **function definitions** followed by exactly one **expression** to evaluate. Statements are separated by newlines or `;`.

```
name(param, ...): expr
name(param, ...): expr
final_expr
```

### Function definitions

```
threshold(X): X > 3 -> X | 0
add(X, Y): X + Y
combo(X, Y): threshold(X) + add(X, Y)
```

Each parameter is bound **once** at the call site. If a parameter appears multiple times in the body, all occurrences see the same roll. Inline dice in function bodies (`dN`) are fresh and independent on every call.

### Final expression

The last statement is the expression to evaluate — a function call or any bare expression:

```
threshold(d6)
add(d6, d4)
d6 + d6
```

### Bare expressions

No function definitions are required. Inline dice work directly:

```bash
python -m dice "d6 + d6"
python -m dice "d6 > 3 -> d4 | 0"
```

Each `dN` occurrence is an independent roll. To reuse the same roll, bind it to a parameter.

---

## Grammar

```text
program := (definition ";"|"\n")* expr

definition := NAME "(" NAME ("," NAME)* ")" ":" expr

expr    := ifelse
ifelse  := orelse ("->" orelse "|" orelse)?
orelse  := andexpr ("||" andexpr)*
andexpr := compare ("&&" compare)*
compare := add (("<" | ">" | "<=" | ">=" | "==" | "!=") add)?
add     := mul (("+" | "-") mul)*
mul     := atom ("*" atom)*
atom    := INT | NAME | dN | NAME "(" expr ("," expr)* ")" | "(" expr ")"
```

### Operator precedence (tightest to loosest)

| Level       | Operators                   |
| ----------- | --------------------------- |
| Arithmetic  | `*` then `+` `-`            |
| Comparison  | `<` `>` `<=` `>=` `==` `!=` |
| Boolean AND | `&&`                        |
| Boolean OR  | `\|\|`                      |
| Conditional | `cond -> then \| else`      |

Parentheses override precedence at any level.

---

## Syntax Reference

### Dice

| Syntax | Meaning                              |
| ------ | ------------------------------------ |
| `d6`   | A fair six-sided die (faces 1–6)     |
| `d20`  | A fair twenty-sided die (faces 1–20) |
| `dN`   | Any positive integer N               |

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
X > 3    X < Y    X >= 3    X <= Y    X == Y    X != 3
```

Results are boolean (0 or 1), usable in conditional tests.

### Boolean operators

```
X > 3 && Y > 3       # both must be true
X > 4 || Y > 4       # at least one must be true
```

`&&` binds tighter than `||`. Both bind tighter than `->`.

### Conditional expression

```
cond -> then | else
```

Evaluates `cond`; if truthy returns `then`, otherwise `else`. Branches can be arbitrary expressions including nested conditionals (wrap inner ones in parentheses):

```
X > 3 -> (Y > 3 -> X*Y | X) | Y
```

---

## Architecture

The evaluator is built as a pipeline:

```
text
  ↓  parse()
     split into statements
     build function registry
     expand Call nodes (macro substitution)
     extract Die nodes → named variables
(Expr AST, var_domains)
  ↓  simplify()
     annotate every node with its variable set
     collapse independent subtrees into virtual variables
(simplified Expr, reduced var_domains)
  ↓  build_pmf()
dict[int, float]   ← PMF
```

### How function expansion works

Function calls are resolved at parse time by macro substitution — the engine never sees them. When `f(d6)` is called:

1. The `d6` argument is extracted to a fresh variable (e.g. `_d0`) **before** substitution, so every occurrence of the parameter in the body refers to the same roll.
2. The body is substituted with `param → _d0`.
3. Any `dN` tokens inside the body become fresh variables at this point.
4. The result is expanded recursively for nested calls.

This guarantees that `double(X): X + X; double(d6)` produces `{2,4,6,8,10,12}` each with P=1/6 (one roll, doubled) — not the distribution of two independent dice.

### Modules

| Module         | Responsibility                                                                          |
| -------------- | --------------------------------------------------------------------------------------- |
| `ast_nodes.py` | Frozen dataclass AST nodes for all expression types                                     |
| `evaluator.py` | `eval_expr(expr, env)` — pure evaluation given a concrete variable assignment           |
| `engine.py`    | `build_pmf(expr, domains)` — joint enumeration over all variable assignments            |
| `simplify.py`  | `simplify(expr, domains)` — AST simplification pass; collapses independent subtrees     |
| `parser.py`    | Recursive-descent parser, macro expansion, die extraction; `parse()` and `parse_expr()` |
| `cli.py`       | Command-line interface                                                                  |

### How build_pmf works

After expansion, every die in the expression is a distinct named variable with a domain of `(face, probability)` pairs. `build_pmf` takes the Cartesian product of all domains, evaluates the expression for each combination, and accumulates probability into a result map:

```python
for outcome in product(domain_lists):
    variable_values = {name: face for name, (face, _) in zip(variable_names, outcome)}
    joint_probability = product(p for _, p in outcome)
    result[eval_expr(expr, variable_values)] += joint_probability
```

Each outcome is weighted by the product of its individual face probabilities (uniform dice all have p=1/N, so the joint probability is 1/N₁·N₂·…). The result is an exact PMF — no sampling, no floating-point shortcuts.

### Complexity

Exact enumeration is `O(kⁿ)` where `k` is die size and `n` is the number of distinct die variables after expansion. For typical DnD expressions (2–5 dice) this is instant.

---

## Known Limitations

- Functions must be defined before use (top-to-bottom only)
- No recursion (detected and rejected at parse time)
- Exponential blow-up in the number of dice: `build_pmf` enumerates the full Cartesian product of all die variables, so complexity is `O(kⁿ)` where `n` is the number of distinct dice after expansion. An expression like `d6 > 5 -> 50d20 | 0` is intractable even though the result is conceptually simple.

  The correct fix is to preserve exact results — an alternative that computes distributions symbolically (propagating PMFs rather than concrete values through the AST) would handle independent subexpressions via convolution, reducing `n`-dice addition to `O(n · k²)` instead of `O(kⁿ)`. The challenge is that variables shared across a conditional's condition and branches require joint enumeration to remain correct; a dependency analysis pass over the AST would identify which subgraphs are truly independent and apply convolution only there, falling back to enumeration where variables are shared. Without that analysis, the symbolic approach silently produces wrong results for shared-variable expressions — trading correctness for feasibility.

---

## Running Tests

```bash
pytest tests/ -v
```

352 tests covering: AST nodes, evaluator, engine (PMF axioms, known expected values, symmetry), parser (precedence, all operators, error cases), functions (binding semantics, independence, composition, error cases), integration (full pipeline, property checks, CLI output), and extreme multi-variable cases.
