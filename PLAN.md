# Dice Expression Evaluator – Project Plan

## 1. Goal

Build an exact evaluator for dice expressions (DnD-style) that produces:

* Full probability mass function (PMF)
* Expected value
* Extensible architecture for later optimization

Initial focus: **correctness over performance**

---

## 2. Core Approach

Model everything as:

> A deterministic function over a finite probability space

Execution strategy:

1. Parse expression → AST
2. Identify variables
3. Enumerate joint assignments (baseline)
4. Evaluate expression per assignment
5. Aggregate probabilities

This is exponential, but **simple and correct**

---

## 3. Scope (v1)

### Supported Features

* Dice: `d6`, `d20`, etc.
* Multiple variables: `e(X, Y)`
* Arithmetic: `+`, `*`
* Comparisons: `>`, `<`, `>=`, etc.
* Conditionals:
  `cond -> A | B`
* Constants

### Explicitly NOT supported (v1)

* Performance optimizations
* Symbolic simplification
* Dependency analysis
* FFT/convolution
* Continuous distributions

---

## 4. Core Data Structures

### 4.1 Random Variable

```python
class RV:
    pmf: dict[int, float]
```

Used only for final results (not internal evaluation).

---

### 4.2 Expression Tree (AST)

```python
class Expr:
    pass

class Const(Expr):
    value: int

class Var(Expr):
    name: str

class Die(Expr):
    sides: int

class Add(Expr):
    left: Expr
    right: Expr

class Mul(Expr):
    left: Expr
    right: Expr

class Compare(Expr):
    left: Expr
    op: str
    right: Expr

class IfElse(Expr):
    condition: Expr
    then_branch: Expr
    else_branch: Expr
```

---

## 5. Evaluation Model

### 5.1 Variable Binding

Example:

```text
e(X, Y); e(d6, d6)
```

→ environment:

```python
{
  X: d6,
  Y: d6
}
```

---

### 5.2 Joint Enumeration (baseline engine)

```python
for assignment in product(domains):
    p = product(probabilities)
    value = eval_expr(expr, assignment)
    result[value] += p
```

---

### 5.3 Expression Evaluation

```python
def eval_expr(expr, env):
    match expr:
        case Const(v): return v
        case Var(name): return env[name]
        case Add(a, b): return eval_expr(a, env) + eval_expr(b, env)
        case Mul(a, b): return eval_expr(a, env) * eval_expr(b, env)
        case Compare(a, op, b): return ...
        case IfElse(cond, t, f):
            if eval_expr(cond, env):
                return eval_expr(t, env)
            else:
                return eval_expr(f, env)
```

---

## 6. CLI Interface

Example:

```bash
python dice.py "e(X): X > 3 -> X | 0; e(d6)"
```

Output:

```text
0: 50%
4: 16.67%
5: 16.67%
6: 16.67%
(E): 2.5
```

---

## 7. Validation Strategy

### 7.1 Unit Tests

* Single die
* Sum of dice
* Simple threshold
* Dependent condition:

```text
X > 3 -> X | 0
```

* Multi-variable:

```text
X + Y > 6 -> X*2 + Y | X + 2*Y
```

---

### 7.2 Property Tests

* PMF sums to 1
* No negative probabilities
* Expectation matches manual calculation

---

## 8. Known Limitations (v1)

* Time complexity: O(k^n)
* Memory grows with output range
* No reuse of subcomputations
* No independence detection

---

## 9. Planned Optimizations (v2+)

### 9.1 Convolution for independent sums

Replace enumeration for:

```text
d6 + d6 + d6
```

---

### 9.2 Map optimization

Detect:

```text
f(X)
```

→ avoid joint expansion

---

### 9.3 Partial factorization

Rewrite:

```text
S = A + T
```

to reduce dimensionality

---

### 9.4 Memoization

Cache:

```text
eval_expr(subtree, partial_env)
```

---

### 9.5 Hybrid execution

* Small variable count → exact
* Large variable count → fallback to Monte Carlo

---

## 10. Milestones

### M1 – Minimal Engine

* AST
* Parser (basic)
* Joint enumeration
* CLI

### M2 – Correctness

* Test suite
* Edge cases
* Floating point stability

### M3 – First Optimization

* Convolution for sums
* Map detection

### M4 – Advanced

* Partial factorization
* Performance profiling

---

## 11. Guiding Principle

> Do not prematurely optimize.
> Build the slow, correct engine first.
> Then optimize based on structure.

---

## 12. Exit Criteria (v1 complete)

* Correct results for all tested expressions
* Handles dependent conditionals correctly
* Clean AST + evaluator separation
* Ready for optimization layer

---
