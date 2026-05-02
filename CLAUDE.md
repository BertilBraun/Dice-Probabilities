# Coding Style Guidelines

## Comments

- Default to writing **no comments**. Only add one when the WHY is non-obvious: a hidden constraint, a subtle invariant, a workaround for a specific bug, or behavior that would surprise a reader.
- Never write what the code does — well-named identifiers already say that.
- No multi-paragraph docstrings or multi-line comment blocks. One short line max.
- No task or PR references ("added for X", "handles case from issue #123") — those belong in commit messages and rot as the codebase evolves.

## Python specifics

- Use `nonlocal name` for mutable closure state — never the `name = [0]` list trick.
- Avoid single-character variable names that are visually ambiguous (`l`, `O`, `I`).
- Prefer `match`/`case` over `isinstance` chains for AST node dispatch.
- Frozen dataclasses for AST nodes — immutable by default.
- Type hints on all public functions; internal closures can omit them when obvious from context.

## Error handling

- No defensive checks for scenarios that can't happen. Trust internal invariants.
- Only validate at system boundaries (user input, external APIs).
- `assert` for invariants that signal bugs; `raise ValueError` for invalid user input.

## Testing

- Tests call production code directly — no mocking internal modules.
- One logical assertion per test name. `test_two_independent_dice_collapse_to_one` beats `test_simplify_1`.
- Parametrize over similar cases with `@pytest.mark.parametrize`.

## Architecture

- Two-pass design: `simplify(expr, domains)` → `build_pmf(expr, domains)`. Keep passes separate and composable.
- `build_pmf` is a pure enumeration primitive — no optimisation logic inside it.
- `simplify` is called explicitly in `cli.py`; tests that want the optimised path call both.
