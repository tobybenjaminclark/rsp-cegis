
# RSP-CEGIS

Counterexample-Guided Inductive Synthesis (CEGIS) for discovering **sound pruning rules** in runway sequencing, using:
- a small AST language for candidate rules (`pr_ast.py`)
- a genetic program search over that AST (`genetic.py`)
- an SMT-backed semantics / verifier for a specific scheduling “form” (`form.py`, Z3)

The goal is to automatically synthesise boolean conditions β that (when they hold) certify that swapping two aircraft in a partial order **cannot improve** the objective (here: delay cost), so the branch can be pruned safely.