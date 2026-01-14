
## RSP-CEGIS

Counterexample-Guided Inductive Synthesis (CEGIS) for discovering sound pruning rules for the runway sequencing problem, using:
- a small AST language for candidate rules (`pr_ast.py`)
- a genetic program search over that AST (`genetic.py`)
- an SMT-backed semantics / verifier for a specific scheduling “form” (`form.py`, Z3 Theorem Prover)

The goal of this experiment is to automatically synthesise boolean conditions β that (when they hold) certify that swapping two aircraft in a partial order cannot improve the objective (here: delay cost), so the branch can be pruned safely. 

> **Note:** RSP-CEGIS is sound but incomplete; it may fail to discover many valid pruning rules due to the heuristic nature of the genetic search.

### Getting Started
RSP-CEGIS is executed from the command line using python main.py. Parameters controlling the CEGIS loop and genetic search (such as the number of rounds, population size, and generations) may be provided via standard GNU-style flags, or collected interactively using the -i option. The tool outputs a set of formally verified pruning rules, if any are discovered, under the given search budget
- Install Python 3.10+
- Run `pip install -r requirements.txt` to install dependencies (Z3, Pydantic, TQDM)
- Run `python main.py -h` for help or `python main.py --interactive` to get started quickly.

### CLI Flags
| CLI Flag              | Type | Description                                                 |
|-----------------------|-----|-------------------------------------------------------------|
| `-h`, `--help`        | — | Show help message and exit                                  |
| `-v`, `--version`     | — | Show program version and exit                               |
| `-i`, `--interactive` | — | Run in interactive prompt mode (arguments taken in the CLI) |
| `--max-rounds`        | `int > 0` | Number of outer CEGIS iterations                            |
| `--starting`          | `int > 0` | Initial genetic population size                             |
| `--generations`       | `int > 0` | Genetic generations per CEGIS round                         |
| `--elite`             | `int > 0` | Elite rules preserved each generation                       |
| `--target-solutions`  | `int` \| `inf` | Stop after this many sound rules                            |
