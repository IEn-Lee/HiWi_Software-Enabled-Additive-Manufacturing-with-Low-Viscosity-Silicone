# Progress-bar update (v1.2)

This version keeps the optimizer/FIFO/LP formulation unchanged and adds a dependency-free console workflow progress bar.

## What the percentage means

The displayed percentage is **overall workflow progress**, based on completed phases such as CSV loading, control construction, FIFO equation construction, Stage 1, Stage 2, output writing, G-code patching, and TRUE FIFO validation.

`scipy.optimize.linprog(method="highs")` does not expose a reliable live solver-completion percentage. Therefore, while Stage 1 or Stage 2 is inside HiGHS, the bar holds at the current workflow percentage and displays a continuously updated elapsed-time heartbeat. It does not fabricate a solver percentage.

Example:

```text
[###########---------------------]  34.00% | Building Stage 1 minimax LP
[############--------------------]  38.00% | Stage 1 free-segment minimax | HiGHS solving; internal solver % unavailable | elapsed 21.0s
[###################-------------]  58.00% | Stage 1 free-segment minimax complete | elapsed 35.4s
```

For multiple `--iterations`, each iteration is mapped into its share of the global 0–100% workflow range.

## New CLI options

- `--no_progress`: disable the progress display.
- `--progress_width 32`: set progress-bar character width.
- `--progress_interval_sec 1.0`: heartbeat refresh interval while HiGHS is solving.

No new Python package is required.
