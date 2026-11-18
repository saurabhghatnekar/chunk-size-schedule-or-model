# Milestone 1. Later milestones add: pilot | grid | ablations | frontier | reproduce
.PHONY: control unit figures clean-results

# The equivalence test, its controls, the classification and the matrix. No training; a few minutes on CPU.
control:
	uv run pytest -q
	uv run python -u -m src.equivalence.test | tee results/logs/equivalence.log
	uv run python -m src.equivalence.controls
	uv run python -m src.equivalence.classify
	uv run python -m src.equivalence.matrix

unit:
	uv run pytest -q

# Regenerate documents and figures from the existing CSV without rerunning the sweep.
figures:
	uv run python -m src.equivalence.classify
	uv run python -m src.equivalence.matrix
