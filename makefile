.PHONY: print activate test eval eval-llm check topiocqa

PY = /workspace/.venv/bin/python

# default target: run by the dev container on start
print:
	@echo "source /workspace/.venv/bin/activate"

activate:
	@. /workspace/.venv/bin/activate

# unit tests (offline: the LLM is replaced by a fake)
test:
	$(PY) -m pytest -q

# routing accuracy on every eval set, one table (offline, rules only)
eval:
	$(PY) eval/summary.py

# the same, with the LLM review of TENTATIVE placements (needs LLM_MODEL, LLM_API_KEY).
# Answers are cached in eval/llm_cache/; TopiOCQA answers end up there too, so
# do not commit them (the dataset is CC BY-NC-SA 4.0).
eval-llm:
	$(PY) eval/summary.py --llm

# after changing code or datasets: unit tests, then the accuracy table
check: test eval

# download and convert TopiOCQA into eval/scenarios-topiocqa/ (git-ignored)
topiocqa:
	$(PY) tools/build_topiocqa_eval.py
