# Reproducibility

Every scenario is generated from an explicit integer seed. Estimator defaults were set on development
seeds `1001+`; reported results use seeds `0-2`.

| result | command | approx. time |
|---|---|---|
| benchmark tables and figure (`docs/benchmark/`) | `oaqe-benchmark --seeds 3 --duration 400 --jobs 4 --out results/benchmark` | 15 min, 4 cores |
| SUMO co-simulation check | `oaqe-sumo --sumocfg examples/sumo/demo.sumocfg --lanes app_E_0 app_E_1 --steps 24000 --out results/sumo.csv` | 5 min, 1 core |
| animation (`docs/assets/queue.gif`) | `oaqe-gif --out docs/assets/queue.gif --seed 4` | 1 min |

Environment: Python 3.12, versions in `requirements-lock.txt`, SUMO 1.28.0 (`pip install -e ".[sumo]"`).
Re-running the benchmark with the same package versions reproduces the reported figures exactly
(`docs/benchmark/manifest.json` records versions, specs and defaults). `make all` runs ruff, mypy and
pytest; CI runs the same on Python 3.10-3.12.
