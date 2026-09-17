import time

from indrajala_ml.benchmark_sweep import estimate_sweep_wallclock, run_parameter_sweep, summarize_sweep_results


def _toy_worker(config: int, seed: int) -> float:
    # deterministic pure function of (config, seed) - no real training, matching this project's
    # agreed validation approach (see docs/architecture/benchmarking.md): prove the runner's own mechanics
    # (dispatch, seeding, aggregation) work, not reprove a real measurement
    return config * 10.0 + seed


def _sleepy_worker(config: int, seed: int) -> float:
    time.sleep(0.05)
    return float(config + seed)


def test_run_parameter_sweep_dispatches_every_config_seed_pair_exactly_once():

    results = run_parameter_sweep(configs=[1, 2, 3], seeds=[0, 1], worker_fn=_toy_worker, report_progress=False)

    assert set(results.keys()) == {1, 2, 3}
    for config in (1, 2, 3):
        assert sorted(results[config]) == sorted(config * 10.0 + seed for seed in (0, 1))


def test_run_parameter_sweep_result_count_matches_configs_times_seeds():

    results = run_parameter_sweep(configs=[5, 9], seeds=[0, 1, 2], worker_fn=_toy_worker, report_progress=False)

    for config in (5, 9):
        assert len(results[config]) == 3


def test_estimate_sweep_wallclock_projects_from_a_single_real_run():

    projected = estimate_sweep_wallclock(
        _sleepy_worker, sample_config=0, sample_seed=0, planned_run_count=20, worker_count=4
    )

    # 20 jobs / 4 workers = 5 serial batches, each taking roughly _sleepy_worker's own ~0.05s
    assert 0.15 <= projected <= 0.5


def test_summarize_sweep_results_renders_a_markdown_table():

    table = summarize_sweep_results({"lr=0.1": [0.9, 0.92, 0.94], "lr=0.5": [0.5, 0.5, 0.5]})

    lines = table.splitlines()
    assert lines[0] == "| config | mean | stdev |"
    assert lines[1] == "|---|---|---|"
    assert any(line.startswith("| lr=0.1 |") for line in lines)
    assert any(line.startswith("| lr=0.5 | 50.00% | 0.00% |") for line in lines)
