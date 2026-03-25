# Optimized GNBG-II evaluator
# Key speedups:
#   1. exec() runs ONCE per worker process (via initializer)
#   2. Half-budget fids (1-15) batched in packs of 3 reps to reduce overhead
#   3. Full-budget fids (16-24) stay as individual tasks (slower, benefit from fine scheduling)
#   4. as_completed() streams results without waiting for ordering

from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import os
import re
import signal
from statistics import geometric_mean, harmonic_mean
import sys
import time
import traceback

import numpy as np
import multiprocessing as mp
from GNBG_Runners.GNBG_II.GNBG_instances import get_gnbg

REPETITIONS_PER_FID = 30
HALF_BUDGET_PACK_SIZE = 3  # pack fast tasks in groups of 3

# ── Per-worker global state (initialized once per process) ────────
_worker_metaheuristic = None
_worker_algorithm_name = None


def _worker_init(code, algorithm_name):
    """Called once when each worker process starts. Exec's code only once."""
    global _worker_metaheuristic, _worker_algorithm_name
    _worker_algorithm_name = algorithm_name
    shared_env = {"__builtins__": __builtins__}
    exec(code, shared_env)
    _worker_metaheuristic = shared_env[algorithm_name]


class EarlyStoppingError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError(
        f"Timed out at line {frame.f_lineno} in {frame.f_code.co_filename}."
    )


def _run_one(fid, budget):
    """Run a single (fid) evaluation, return absolute error."""
    problem = get_gnbg(fid)
    algorithm = _worker_metaheuristic(dim=problem.Dimension)
    stopping_condition = lambda: problem.FE >= budget
    algorithm(problem.fitness, stopping_condition)

    if problem.FE < budget:
        raise EarlyStoppingError(
            "The algorithm stopped before stopping_condition() returned True"
        )

    best_value = min(problem.FEhistory[:budget])
    return max(best_value - problem.OptimumValue, 1e-8)


def run_packed_gnbg(args):
    """Run a PACK of (fid, rep) evaluations sequentially. Returns list of results."""
    fid, num_reps, budget, timeout_seconds = args

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        errors_list = []
        for _ in range(num_reps):
            errors_list.append(_run_one(fid, budget))

        return ("ok", fid, errors_list)

    except Exception as excp:
        tb_str = traceback.format_exc()
        print(f"Error in fid={fid}: {tb_str}")
        return ("error", fid, excp, tb_str)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def run_single_gnbg(args):
    """Run ONE (fid, rep) evaluation. For full-budget tasks."""
    fid, rep_index, budget, timeout_seconds = args

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        abs_error = _run_one(fid, budget)
        return ("ok", fid, [abs_error])

    except Exception as excp:
        tb_str = traceback.format_exc()
        print(f"Error in fid={fid} rep={rep_index}: {tb_str}")
        return ("error", fid, excp, tb_str)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        signal.alarm(0)


def get_first_non_enum_class(code: str) -> str:
    pattern = r"class\s+(\w+)(?:\s*\([^)]*\))?:"
    for match in re.finditer(pattern, code):
        class_name = match.group(1)
        full_match = match.group(0)
        if not re.search(r"\(\s*\w*Enum\w*\s*\)", full_match):
            return class_name
    return "Not-Identified"


def evaluateGNGB(code, explogger=None, details=True, iterations=1000000, repetitions_per_fid=6):
    algorithm_name = get_first_non_enum_class(code)
    half_iterations = iterations // 2

    # ── Build tasks ───────────────────────────────────────────────
    # Fids 1-15 (half budget, fast): 6 reps packed into 2 tasks of 3
    #   → 15 fids × 2 packs = 30 tasks
    # Fids 16-24 (full budget, slow): 6 individual tasks each
    #   → 9 fids × 6 = 54 tasks
    # Total: 84 tasks
    packed_tasks = []    # (func, args) pairs
    for fid in range(1, 16):
        for pack in range(repetitions_per_fid // HALF_BUDGET_PACK_SIZE):
            packed_tasks.append(
                (run_packed_gnbg, (fid, HALF_BUDGET_PACK_SIZE, half_iterations, 10 * 3600))
            )

    single_tasks = []
    for fid in range(16, 25):
        for rep in range(repetitions_per_fid):
            single_tasks.append(
                (run_single_gnbg, (fid, rep, iterations, 10 * 3600))
            )

    all_tasks = packed_tasks + single_tasks
    num_workers = mp.cpu_count()
    print(f"Dispatching {len(all_tasks)} tasks ({len(packed_tasks)} packed + "
          f"{len(single_tasks)} single) across {num_workers} workers...")

    # ── Execute ───────────────────────────────────────────────────
    results_dict: dict[int, list[float]] = {}
    errors = []

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_worker_init,
        initargs=(code, algorithm_name),
    ) as executor:
        futures = {
            executor.submit(func, args): args
            for func, args in all_tasks
        }
        for future in as_completed(futures):
            elem = future.result()
            if elem[0] == "error":
                errors.append(elem)
            else:
                _, fid, abs_errors = elem
                results_dict.setdefault(fid, []).extend(abs_errors)

    if errors:
        print(f"{len(errors)} task(s) failed. First error:")
        _, fid, excp, tb_str = errors[0]
        print(f"  fid={fid}: {tb_str}")
        return [-np.inf for _ in range(24)]

    # ── Score each fid ────────────────────────────────────────────
    results_list = [0.0] * 24
    for i in range(24):
        fid_errors = sorted(results_dict[i + 1])
        results_list[i] = harmonic_mean(fid_errors[1:5])

    worst_index = int(np.argmax(np.array(results_list)))
    worst_absolute_error = results_list[worst_index]
    absolute_error_mean = geometric_mean(results_list)

    print(
        f"Algorithm {algorithm_name}: avg error {absolute_error_mean:.2e}, "
        f"worst {worst_absolute_error:.2e} at f{worst_index + 1} (target: 1e-8)"
    )

    return results_list


if __name__ == "__main__":
    with open("GNBG_Runners/example_algorithm.py", "r") as f:
        code = f.read()
    budgets = [10000]
    with open("time_comparison.csv", "+w") as file:
        file.write("budget,time\n")
        for budget in budgets:
            start = time.perf_counter()
            print("Start time", str(datetime.datetime.now()))
            print(evaluateGNGB(code, iterations=budget, repetitions_per_fid=REPETITIONS_PER_FID))
            end = time.perf_counter()
            elapsed = end - start
            print(f"Elapsed time: {elapsed:.6f} seconds for {budget} iterations")
            file.write(f"{budget},{elapsed}\n")