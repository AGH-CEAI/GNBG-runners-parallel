# Modified evaluator with recording support

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
HALF_BUDGET_PACK_SIZE = 3

_worker_metaheuristic = None
_worker_algorithm_name = None


def _worker_init(code, algorithm_name):
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


def _run_one(fid, budget, save_path=None):
    """Run a single (fid) evaluation. Optionally save recorder data."""
    problem = get_gnbg(fid, save=save_path is not None)
    algorithm = _worker_metaheuristic(dim=problem.Dimension)
    stopping_condition = lambda: problem.FE >= budget
    algorithm(problem.fitness, stopping_condition)

    if problem.FE < budget:
        raise EarlyStoppingError(
            "The algorithm stopped before stopping_condition() returned True"
        )

    fe_count = min(problem.FE, budget)
    best_value = min(problem.FEhistory[:fe_count])
    abs_error = max(best_value - problem.OptimumValue, 1e-8)

    if save_path is not None:
        np.savez_compressed(
            save_path,
            FEhistory=problem.FEhistory[:fe_count],
            FEsamples=problem.FEsamples[:fe_count],
            FE=fe_count,
        )

    return abs_error


def run_packed_gnbg(args):
    fid, rep_indices, budget, timeout_seconds, save_dir = args

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        errors_list = []
        for rep_idx in rep_indices:
            save_path = os.path.join(save_dir, f"f{fid}_rep{rep_idx}") if save_dir else None
            errors_list.append(_run_one(fid, budget, save_path))
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
    fid, rep_index, budget, timeout_seconds, save_dir = args

    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        save_path = os.path.join(save_dir, f"f{fid}_rep{rep_index}") if save_dir else None
        abs_error = _run_one(fid, budget, save_path)
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


def evaluateGNGB(code, explogger=None, details=True, iterations=1000000,
                 repetitions_per_fid=6, save_dir=None):
    algorithm_name = get_first_non_enum_class(code)
    half_iterations = iterations // 2

    # Create save directory if recording
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # ── Build tasks ───────────────────────────────────────────────
    packed_tasks = []
    for fid in range(1, 16):
        for pack_idx in range(repetitions_per_fid // HALF_BUDGET_PACK_SIZE):
            start_rep = pack_idx * HALF_BUDGET_PACK_SIZE
            rep_indices = list(range(start_rep, start_rep + HALF_BUDGET_PACK_SIZE))
            packed_tasks.append(
                (run_packed_gnbg, (fid, rep_indices, half_iterations, 10 * 3600, save_dir))
            )

    single_tasks = []
    for fid in range(16, 25):
        for rep in range(repetitions_per_fid):
            single_tasks.append(
                (run_single_gnbg, (fid, rep, iterations, 10 * 3600, save_dir))
            )

    all_tasks = packed_tasks + single_tasks
    num_workers = mp.cpu_count()
    print(f"Dispatching {len(all_tasks)} tasks across {num_workers} workers...")

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