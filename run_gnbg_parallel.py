# This is a minimal example without any other dependencies than LLaMEA and the Gemini LLM.

from concurrent.futures import ProcessPoolExecutor
import datetime
import os
import re
import signal
from statistics import geometric_mean, harmonic_mean
import sys
import time
import traceback

import numpy as np
from typing import List, Tuple, Any, Callable
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
import traceback
from GNBG_Runners.GNBG_II.GNBG_instances import get_gnbg

thread_repetitions=2
extra_repetitions=3

class EarlyStoppingError(Exception):
    pass

def timeout_handler(signum, frame):
    # frame contains info about where we were
    raise TimeoutError(f"Timed out at line {frame.f_lineno} in {frame.f_code.co_filename}. Exceeded 3h timeout.")

def run_experiment_gnbg_II(args):
    thread_num, code, algorithm_name, budget, timeout_seconds = args
    absolute_errors = []
    shared_env = {"__builtins__": __builtins__}
    old_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    result = None
    try:
        exec(code, shared_env)
        metaheuristic = shared_env[algorithm_name]
        if thread_num < 16:
            fid = thread_num
            repetitions = thread_repetitions * extra_repetitions
        else:
            fid = thread_num
            while fid > 24:
                fid -= 9
            repetitions = extra_repetitions
        
        
        for _ in range(repetitions):
            problem = get_gnbg(fid)
            algorithm = metaheuristic(dim=problem.Dimension)
            stopping_condition = lambda: problem.FE >= budget
            algorithm(problem.fitness, stopping_condition)
            
            if problem.FE < budget:
                raise EarlyStoppingError("The algorithm stopped before stopping_condition() returned True")
            
            best_value = min(problem.FEhistory[:budget])
            
            difference = max(best_value - problem.OptimumValue, 10**-8)
            absolute_errors.append(difference)
        
        print("Finished:", thread_num)
        result = (fid, absolute_errors)
         
    except Exception as excp:
        # Capture full traceback as string while still in the process
        tb_str = traceback.format_exc()
        print(f"Error in fid={fid}: {tb_str}")
        result = (excp, tb_str, fid)  # Return more context
    finally:
        # Cancel alarm FIRST, then restore handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    
    # Double-check alarm is cancelled before returning
    signal.alarm(0)
    
    return result

def get_first_non_enum_class(code: str) -> str:
    """Get the name of the first class that doesn't inherit from Enum."""
    pattern = r'class\s+(\w+)(?:\s*\([^)]*\))?:'
    
    for match in re.finditer(pattern, code):
        class_name = match.group(1)
        full_match = match.group(0)
        
        # Check if it inherits from Enum
        if not re.search(r'\(\s*\w*Enum\w*\s*\)', full_match):
            return class_name
    
    return "Not-Identified"


def evaluateGNGB(code, explogger=None, details=True, iterations=1000000):
      
    algorithm_name = get_first_non_enum_class(code)

    results_list = 24*[0]
    errors=[]
    # analyze_code_and_install_deps(code)

    fids = np.arange(1, 15+9*thread_repetitions+1)
    half_iterations=iterations//2
    args_list = [(f, code, algorithm_name, half_iterations if f<16 else iterations, 10*3600) for f in fids]

    with ProcessPoolExecutor() as executor:
        # Add timeout to prevent hanging
        results = list(executor.map(run_experiment_gnbg_II, args_list))
    
    results_dict={}
    for elem in results:
        # Check if it's an error tuple
        if isinstance(elem, tuple) and len(elem) == 3 and isinstance(elem[0], Exception):
            errors.append(elem)
            continue
        
        fid, aucs_fun = elem
        if fid in results_dict:
            results_dict[fid] += aucs_fun
        else:
            results_dict[fid] = aucs_fun
    
    # IMPROVED ERROR HANDLING
    if len(errors) > 0:
        # excp, tb_str, fid = errors[0]
        
        # # Clean up traceback but preserve structure
        # # Remove the exec wrapper parts but keep useful info
        # tb_lines = tb_str.split('\n')
        # cleaned_lines = []
        # in_user_code = False
        
        # for line in tb_lines:
        #     if 'File "<string>"' in line:
        #         in_user_code = True
        #         # Extract just the relevant part
        #         cleaned_lines.append(line.replace('File "<string>"', 'File "algorithm.py"'))
        #     elif in_user_code or line.strip().startswith(('ValueError', 'TypeError', 'IndexError', 'KeyError', 'AttributeError', 'RuntimeError')):
        #         cleaned_lines.append(line)
        
        # cleaned_tb = '\n'.join(cleaned_lines) if cleaned_lines else tb_str
        
        # # Extract structured error info
        # error_info = extract_error_context(code, cleaned_tb, excp)
        
        # # Set scores with BOTH feedback and structured error
        # feedback = f"The algorithm {algorithm_name} failed with a {error_info['error_type']}."
        
        # # Store error info in solution for construct_prompt to use
        # solution.set_scores(-np.inf, feedback, error=excp, error_info=error_info)
        # solution.error_info = error_info
        
        return [-np.inf for _ in range(24)]
    
    for i in range(24):
        results_of_fid=results_dict[i+1]
        results_of_fid.sort()
        
        results_list[i] = harmonic_mean(results_of_fid[1:5])

    worst_index = np.argmax(np.array(results_list))
    worst_absolute_error = results_list[worst_index]
    absolute_error_mean = geometric_mean(results_list)
    
    feedback = f"The algorithm {algorithm_name} achieved an average absolute error of {absolute_error_mean:.2e}. Worst absolute error {worst_absolute_error:.2e} appeared at function {worst_index + 1}, where the targewt is target: 1e-8."

    return results_list

if __name__ == "__main__":
    with open("GNBG_Runners/example_algorithm.py", "r") as f:
        code = f.read()
    budgets=[1000000]
    with open("time_comparison.csv",'+w') as file:
        file.write(f"budget,time\n")
        for budget in budgets:
            start = time.perf_counter()
            print("Start time",str(datetime.datetime.now()))
            print(evaluateGNGB(code,iterations=budget))
            end = time.perf_counter()
            print(f"Elapsed time: {end - start:.6f} seconds for {budget} iterations")
            file.write(f"{budget},{end - start}\n")
