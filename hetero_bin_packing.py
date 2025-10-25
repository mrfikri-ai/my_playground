import math
import time
from functools import lru_cache
from typing import Sequence, Union, List, Optional, Set, Tuple
from datetime import datetime
import pandas as pd
import ast
import os
import numbers
import itertools

# Keep only LPT-f and LPT-g. All other algorithms removed as requested.

# Controls whether to append a separate 'BruteForce' row per case to results.
APPEND_BRUTEFORCE_RESULT_ROW = True
# Number of decimal places for rounding in output formatting
ROUND_DIGITS = 4
LOOKAHEAD_EPS = 1e-12
LOOKAHEAD_CACHE_DIGITS = 12
LOOKAHEAD_DEPTHS = [0, 1, 2, 3, 4, 5]

# ---------------------------
# Objective functions
# ---------------------------
def objective(
    output: Sequence[Union[int, float, Sequence[Union[int, float]]]],
    capacities: Union[float, Sequence[Union[int, float]]]
) -> float:
    """Compute f(x)=sum_j (cap_j - load_j)^2; capacities can be scalar or list."""
    if output and all(isinstance(x, numbers.Number) for x in output):
        loads = [float(x) for x in output]  # type: ignore
    elif output and all(isinstance(bin_, (list, tuple)) and all(isinstance(v, numbers.Number) for v in bin_) for bin_ in output):
        loads = []
        for bin_ in output:
            loads.append(sum(float(v) for v in bin_))  # type: ignore
    else:
        raise ValueError("'output' must be either a flat sequence of numbers or a sequence of numeric sequences.")

    if isinstance(capacities, numbers.Number):
        caps = [float(capacities)] * len(loads)  # type: ignore
    elif isinstance(capacities, (list, tuple)):
        caps = [float(c) for c in capacities]  # type: ignore
        if len(caps) != len(loads):
            raise ValueError(f"capacities length {len(caps)} != loads length {len(loads)}")
    else:
        raise ValueError(f"Invalid capacities type: {type(capacities)}")
    return sum((cap - l) ** 2 for cap, l in zip(caps, loads))

def g_objective(
    output: Sequence[Union[int, float, Sequence[Union[int, float]]]],
    capacities: Sequence[Union[int, float]]
) -> float:
    """Compute g(x)=sum_{j<k} (normalized_k - normalized_j)^2."""
    if output and all(isinstance(val, numbers.Number) for val in output):
        loads = [float(val) for val in output]  # type: ignore
    elif output and all(isinstance(bin_, (list, tuple)) and all(isinstance(v, numbers.Number) for v in bin_) for bin_ in output):
        loads = []
        for bin_ in output:
            loads.append(sum(float(v) for v in bin_))  # type: ignore
    else:
        raise ValueError("'output' must be either a flat sequence of numbers or a sequence of numeric sequences.")
    n = len(loads)
    if len(capacities) != n:
        raise ValueError(f"'capacities' must have length {n}, got {len(capacities)}")
    for cap in capacities:
        if float(cap) == 0.0:
            raise ValueError("capacities must be non-zero for g_objective")
    normalized = [L / float(cap) for L, cap in zip(loads, capacities)]
    g = 0.0
    for j in range(n):
        for k in range(j+1, n):
            diff = normalized[k] - normalized[j]
            g += diff * diff
    return g


def _simulate_lpt_f_loads(loads: List[float], caps: List[float], remaining_items: Sequence[float]) -> List[float]:
    """Simulate the remainder of the standard LPT-f heuristic from the current state."""
    sim_loads = loads[:]
    m = len(caps)
    for item in remaining_items:
        empty_idxs = [idx for idx in range(m) if sim_loads[idx] == 0.0]
        if empty_idxs:
            j_sel = max(empty_idxs, key=lambda j: (caps[j], -j))
        else:
            j_sel = max(range(m), key=lambda j: (caps[j] - sim_loads[j], -j))
        sim_loads[j_sel] += item
    return sim_loads


def _simulate_lpt_g_loads(loads: List[float], caps: List[float], remaining_items: Sequence[float]) -> List[float]:
    """Simulate the remainder of the standard LPT-g heuristic from the current state."""
    sim_loads = loads[:]
    k = len(caps)
    for item in remaining_items:
        j_sel = min(range(k), key=lambda j: (sim_loads[j] / caps[j], j))
        sim_loads[j_sel] += item
    return sim_loads


def _loads_to_cache_key(loads: Sequence[float]) -> Tuple[float, ...]:
    """Round loads so they can be memoized reliably in lookahead recursion."""
    return tuple(round(val, LOOKAHEAD_CACHE_DIGITS) for val in loads)


# ---------------------------
# LPT-f and LPT-g only
# ---------------------------
def lpt_f_partition(items: List[Union[int, float]],
                    capacities: Sequence[Union[int, float]],
                    alpha: float = 2.0,
                    delta: float = 1e-5,
                    lookahead_depth: Union[int, bool] = 0) -> List[List[Union[int, float]]]:
    """LPT-f (two-phase) for f(x) = Σ_j (c_j - S_j)^2.

    ``lookahead_depth`` controls how many future placements are explored with full
    branching before falling back to the baseline heuristic simulation. ``0``
    reproduces the classic algorithm, ``1`` matches the previous one-step
    lookahead, and larger depths expand the search tree accordingly.
    """
    if not capacities:
        raise ValueError("capacities must be non-empty")
    caps = [float(c) for c in capacities]
    if any(c <= 0 for c in caps):
        raise ValueError("capacities must be strictly positive")

    xs = [float(x) for x in items]
    if any(x <= 0 for x in xs):
        raise ValueError("all item sizes must be > 0")

    m = len(caps)
    bins: List[List[Union[int, float]]] = [[] for _ in range(m)]
    loads: List[float] = [0.0] * m

    xs_sorted = sorted(xs, reverse=True)
    n = len(xs_sorted)

    depth_int = int(lookahead_depth)
    if depth_int < 0:
        raise ValueError("lookahead_depth must be non-negative")

    @lru_cache(maxsize=None)
    def _project(loads_key: Tuple[float, ...], remaining_items: Tuple[float, ...], depth: int) -> float:
        if not remaining_items:
            return objective(loads_key, capacities=caps)
        if depth <= 0:
            future_loads = _simulate_lpt_f_loads(list(loads_key), caps, remaining_items)
            return objective(future_loads, capacities=caps)
        next_item = remaining_items[0]
        rest = remaining_items[1:]
        loads_list = list(loads_key)
        best_score = math.inf
        seen_states: Set[Tuple[float, ...]] = set()
        for j in range(m):
            trial_loads = loads_list[:]
            trial_loads[j] += next_item
            trial_key = _loads_to_cache_key(trial_loads)
            if trial_key in seen_states:
                continue
            seen_states.add(trial_key)
            score = _project(trial_key, rest, depth - 1)
            if score < best_score - LOOKAHEAD_EPS:
                best_score = score
        return best_score

    for idx, x in enumerate(xs_sorted):
        remaining = xs_sorted[idx + 1:]
        empty_idxs = [j for j in range(m) if loads[j] == 0.0]

        if depth_int > 0 and remaining:
            best_j = None
            best_score = math.inf
            best_priority: Optional[Tuple[float, float]] = None
            for j in range(m):
                if loads[j] == 0.0:
                    priority = (2.0, caps[j], -j)
                else:
                    priority = (1.0, caps[j] - loads[j], -j)
                trial_loads = loads[:]
                trial_loads[j] += x
                score = _project(_loads_to_cache_key(trial_loads), tuple(remaining), depth_int - 1)
                if best_j is None or score < best_score - LOOKAHEAD_EPS:
                    best_j = j
                    best_score = score
                    best_priority = priority
                elif abs(score - best_score) <= LOOKAHEAD_EPS:
                    if best_priority is None or priority > best_priority:
                        best_j = j
                        best_score = score
                        best_priority = priority
            if best_j is None:
                raise RuntimeError("Failed to select a bin during LPT-f lookahead phase")
            j_star = best_j
        else:
            candidate_bins = empty_idxs if empty_idxs else list(range(m))
            if empty_idxs:
                j_star = max(empty_idxs, key=lambda j: (caps[j], -j))
            else:
                j_star = max(range(m), key=lambda j: (caps[j] - loads[j], -j))

        bins[j_star].append(x)
        loads[j_star] += x

    return bins

def lpt_g_partition(items: List[Union[int, float]],
                          capacities: Sequence[Union[int, float]],
                          lookahead_depth: Union[int, bool] = 0) -> List[List[Union[int, float]]]:
    """LPT-g (ratio rule): for each item in descending order, assign to the bin
    with the smallest normalized load ℓ_j / c_j (ties → smallest index).

    ``lookahead_depth`` mirrors :func:`lpt_f_partition`, expanding how many future
    placements are explored via branching before reverting to the baseline ratio
    simulation. ``0`` is the classic heuristic; ``1`` is the former one-step
    lookahead; larger values extend the preview horizon.
    """
    if not capacities:
        raise ValueError("capacities must be non-empty")
    caps = [float(c) for c in capacities]
    if any(c <= 0.0 for c in caps):
        raise ValueError("capacities must be strictly positive")
    xs = [float(x) for x in items]
    if any(x <= 0.0 for x in xs):
        raise ValueError("all item sizes must be > 0")

    k = len(caps)
    bins: List[List[Union[int, float]]] = [[] for _ in range(k)]
    loads: List[float] = [0.0] * k

    depth_int = int(lookahead_depth)
    if depth_int < 0:
        raise ValueError("lookahead_depth must be non-negative")

    @lru_cache(maxsize=None)
    def _project(loads_key: Tuple[float, ...], remaining_items: Tuple[float, ...], depth: int) -> float:
        if not remaining_items:
            return g_objective(loads_key, capacities=caps)
        if depth <= 0:
            future_loads = _simulate_lpt_g_loads(list(loads_key), caps, remaining_items)
            return g_objective(future_loads, capacities=caps)
        next_item = remaining_items[0]
        rest = remaining_items[1:]
        best_score = math.inf
        for j in range(k):
            trial_loads = list(loads_key)
            trial_loads[j] += next_item
            score = _project(_loads_to_cache_key(trial_loads), rest, depth - 1)
            if score < best_score - LOOKAHEAD_EPS:
                best_score = score
        return best_score

    xs_sorted = sorted(xs, reverse=True)
    for idx, s in enumerate(xs_sorted):
        remaining = xs_sorted[idx + 1:]
        if depth_int > 0 and remaining:
            best_j = None
            best_score = math.inf
            best_priority: Optional[Tuple[float, float]] = None
            for j in range(k):
                priority = (loads[j] / caps[j], j)
                trial_loads = loads[:]
                trial_loads[j] += s
                score = _project(_loads_to_cache_key(trial_loads), tuple(remaining), depth_int - 1)
                if best_j is None or score < best_score - LOOKAHEAD_EPS:
                    best_j = j
                    best_score = score
                    best_priority = priority
                elif abs(score - best_score) <= LOOKAHEAD_EPS:
                    if best_priority is None or priority < best_priority:
                        best_j = j
                        best_score = score
                        best_priority = priority
            if best_j is None:
                raise RuntimeError("Failed to select a bin during LPT-g lookahead phase")
            j_sel = best_j
        else:
            j_sel = min(range(k), key=lambda j: (loads[j] / caps[j], j))

        bins[j_sel].append(s)
        loads[j_sel] += s

    return bins


# ---------------------------
# Brute-force enumeration (RGS) and ranking
# ---------------------------
def _best_perm_metrics_for_unlabeled_loads(
    loads: List[float], capacities: Sequence[Union[int, float]]
) -> Tuple[float, float, Tuple[int, ...], Tuple[int, ...]]:
    """Given unlabeled block loads and labeled capacities, find best labeling for f and g."""
    caps = [float(c) for c in capacities]
    k = len(caps)
    best_f = math.inf
    best_g = math.inf
    best_perm_f = tuple(range(k))
    best_perm_g = tuple(range(k))
    for perm in itertools.permutations(range(k)):
        perm_loads = [loads[perm[j]] for j in range(k)]
        f_val = objective(perm_loads, capacities=caps)
        g_val = g_objective(perm_loads, capacities=caps)
        if f_val < best_f - 1e-12:
            best_f = f_val
            best_perm_f = perm
        if g_val < best_g - 1e-12:
            best_g = g_val
            best_perm_g = perm
    return best_f, best_g, best_perm_f, best_perm_g

def brute_force_and_rank_solutions(
    items: Sequence[Union[int, float]],
    k: int,
    algos: dict,
    capacities: Union[float, Sequence[Union[int, float]]],
    ideal_loads_f: Optional[List[float]] = None,
    ideal_loads_g: Optional[List[float]] = None
) -> Tuple[List[Tuple], List[Tuple], List[Tuple], float, float, float, List[dict]]:
    """Enumerate all unique partitions (RGS with exactly k blocks) and rank by makespan, f, g."""
    seen_rgs: Set[Tuple[int, ...]] = set()
    solution_metrics: List[Tuple] = []
    n = len(items)
    print("Starting brute-force partition enumeration via RGS (full enumeration of all partitions)...")
    start_time = time.perf_counter()

    def generate_rgs(n, k):
        a = [0] * n
        def backtrack(i, m):
            if i == n:
                if m == k - 1:
                    yield a.copy()
                return
            for v in range(m + 2):
                if v < k:
                    a[i] = v
                    yield from backtrack(i + 1, max(m, v))
        yield from backtrack(1, 0)

    for rgs in generate_rgs(n, k):
        rgs_key = tuple(rgs)
        if rgs_key in seen_rgs:
            continue
        seen_rgs.add(rgs_key)
        loads = [0.0] * k
        for idx_item, bin_id in enumerate(rgs_key):
            loads[bin_id] += items[idx_item]
        makespan = max(loads)
        if isinstance(capacities, (int, float)):
            cap_val = float(capacities)  # type: ignore
            caps_list = [cap_val] * k
        elif isinstance(capacities, (list, tuple)):
            caps_list = [float(c) for c in capacities]
        else:
            raise ValueError(f"Invalid capacities type: {type(capacities)}")
        best_f_val, best_g_val, best_perm_f, best_perm_g = _best_perm_metrics_for_unlabeled_loads(loads, caps_list)
        solution_metrics.append((rgs_key, makespan, best_f_val, best_g_val, best_perm_f, best_perm_g))

    end_time = time.perf_counter()
    brute_force_execution_time = end_time - start_time
    unique_rgs_count = len(seen_rgs)
    print(f"Brute-force partition enumeration finished in {brute_force_execution_time:.4f} seconds.")
    print(f"Unique partitions S({n},{k}): {unique_rgs_count}")
    print()

    if not solution_metrics:
        print("No solutions found.")
        return ([], [], [], 0.0, 0.0, 0.0, [])

    ranked_by_makespan = sorted(solution_metrics, key=lambda s: (round(s[1], 12), s[0]))
    ranked_by_f        = sorted(solution_metrics, key=lambda s: (round(s[2], 12), s[0]))
    ranked_by_g        = sorted(solution_metrics, key=lambda s: (round(s[3], 12), s[0]))

    opt_f = ranked_by_f[0][2]
    opt_g = ranked_by_g[0][3]

    # Allow external ideals to adjust baselines (not rankings)
    eps = 1e-10
    def _caps_list(cap_like, n_bins):
        if isinstance(cap_like, (int, float)):
            return [float(cap_like)] * n_bins
        elif isinstance(cap_like, (list, tuple)):
            return [float(c) for c in cap_like]
        else:
            raise ValueError(f"Invalid capacities type: {type(cap_like)}")
    try:
        if ideal_loads_f is not None:
            ideal_f_val = objective(ideal_loads_f, capacities=capacities)
            if ideal_f_val + eps < opt_f:
                opt_f = ideal_f_val
    except Exception:
        pass
    try:
        if ideal_loads_g is not None:
            caps_list = _caps_list(capacities, len(ideal_loads_g))
            ideal_g_val = g_objective(ideal_loads_g, capacities=caps_list)
            if ideal_g_val + eps < opt_g:
                opt_g = ideal_g_val
    except Exception:
        pass

    print("Optimal Solution by f(x):", ranked_by_f[0][0], "     | f(x):", ranked_by_f[0][2])
    print("Optimal Solution by g(x):", ranked_by_g[0][0], "     | g(x):", f"{ranked_by_g[0][3]:.8f}")
    print(f"Brute-force Execution Time: {brute_force_execution_time:.6f}s")
    print("-" * 60)

    # Evaluate the two heuristics only
    num_solutions = len(solution_metrics)
    def find_rank(value, sorted_list, metric_index, epsilon=1e-10):
        for i, solution in enumerate(sorted_list):
            if value <= solution[metric_index] + epsilon:
                return min(i + 1, num_solutions)
        return num_solutions

    results = []
    for name, func in algos.items():
        print(f"Evaluating: {name}")
        start = time.perf_counter()
        partition = func(items)
        end = time.perf_counter()
        elapsed = end - start

        raw_loads = [sum(b) for b in partition]
        h_makespan = max(raw_loads)
        h_f_val = objective(raw_loads, capacities=capacities)
        caps_list = [float(c) for c in (capacities if isinstance(capacities, (list, tuple)) else [capacities]*len(raw_loads))]
        h_g_val = g_objective(raw_loads, capacities=caps_list)

        makespan_rank = find_rank(h_makespan, ranked_by_makespan, 1)
        f_rank = find_rank(h_f_val, ranked_by_f, 2)
        g_rank = find_rank(h_g_val, ranked_by_g, 3)

        print(f"  Assignment: {partition}")
        print(f"  Loads: {raw_loads}")
        print(f"  f(x): {h_f_val} (Rank: {f_rank}/{num_solutions})")
        print(f"  g(x): {h_g_val:.8f} (Rank: {g_rank}/{num_solutions})")
        print(f"  Execution Time: {elapsed:.6f}s")
        print("-" * 60)

        results.append({
            "CaseID": case_id,
            "Iteration": iteration,
            "Cluster_Size": str(capacities),
            "Sensing_Range": str(items),
            "Metric_1_Alpha_1": row.get('Metric_1_Alpha_1'),
            "Metric_2_Alpha_1": row.get('Metric_2_Alpha_1'),
            "Algorithm": name,
            "f(x)": h_f_val,
            "g(x)": h_g_val,
            "Execution_Time": f"{elapsed:.15f}".replace('.', ','),
            "Assignment_Matrix": str(partition),
            "Gap_f(%)": round((h_f_val - ranked_by_f[0][2]) / ranked_by_f[0][2] * 100, 2) if ranked_by_f[0][2] else None,
            "Gap_g(%)": round((h_g_val - ranked_by_g[0][3]) / ranked_by_g[0][3] * 100, 2) if ranked_by_g[0][3] else None,
            "Rank_Makespan": makespan_rank,
            "Rank_f": f_rank,
            "Rank_f_full": f"{f_rank}/{num_solutions}",
            "Rank_g": g_rank,
            "Rank_g_full": f"{g_rank}/{num_solutions}"
        })

    # Append brute-force labeled optima row(s)
    best_rgs    = ranked_by_f[0][0]
    best_perm_f = ranked_by_f[0][4]
    bf_bins = [[] for _ in range(k)]
    for item_idx, block_id in enumerate(best_rgs):
        bin_j = best_perm_f.index(block_id)
        bf_bins[bin_j].append(items[item_idx])

    best_rgs_g    = ranked_by_g[0][0]
    best_perm_g   = ranked_by_g[0][5]
    bg_bins = [[] for _ in range(k)]
    for item_idx, block_id in enumerate(best_rgs_g):
        bin_j = best_perm_g.index(block_id)
        bg_bins[bin_j].append(items[item_idx])

    if APPEND_BRUTEFORCE_RESULT_ROW:
        results.append({
            "CaseID": case_id,
            "Iteration": iteration,
            "Cluster_Size": str(capacities),
            "Sensing_Range": str(items),
            "Metric_1_Alpha_1": row.get('Metric_1_Alpha_1'),
            "Metric_2_Alpha_1": row.get('Metric_2_Alpha_1'),
            "Algorithm": "BruteForce",
            "f(x)": ranked_by_f[0][2],
            "g(x)": ranked_by_g[0][3],
            "Execution_Time": f"{brute_force_execution_time:.15f}".replace('.', ','),
            "Assignment_Matrix": str(bf_bins),
            "Assignment_Matrix_g": str(bg_bins)
        })

    return ranked_by_makespan, ranked_by_f, ranked_by_g, opt_f, opt_g, brute_force_execution_time, results

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    # Define file paths
    input_file = r"nonuniform_sorted_simulation_results_.xlsx"
    output_dir = os.path.dirname(input_file)
    timestamp = datetime.now().strftime("%d-%m-%y-%H%M%S")
    output_file = os.path.join(output_dir, f"hetero_bin_packing_results_{timestamp}.xlsx")
    df = pd.read_excel(input_file)

    # Optional: set a single case or range; set to None to process all
    INSPECT_CASE: Optional[Union[int, Tuple[int, int]]] = (10, 20)

    if isinstance(INSPECT_CASE, (list, tuple)):
        if len(INSPECT_CASE) != 2 or not all(isinstance(i, (int, float)) for i in INSPECT_CASE):
            raise ValueError(f"INSPECT_CASE range must be a tuple/list of two ints, got {INSPECT_CASE}")
        start_case, end_case = sorted([int(INSPECT_CASE[0]), int(INSPECT_CASE[1])])
        if 'CaseID' in df.columns:
            df_case_ids = [int(r) for r in df['CaseID'] if pd.notna(r) and isinstance(r, (int, float))]
        else:
            df_case_ids = list(range(1, len(df) + 1))
        if df_case_ids:
            min_avail, max_avail = min(df_case_ids), max(df_case_ids)
            if start_case < min_avail or end_case > max_avail:
                print(f"Warning: Specified INSPECT_CASE range {start_case}-{end_case} outside available CaseID range {min_avail}-{max_avail}.")

    all_results: List[dict] = []
    fg_checks_rows: List[dict] = []
    processed_cases: List[int] = []

    # Only the two requested heuristics:
    # NOTE: lambdas close over per-row variables (k, capacities) defined inside the loop.
    #       Do not move this dict outside the per-row loop unless you refactor arguments.
    for idx, (_, row) in enumerate(df.iterrows()):
        case_id = row.get('CaseID', idx + 1)
        if not isinstance(case_id, (int, float)):
            case_id = idx + 1
        iteration = int(row['Iteration']) if 'Iteration' in row and pd.notna(row['Iteration']) else idx

        if INSPECT_CASE is not None:
            if isinstance(INSPECT_CASE, (list, tuple)):
                if not (start_case <= case_id <= end_case):
                    continue
            else:
                if case_id != INSPECT_CASE:
                    continue

        sensing_range = ast.literal_eval(row['Sensing_Range'])
        cluster_size  = ast.literal_eval(row['Cluster_Size'])
        k             = len(cluster_size)
        capacities    = cluster_size
        total_sensing = sum(sensing_range)

        # Optional “ideal” loads from Excel (if present)
        ideal_loads_f = None
        ideal_matrix_f = None
        if 'Assignment_Matrix_Alpha_1' in row and pd.notna(row['Assignment_Matrix_Alpha_1']):
            try:
                mat_f = ast.literal_eval(row['Assignment_Matrix_Alpha_1'])
                if isinstance(mat_f, (list, tuple)) and len(mat_f) == k:
                    loads_f = [sum(bin_) for bin_ in mat_f]
                    if abs(sum(loads_f) - total_sensing) < 1e-6:
                        ideal_loads_f = [float(x) for x in loads_f]
                        ideal_matrix_f = mat_f
            except Exception:
                pass

        ideal_loads_g = None
        ideal_matrix_g = None
        if 'Assignment_Matrix_Alpha_0' in row and pd.notna(row['Assignment_Matrix_Alpha_0']):
            try:
                mat_g = ast.literal_eval(row['Assignment_Matrix_Alpha_0'])
                if isinstance(mat_g, (list, tuple)) and len(mat_g) == k:
                    loads_g = [sum(bin_) for bin_ in mat_g]
                    if abs(sum(loads_g) - total_sensing) < 1e-6:
                        ideal_loads_g = [float(x) for x in loads_g]
                        ideal_matrix_g = mat_g
            except Exception:
                pass

        print(f"\n=== Processing CaseID {case_id} ===\n")

        heuristic_algos = {}
        for depth in LOOKAHEAD_DEPTHS:
            f_name = "LPT_f" if depth == 0 else f"LPT_f_Lookahead_d{depth}"
            g_name = "LPT_g" if depth == 0 else f"LPT_g_Lookahead_d{depth}"
            heuristic_algos[f_name] = (
                lambda items, d=depth: lpt_f_partition(items, capacities, alpha=20.0, delta=1e-5, lookahead_depth=d)
            )
            heuristic_algos[g_name] = (
                lambda items, d=depth: lpt_g_partition(items, capacities, lookahead_depth=d)
            )

        ranked_by_makespan, ranked_by_f, ranked_by_g, opt_f, opt_g, brute_force_time, case_results = (
            brute_force_and_rank_solutions(
                sensing_range,
                k,
                heuristic_algos,
                capacities,
                ideal_loads_f=ideal_loads_f,
                ideal_loads_g=ideal_loads_g
            )
        )

        # If an “ideal” beats brute-force baseline after rounding tolerance, fix up reporting
        for result in case_results:
            algo_f = result['f(x)']
            algo_g = result['g(x)']
            if ideal_loads_f and ideal_matrix_f and round(algo_f, ROUND_DIGITS) < round(opt_f, ROUND_DIGITS):
                result['Assignment_Matrix'] = str(ideal_matrix_f)
                result['Gap_f(%)'] = 0.0
                result['Rank_f'] = 1
            if ideal_loads_g and ideal_matrix_g and round(algo_g, ROUND_DIGITS) < round(opt_g, ROUND_DIGITS):
                result['Assignment_Matrix'] = str(ideal_matrix_g)
                result['Gap_g(%)'] = 0.0
                result['Rank_g'] = 1

        all_results.extend(case_results)

        # Build f/g check rows comparing corresponding LPT-f and LPT-g depths
        alg_map = {r['Algorithm']: r for r in case_results if 'Algorithm' in r}
        for depth in LOOKAHEAD_DEPTHS:
            lhs = "LPT_f" if depth == 0 else f"LPT_f_Lookahead_d{depth}"
            rhs = "LPT_g" if depth == 0 else f"LPT_g_Lookahead_d{depth}"
            if lhs in alg_map and rhs in alg_map:
                f_lhs = float(alg_map[lhs]['f(x)'])
                f_rhs = float(alg_map[rhs]['f(x)'])
                g_lhs = float(alg_map[lhs]['g(x)'])
                g_rhs = float(alg_map[rhs]['g(x)'])
                fg_checks_rows.append({
                    'CaseID': case_id,
                    'Iteration': iteration,
                    'Pair': f"{lhs}_vs_{rhs}",
                    'Cluster_Size': str(capacities),
                    'Sensing_Range': str(sensing_range),
                    'f(x_f*)': f_lhs,
                    'f(x_g*)': f_rhs,
                    'g(x_f*)': g_lhs,
                    'g(x_g*)': g_rhs,
                    'f_values_different': abs(f_lhs - f_rhs) > 1e-10,
                    'g_values_different': abs(g_lhs - g_rhs) > 1e-10,
                    'Assignment_Matrix_f': alg_map[lhs]['Assignment_Matrix'],
                    'Assignment_Matrix_g': alg_map[rhs]['Assignment_Matrix'],
                })

        processed_cases.append(int(case_id))

    print(f"Processed CaseIDs: {processed_cases}")

    # ---------------------------
    # Output (Results + Averages)
    # ---------------------------
    results_df = pd.DataFrame(all_results)

    # Pivot to wide format
    wide_df = results_df.set_index(
        ['CaseID','Cluster_Size','Sensing_Range','Iteration']
    ).pivot(columns='Algorithm')
    wide_df.columns = [f"{alg}_{metric}" for (metric, alg) in wide_df.columns]
    wide_df.reset_index(inplace=True)

    # Compact renaming
    algorithm_codes = {
        'LPT_f': 'LPTf',
        'LPT_g': 'LPTg',
        'BruteForce': 'BF'
    }
    for depth in LOOKAHEAD_DEPTHS[1:]:
        algorithm_codes[f'LPT_f_Lookahead_d{depth}'] = f'LPTfLA{depth}'
        algorithm_codes[f'LPT_g_Lookahead_d{depth}'] = f'LPTgLA{depth}'
    metric_suffix_map = {
        'Assignment_Matrix': 'A',
        'Assignment_Matrix_g': 'Ag',
        'Execution_Time': 't',
        'f(x)': 'f',
        'g(x)': 'g',
        'Gap_f(%)': 'gapf',
        'Gap_g(%)': 'gapg',
        'Rank_f': 'rf',
        'Rank_f_full': 'rf_full',
        'Rank_g': 'rg',
        'Rank_g_full': 'rg_full'
    }
    rename_map = {}
    for alg, code in algorithm_codes.items():
        for metric, suff in metric_suffix_map.items():
            col = f"{alg}_{metric}"
            if col in wide_df.columns:
                rename_map[col] = f"{code}_{suff}"
    wide_df.rename(columns=rename_map, inplace=True)

    # Column ordering
    desired_cols = ['Cluster_Size','Iteration','Sensing_Range']
    order_codes = ['LPTf']
    order_codes.extend(f'LPTfLA{depth}' for depth in LOOKAHEAD_DEPTHS[1:])
    order_codes.append('LPTg')
    order_codes.extend(f'LPTgLA{depth}' for depth in LOOKAHEAD_DEPTHS[1:])
    order_codes.append('BF')
    for code in order_codes:
        for suff in ['A','Ag','t','f','gapf','g','gapg','rf','rf_full','rg','rg_full']:
            col = f'{code}_{suff}'
            if col in wide_df.columns:
                desired_cols.append(col)
    existing = [c for c in desired_cols if c in wide_df.columns]
    wide_df = wide_df[existing]

    # Averages sheet
    group_keys = ['Cluster_Size','Sensing_Range','Iteration','Algorithm']
    avg_f = (results_df.groupby(group_keys)['f(x)'].mean().unstack('Algorithm'))
    avg_f.columns = [f"{alg}_f(x)" for alg in avg_f.columns]
    if 'g(x)' in results_df.columns:
        avg_g = (results_df.groupby(group_keys)['g(x)'].mean().unstack('Algorithm'))
        avg_g.columns = [f"{alg}_g(x)" for alg in avg_g.columns]
    else:
        avg_g = pd.DataFrame(index=avg_f.index)
    if 'Execution_Time_Num' in results_df.columns:
        avg_time = (results_df.groupby(group_keys)['Execution_Time_Num'].mean().unstack('Algorithm'))
        avg_time.columns = [f"{alg}_Execution_Time" for alg in avg_time.columns]
    else:
        avg_time = pd.DataFrame(index=avg_f.index)

    avg_df = pd.concat([avg_f, avg_g, avg_time], axis=1).reset_index()

    # Bring in reference metrics from input (if present)
    metrics_df = results_df.groupby(
        ['Cluster_Size','Sensing_Range','Iteration']
    )[['Metric_1_Alpha_1','Metric_2_Alpha_1']].first().reset_index()
    avg_df = avg_df.merge(metrics_df, on=['Cluster_Size','Sensing_Range','Iteration'])

    # Optional gaps
    def _gap(df, val_col, ref_col, out_col):
        if val_col in df.columns and ref_col in df.columns:
            try:
                df[out_col] = ((df[val_col] - df[ref_col]) / df[ref_col] * 100).round(2)
            except ZeroDivisionError:
                df[out_col] = float('nan')

    if 'LPT_f(x)' in avg_df.columns:
        _gap(avg_df, 'LPT_f(x)', 'Metric_1_Alpha_1', 'LPT_f_opt_gap(%)')
    if 'LPT_g(x)' in avg_df.columns:
        _gap(avg_df, 'LPT_g(x)', 'Metric_2_Alpha_1', 'LPT_g_opt_gap(%)')

    # Std of gaps and times (optional)
    if 'Execution_Time' in results_df.columns and 'Execution_Time_Num' not in results_df.columns:
        try:
            results_df['Execution_Time_Num'] = results_df['Execution_Time'].astype(str).str.replace(',', '.').astype(float)
        except Exception:
            results_df['Execution_Time_Num'] = pd.NA

    if 'Execution_Time_Num' in results_df.columns:
        time_std = results_df.groupby(
            ['Cluster_Size','Sensing_Range','Algorithm']
        )['Execution_Time_Num'].std().unstack('Algorithm').reset_index()
        time_std.columns = [
            'Cluster_Size','Sensing_Range'
        ] + [f"{alg}_Execution_Time_std" for alg in time_std.columns[2:]]
        avg_df = avg_df.merge(time_std, on=['Cluster_Size','Sensing_Range'], how='left')

    # Save
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        wide_df.to_excel(writer, sheet_name='Results', index=False)
        avg_df.to_excel(writer, sheet_name='Averages', index=False)
        if fg_checks_rows:
            pd.DataFrame(fg_checks_rows).to_excel(writer, sheet_name='FG_Checks', index=False)

    print(f"\nResults and Averages saved to {output_file}")
