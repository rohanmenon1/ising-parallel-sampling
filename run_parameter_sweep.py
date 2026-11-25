"""Execute comprehensive parameter sweep for Ising parallel sampling.

Grid parameters:
- n: 16
- J: [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5]
- k: [1, 2, 3, 4, 6, 8]
- s: [1, 2, ..., 16]

Outputs:
- experiment_results.csv
- experiment_results.log
"""

import csv
from itertools import product
from pathlib import Path
import sys
import time

import numpy as np

REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from ising import conditional, exact_distribution
from sampling import (
    approximate_schedule,
    checkerboard_schedule,
    exact_schedule,
    independent_schedule,
    validate_schedule,
)


def tv_distance(P: np.ndarray, Q: np.ndarray) -> float:
    return float(0.5 * np.sum(np.abs(P - Q)))


def fast_output_distribution(schedule: list, configs: np.ndarray, P: np.ndarray) -> np.ndarray:
    n = configs.shape[1]
    validate_schedule(schedule, n)
    Q = np.ones(len(configs), dtype=np.float64)
    committed_spins = set()

    for round_blocks in schedule:
        for block in round_blocks:
            left_neighbors = [idx for idx in committed_spins if idx < min(block)]
            right_neighbors = [idx for idx in committed_spins if idx > max(block)]
            L = max(left_neighbors) if left_neighbors else None
            R = min(right_neighbors) if right_neighbors else None

            # Fast bitshift mapping for block configurations
            b_bits = (configs[:, block] == 1).astype(np.int32)
            b_powers = 2 ** np.arange(len(block) - 1, -1, -1, dtype=np.int32)
            block_idx = b_bits @ b_powers

            neighbor_spins = ([L] if L is not None else []) + ([R] if R is not None else [])
            if not neighbor_spins:
                probs = conditional(block, {}, configs, P)
                Q *= probs[block_idx]
            else:
                n_states = list(product([-1, 1], repeat=len(neighbor_spins)))
                table = np.zeros((len(n_states), 2 ** len(block)), dtype=np.float64)
                for s_i, state in enumerate(n_states):
                    comm_dict = dict(zip(neighbor_spins, state))
                    table[s_i] = conditional(block, comm_dict, configs, P)

                n_bits = (configs[:, neighbor_spins] == 1).astype(np.int32)
                n_powers = 2 ** np.arange(len(neighbor_spins) - 1, -1, -1, dtype=np.int32)
                state_idx = n_bits @ n_powers

                Q *= table[state_idx, block_idx]

        for block in round_blocks:
            committed_spins.update(block)

    return Q


def main():
    n = 16
    J_values = [0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
    k_values = [1, 2, 3, 4, 6, 8]
    s_values = list(range(1, 17))

    csv_path = REPO_DIR / "experiment_results.csv"
    log_path = REPO_DIR / "experiment_results.log"

    print(f"Starting parameter sweep: n={n}")
    print(f"J values ({len(J_values)}): {J_values}")
    print(f"k values ({len(k_values)}): {k_values}")
    print(f"s values ({len(s_values)}): 1..16")
    print(f"Logging to: {csv_path.name} and {log_path.name}")

    start_total = time.time()
    total_runs = 0

    with open(csv_path, mode="w", newline="", encoding="utf-8") as f_csv, \
         open(log_path, mode="w", encoding="utf-8") as f_log:

        csv_writer = csv.writer(f_csv)
        csv_writer.writerow(["n", "J", "k", "schedule", "spacing", "rounds", "tv_error"])

        header = f"{'n':>3} {'J':>5} {'k':>3} {'Schedule':<26} {'Spacing':>7} {'Rounds':>6} {'TV Error':>15}\n"
        divider = "-" * 72 + "\n"
        f_log.write(header)
        f_log.write(divider)

        for J_idx, J in enumerate(J_values, 1):
            t_j = time.time()
            print(f"[{J_idx}/{len(J_values)}] Computing P for J={J}...")
            configs, P = exact_distribution(n, J)

            for k in k_values:
                # 1. Baseline schedules
                schedules_to_test = [
                    ("Independent", independent_schedule(n, k), None),
                    ("Checkerboard", checkerboard_schedule(n), None),
                    ("Recursive separator", exact_schedule(n, k), None),
                ]

                # 2. Approximate schedules for each spacing s
                for s in s_values:
                    schedules_to_test.append(
                        (f"Approximate (spacing={s})", approximate_schedule(n, k, s), s)
                    )

                for name, sched, sp in schedules_to_test:
                    Q = fast_output_distribution(sched, configs, P)
                    tv = tv_distance(P, Q)
                    rounds = len(sched)
                    sp_val = sp if sp is not None else ""

                    csv_writer.writerow([n, J, k, name, sp_val, rounds, f"{tv:.10e}"])

                    sp_display = str(sp) if sp is not None else "-"
                    f_log.write(f"{n:>3} {J:>5.2f} {k:>3} {name:<26} {sp_display:>7} {rounds:>6} {tv:>15.6e}\n")
                    total_runs += 1

            f_log.flush()
            f_csv.flush()
            print(f"      Completed J={J} in {time.time() - t_j:.2f}s")

    elapsed_total = time.time() - start_total
    print("=" * 70)
    print(f"Finished {total_runs} experimental configurations in {elapsed_total:.2f} seconds!")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved Log: {log_path}")


if __name__ == "__main__":
    main()
