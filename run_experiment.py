"""Run Ising parallel sampling experiments and analyze the rounds vs. TV error trade-off.

This runner evaluates all sampling policies defined in sampling.py:
1. One-round independent schedule (k-sized disjoint blocks)
2. Two-round checkerboard schedule (odd, then even spins)
3. Approximate schedules (sparse first-round separators with spacing s, exact completion)
4. Recursive separator schedule (exact hierarchical policy)

It compares their exact output laws Q against the true Boltzmann distribution P,
computes the Total Variation (TV) error vs. number of parallel rounds,
optionally verifies the empirical Monte Carlo sampler, and generates publication-quality plots.
"""

from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path
import sys

# Ensure repository root is in sys.path regardless of invocation directory
REPO_DIR = Path(__file__).resolve().parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

import matplotlib.pyplot as plt
import numpy as np

from ising import conditional, exact_distribution
from sampling import (
    approximate_schedule,
    checkerboard_schedule,
    exact_schedule,
    independent_schedule,
    sample,
    validate_schedule,
)


def tv_distance(P: np.ndarray, Q: np.ndarray) -> float:
    """Return total variation distance between two aligned probability distributions."""
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.shape != Q.shape:
        raise ValueError(f"Shape mismatch: {P.shape} vs {Q.shape}")
    return float(0.5 * np.sum(np.abs(P - Q)))


def check_distribution(probabilities: np.ndarray, atol: float = 1e-10) -> None:
    """Validate that an array represents a well-formed probability distribution."""
    probs = np.asarray(probabilities)
    if not np.all(np.isfinite(probs)):
        raise ValueError("Probabilities contain NaN or infinite values.")
    if np.any(probs < -1e-14):
        raise ValueError(f"Probabilities contain negative values: min = {np.min(probs)}")
    total = np.sum(probs)
    if not np.isclose(total, 1.0, atol=atol):
        raise ValueError(f"Probabilities do not sum to 1.0: sum = {total}")


def fast_output_distribution(schedule: list, configs: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Compute the sampler's exact output law Q efficiently using 1D Markov blankets and vectorization."""
    n = configs.shape[1]
    validate_schedule(schedule, n)
    Q = np.ones(len(configs), dtype=np.float64)
    committed_spins: set[int] = set()

    for round_blocks in schedule:
        for block in round_blocks:
            left_neighbors = [idx for idx in committed_spins if idx < min(block)]
            right_neighbors = [idx for idx in committed_spins if idx > max(block)]
            L = max(left_neighbors) if left_neighbors else None
            R = min(right_neighbors) if right_neighbors else None

            assignments = list(product([-1, 1], repeat=len(block)))
            assign_map = {a: i for i, a in enumerate(assignments)}
            block_tuples = [tuple(row) for row in configs[:, block]]
            block_idx = np.array([assign_map[t] for t in block_tuples], dtype=np.int32)

            neighbor_spins = ([L] if L is not None else []) + ([R] if R is not None else [])
            if not neighbor_spins:
                probs = conditional(block, {}, configs, P)
                Q *= probs[block_idx]
            else:
                n_states = list(product([-1, 1], repeat=len(neighbor_spins)))
                table = np.zeros((len(n_states), len(assignments)), dtype=np.float64)
                for s_i, state in enumerate(n_states):
                    comm_dict = dict(zip(neighbor_spins, state))
                    table[s_i] = conditional(block, comm_dict, configs, P)

                state_map = {s: i for i, s in enumerate(n_states)}
                state_tuples = [tuple(row) for row in configs[:, neighbor_spins]]
                state_idx = np.array([state_map[t] for t in state_tuples], dtype=np.int32)

                Q *= table[state_idx, block_idx]

        for block in round_blocks:
            committed_spins.update(block)

    return Q


def evaluate_schedule(
    name: str,
    schedule: list,
    configs: np.ndarray,
    P: np.ndarray,
    spacing: int | None = None,
) -> dict:
    """Validate schedule, compute its output distribution Q, and record metrics."""
    n = configs.shape[1]
    validate_schedule(schedule, n)
    Q = fast_output_distribution(schedule, configs, P)
    check_distribution(Q)
    tv = tv_distance(P, Q)
    return {
        "schedule": name,
        "rounds": len(schedule),
        "spacing": spacing,
        "tv_error": tv,
        "Q": Q,
    }


def run_experiment(
    n: int = 8,
    J: float = 0.5,
    k: int = 1,
    test_spacings: bool = True,
    mc_samples: int = 0,
    output_path: Path | str | None = None,
    show: bool = False,
) -> list[dict]:
    """Execute the full experimental suite for Ising parallel sampling."""
    if output_path is None:
        output_path = REPO_DIR / "tv_error_vs_rounds.png"
    else:
        output_path = Path(output_path)

    print("=" * 70)
    print(f"Ising Parallel Sampling Experiment: n={n}, J={J}, k={k}")
    print("=" * 70)

    # 1. Sanity check: At J=0, every spin is independent; all valid schedules must be exact.
    print("[1/4] Running J=0 sanity check...")
    zero_configs, zero_P = exact_distribution(n, 0.0)
    check_distribution(zero_P)
    baseline_schedules = [
        ("Independent", independent_schedule(n, k)),
        ("Checkerboard", checkerboard_schedule(n)),
        ("Recursive separator", exact_schedule(n, k)),
    ]
    for b_name, b_sched in baseline_schedules:
        validate_schedule(b_sched, n)
        zero_Q = fast_output_distribution(b_sched, zero_configs, zero_P)
        check_distribution(zero_Q)
        np.testing.assert_allclose(zero_Q, zero_P, atol=1e-12,
                                   err_msg=f"Schedule {b_name} failed exactness check at J=0")
    print("      J=0 check passed: all schedules exact under independence.")

    # 2. Compute exact ground truth distribution at coupling J
    print(f"[2/4] Computing exact Ising Boltzmann distribution (2^{n} = {2**n} states)...")
    configs, P = exact_distribution(n, J)
    check_distribution(P)

    # 3. Evaluate schedules
    print("[3/4] Evaluating parallel sampling schedules...")
    results: list[dict] = []

    # Baselines
    for b_name, b_sched in baseline_schedules:
        res = evaluate_schedule(b_name, b_sched, configs, P)
        results.append(res)

    # Verify recursive separator is analytically exact
    rec_res = next(r for r in results if r["schedule"] == "Recursive separator")
    if not np.isclose(rec_res["tv_error"], 0.0, atol=1e-10):
        raise AssertionError(f"Recursive separator not exact! TV error = {rec_res['tv_error']}")

    # Approximate policies across spacings: s = 1..n
    if test_spacings:
        for s in range(1, n + 1):
            sched = approximate_schedule(n, k, spacing=s)
            res = evaluate_schedule(f"Approximate (spacing={s})", sched, configs, P, spacing=s)
            results.append(res)

    # Optional Monte Carlo sampling verification
    if mc_samples > 0:
        print(f"      Verifying Monte Carlo sampler using {mc_samples} draws...")
        rng = np.random.default_rng(42)
        # Check on Checkerboard schedule
        cb_sched = checkerboard_schedule(n)
        empirical_counts = np.zeros(len(configs))
        # Map config tuple to index for fast histogramming
        config_to_idx = {tuple(int(x) for x in row): idx for idx, row in enumerate(configs)}
        for _ in range(mc_samples):
            drawn = sample(cb_sched, configs, P, rng=rng)
            empirical_counts[config_to_idx[tuple(drawn)]] += 1
        empirical_Q = empirical_counts / mc_samples
        mc_tv = tv_distance(P, empirical_Q)
        print(f"      Checkerboard Monte Carlo TV error (M={mc_samples}): {mc_tv:.4f}")

    # Display results table
    print("\n" + "-" * 70)
    print(f"{'Schedule':<30} {'Rounds':>8} {'Spacing':>9} {'TV Error':>16}")
    print("-" * 70)
    for r in results:
        sp_str = str(r["spacing"]) if r["spacing"] is not None else "-"
        print(f"{r['schedule']:<30} {r['rounds']:>8} {sp_str:>9} {r['tv_error']:>16.8e}")
    print("-" * 70)

    # 4. Generate Plot
    print(f"[4/4] Plotting TV Error vs. Rounds...")
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Plot baselines
    for r in results:
        if r["spacing"] is None:
            ax.scatter(
                r["rounds"],
                r["tv_error"],
                s=90,
                zorder=5,
                label=f"{r['schedule']} ({r['rounds']} rounds)",
            )

    # Plot approximate schedules as connected curve
    approx_results = [r for r in results if r["spacing"] is not None]
    if approx_results:
        # Sort by spacing
        approx_results.sort(key=lambda x: x["spacing"])
        rounds_app = [r["rounds"] for r in approx_results]
        tv_app = [r["tv_error"] for r in approx_results]

        ax.plot(rounds_app, tv_app, "o--", color="tab:purple", alpha=0.7,
                label="Approximate (varying spacing s=1..n)", zorder=3)
        for r in approx_results:
            ax.annotate(
                f"s={r['spacing']}",
                (r["rounds"], r["tv_error"]),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=8,
                alpha=0.85,
            )

    all_rounds = sorted({r["rounds"] for r in results})
    ax.set_xlabel("Number of Parallel Rounds", fontsize=11)
    ax.set_ylabel("Total Variation Distance $d_{TV}(P, Q)$", fontsize=11)
    ax.set_title(f"Parallel Ising Chain Sampling: TV Error vs. Rounds\n(n={n}, J={J}, max block size k={k})", fontsize=12)
    ax.set_xticks(all_rounds)
    ax.set_ylim(bottom=-0.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(frameon=True)
    fig.tight_layout()

    fig.savefig(output_path, dpi=150)
    print(f"      Plot successfully saved to: {output_path}")

    if show:
        plt.show()
    plt.close(fig)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Ising parallel sampling experiments and evaluate TV error vs. rounds trade-off."
    )
    parser.add_argument("--n", type=int, default=8, help="Number of spins in Ising chain (default: 8)")
    parser.add_argument("--J", type=float, default=0.5, help="Coupling constant J (default: 0.5)")
    parser.add_argument("--k", type=int, default=1, help="Maximum separator block size (default: 1)")
    parser.add_argument(
        "--no-spacings",
        action="store_true",
        help="Skip evaluating approximate schedules across all spacings",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=0,
        help="Number of Monte Carlo draws to verify sampling.py (0 to skip, default: 0)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="File path to save the output plot (default: tv_error_vs_rounds.png in repo dir)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plot interactively in GUI window",
    )
    args = parser.parse_args()

    if args.n < 1:
        parser.error("--n must be a positive integer.")
    if not (1 <= args.k <= args.n):
        parser.error("--k must satisfy 1 <= k <= n.")
    if not np.isfinite(args.J):
        parser.error("--J must be finite.")

    run_experiment(
        n=args.n,
        J=args.J,
        k=args.k,
        test_spacings=not args.no_spacings,
        mc_samples=args.mc_samples,
        output_path=args.output,
        show=args.show,
    )


if __name__ == "__main__":
    main()
