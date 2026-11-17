"""Compare exact output laws of three parallel Ising sampling schedules.

Requires ising.exact_distribution(n, J) -> (configs, P) for an open chain,
and the conditional oracle documented in sampling.py. All laws use the
same configuration ordering; no Monte Carlo estimates are used.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np


def tv_distance(P, Q):
    """Return total variation distance between aligned probability arrays."""
    P, Q = np.asarray(P), np.asarray(Q)
    if P.shape != Q.shape:
        raise ValueError("P and Q must have the same shape")
    return float(0.5 * np.abs(P - Q).sum())


def check_distribution(probabilities):
    """Fail on invalid probabilities rather than silently renormalizing them."""
    assert np.all(np.isfinite(probabilities)), "nonfinite probabilities"
    assert np.all(probabilities >= 0), "negative probabilities"
    np.testing.assert_allclose(probabilities.sum(), 1.0, rtol=0, atol=1e-12)


def run_experiment(n=8, J=0.5, k=1, output="tv_error_vs_rounds.png", show=True):
    """Check and evaluate schedules, plot errors, and return result records.

    k is a maximum block size. The checkerboard baseline always uses singletons.
    Exact enumeration costs grow exponentially with n.
    """
    if not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(k, (int, np.integer)) or not 1 <= k <= n:
        raise ValueError("k must be an integer with 1 <= k <= n")
    if not np.isfinite(J):
        raise ValueError("J must be finite")

    # Defer project imports so --help and the TV helper work independently.
    from ising import exact_distribution
    from sampling import (
        checkerboard_schedule,
        exact_schedule,
        independent_schedule,
        output_distribution,
        validate_schedule,
    )

    schedules = [
        ("One-round independent", independent_schedule(n, k)),
        ("Two-round checkerboard", checkerboard_schedule(n)),
        ("Recursive separator", exact_schedule(n, k)),
    ]
    for _, schedule in schedules:
        validate_schedule(schedule, n)

    # At J=0 every spin is independent, so every valid schedule is exact.
    zero_configs, zero_P = exact_distribution(n, 0.0)
    check_distribution(zero_P)
    np.testing.assert_allclose(zero_P, 1.0 / (2 ** n), rtol=0, atol=1e-12)
    for _, schedule in schedules:
        zero_Q = output_distribution(schedule, zero_configs, zero_P)
        check_distribution(zero_Q)
        np.testing.assert_allclose(zero_Q, zero_P, rtol=0, atol=1e-12)

    configs, P = exact_distribution(n, J)
    check_distribution(P)
    results = []
    for name, schedule in schedules:
        Q = output_distribution(schedule, configs, P)
        check_distribution(Q)
        if name == "Recursive separator":
            np.testing.assert_allclose(Q, P, rtol=0, atol=1e-12)
        results.append({"schedule": name, "rounds": len(schedule),
                        "tv_error": tv_distance(P, Q)})

    print(f"Open Ising chain: n={n}, J={J}, k={k} (maximum block size)")
    print("Checks passed: normalized laws, J=0 sanity check, recursive exactness.")
    print(f"{'Schedule':<26} {'Rounds':>6} {'TV error':>14}")
    for row in results:
        print(f"{row['schedule']:<26} {row['rounds']:>6} {row['tv_error']:>14.8g}")

    fig, ax = plt.subplots(figsize=(8, 5))
    for row in results:
        ax.scatter(row["rounds"], row["tv_error"], label=row["schedule"])
    ax.set_xlabel("Number of rounds")
    ax.set_ylabel("Total variation distance")
    ax.set_title(f"Parallel sampling: n={n}, J={J}, k={k}")
    ax.set_xticks(sorted({row["rounds"] for row in results}))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if output is not None:
        fig.savefig(output, dpi=150)
        print(f"Plot saved to {output}")
    if show:
        plt.show()
    plt.close(fig)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=8, help="number of spins (default: 8)")
    parser.add_argument("--J", type=float, default=0.5, help="coupling (default: 0.5)")
    parser.add_argument("--k", type=int, default=1, help="maximum block size (default: 1)")
    parser.add_argument("--output", default="tv_error_vs_rounds.png", help="plot file path")
    parser.add_argument("--no-show", action="store_true", help="save without opening a window")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    if not 1 <= args.k <= args.n:
        parser.error("--k must satisfy 1 <= k <= n")
    if not np.isfinite(args.J):
        parser.error("--J must be finite")
    run_experiment(args.n, args.J, args.k, args.output, not args.no_show)


if __name__ == "__main__":
    main()
