"""Sampling schedules and evaluation for the Ising-chain MVP.

Requires ising.py to provide:
    conditional(block, committed, configs, P) -> probabilities
    ordered as itertools.product([-1, 1], repeat=len(block))

All spin indices are zero-based. A schedule is a list of rounds;
each round is a list of blocks, and each block is a list of spin indices.
For example, [[[1]], [[0], [2]]] samples spin 1 first, then 0 and 2
independently conditioned on spin 1.
"""

import numpy as np
from itertools import product
from ising import conditional


def independent_schedule(n, k):
    """One round: disjoint consecutive blocks sampled independently."""
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    return [[list(range(i, min(i + k, n))) for i in range(0, n, k)]]


def checkerboard_schedule(n):
    """Two rounds, singleton blocks: odd indices, then even indices."""
    return [round_blocks for round_blocks in [
        [[i] for i in range(1, n, 2)],
        [[i] for i in range(0, n, 2)],
    ] if round_blocks]


def exact_schedule(n, k, committed=()):
    """Proof's exact policy: jointly sample up to k separators per interval."""
    if not (1 <= k <= n):
        raise ValueError("require 1 <= k <= n")
    known = set(committed)
    rounds = []

    while len(known) < n:
        boundaries = [-1] + sorted(known) + [n]
        blocks = []

        for left, right in zip(boundaries, boundaries[1:]):
            length = right - left - 1  # Number of unrevealed spins here
            if length == 0:
                continue

            count = min(k, length)
            # Place separators roughly evenly within this interval.
            block = [
                left + (j * (length + 1)) // (count + 1)
                for j in range(1, count + 1)
            ]
            blocks.append(block)  # Joint WITHIN a block

        rounds.append(blocks)      # Independent BETWEEN blocks
        known.update(i for block in blocks for i in block)

    return rounds


def approximate_schedule(n, k, spacing):
    """Proof's approximate policy: sparse first round, exact completion."""
    if not (1 <= spacing <= n):
        raise ValueError("spacing must lie between 1 and n")

    separators = list(range(spacing - 1, n, spacing))
    first_round = [
        separators[i:i + k]
        for i in range(0, len(separators), k)
    ]
    return [first_round] + exact_schedule(n, k, committed=separators)


def validate_schedule(schedule, n):
    """Ensure every spin is sampled once, and no round has empty blocks."""
    indices = []
    for round_blocks in schedule:
        if not round_blocks:
            raise ValueError("empty round")
        for block in round_blocks:
            if not block:
                raise ValueError("empty block")
            indices.extend(block)
    if sorted(indices) != list(range(n)):
        raise ValueError("each spin must occur in exactly one block")


def sample(schedule, configs, P, rng=None):
    """Draw ONE complete configuration by calling the conditional oracle."""
    n = configs.shape[1]
    validate_schedule(schedule, n)
    rng = np.random.default_rng() if rng is None else rng
    committed = {}

    for round_blocks in schedule:
        new_values = {}
        for block in round_blocks:
            probs = conditional(block, committed, configs, P)
            assignments = list(product([-1, 1], repeat=len(block)))
            chosen = assignments[rng.choice(len(assignments), p=probs)]
            for index, value in zip(block, chosen):
                new_values[index] = int(value)

        # Commit AFTER the round; simultaneous blocks do not see one another.
        committed.update(new_values)

    return np.array([committed[i] for i in range(n)], dtype=int)


def output_distribution(schedule, configs, P):
    """Compute the sampler's ENTIRE output law Q, without Monte Carlo."""
    n = configs.shape[1]
    validate_schedule(schedule, n)
    Q = np.zeros(len(configs))

    for row, x in enumerate(configs):
        committed = {}
        probability = 1.0

        for round_blocks in schedule:
            for block in round_blocks:
                probs = conditional(block, committed, configs, P)
                assignments = list(product([-1, 1], repeat=len(block)))
                target = tuple(int(x[i]) for i in block)
                probability *= probs[assignments.index(target)]

            # Make this round visible only after ALL its block probabilities.
            for block in round_blocks:
                for index in block:
                    committed[index] = int(x[index])

        Q[row] = probability

    return Q
