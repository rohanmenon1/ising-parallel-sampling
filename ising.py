import numpy as np
from itertools import product


def exact_distribution(n, J):
    """
    Calculate the exact probability distribution of a 1D Ising chain.

    Parameters:
        n : int
            Number of spins.
        J : float
            Coupling strength between neighboring spins.

    Returns:
        configs : numpy array, shape (2**n, n)
            All possible spin configurations (-1, +1).
        P : numpy array, shape (2**n,)
            Exact normalized probability of each configuration.
    """

    # Enumerate all possible spin configurations
    configs = np.array(list(product([-1, 1], repeat=n)))

    # Calculate energy for each configuration
    energies = np.zeros(len(configs))

    for i in range(n - 1):
        energies -= J * configs[:, i] * configs[:, i + 1]

    # Boltzmann probabilities (k = 1, T = 1)
    weights = np.exp(-energies)

    # Normalize probabilities
    P = weights / np.sum(weights)

    return configs, P


def conditional(block, committed, configs, P):
    """
    Calculate P(X_block | X_committed).

    Parameters:
        block : list of int
            Indices of spins to sample.
        committed : dict
            Previously committed spins, {index: spin_value}.
        configs : numpy array
            All possible configurations.
        P : numpy array
            Exact probability distribution.

    Returns:
        probabilities : numpy array
            Conditional probability for every possible assignment
            to the requested block.

    The assignments are ordered lexicographically using
    itertools.product([-1, 1], repeat=len(block)).
    """

    # Enumerate all possible assignments to the requested block
    assignments = list(product([-1, 1], repeat=len(block)))

    probabilities = np.zeros(len(assignments))

    # Filter configurations consistent with committed spins
    committed_mask = np.ones(len(configs), dtype=bool)

    for index, value in committed.items():
        committed_mask &= (configs[:, index] == value)

    # Calculate P(X_committed)
    denominator = np.sum(P[committed_mask])

    if denominator == 0:
        raise ValueError("Committed configuration has zero probability.")

    # Calculate conditional probability for each block assignment
    for i, assignment in enumerate(assignments):

        mask = committed_mask.copy()

        for index, value in zip(block, assignment):
            mask &= (configs[:, index] == value)

        numerator = np.sum(P[mask])

        probabilities[i] = numerator / denominator

    return probabilities