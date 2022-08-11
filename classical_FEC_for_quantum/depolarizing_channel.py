import numpy as np

'De-correlated depolarizing channel'


def depolarizing_channel(qubit, d):
    # d = np.random.random() * 0.75
    d = d
    prob = 0.5 - 0.5 * np.sqrt(1 - (4 / 3) * d)

    prob_choices = [1, 2]

    pick = np.random.choice(prob_choices, p=[1 - prob, prob])

    if pick == 1:
        qubit.I()
    elif pick == 2:
        pauli_gates = ['X', 'Y', 'Z']
        gate = np.random.choice(pauli_gates)
        if gate == 'X':
            qubit.X()
        elif gate == 'Y':
            qubit.Y()
        else:
            qubit.Z()
    return qubit


