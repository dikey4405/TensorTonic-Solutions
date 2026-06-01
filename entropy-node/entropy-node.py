import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    # Write code here
    _, counts = np.unique(y, return_counts=True)

    probs = counts / len(y)

    entropy = -np.sum(probs * np.log2(probs))
    return entropy
    pass