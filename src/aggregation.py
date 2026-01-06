import numpy as np


def aggregate_scores(scores):
    """
    Conservative aggregation:
    one strong contradiction dominates.
    """
    if not scores:
        return 0.0

    max_score = max(scores)
    mean_score = np.mean(scores)
    contradiction_count = sum(s > 0.7 for s in scores)

    return {
        "max_score": max_score,
        "mean_score": mean_score,
        "contradiction_count": contradiction_count
    }
