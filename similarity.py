# similarity.py
# For every pair of items in the same city and year, we measure how similar
# their price-change patterns are using cosine similarity.
#
# Cosine similarity = (dot product of two vectors) / (product of their lengths)
# A value of 1.0 means identical patterns. 0.0 means no relationship. -1.0 means opposite.

import math
import pandas as pd
from config import PROJECT_YEARS


def cosine_similarity(v1, v2):
    """
    Compute the cosine similarity between two price-change vectors.

    The dot product measures how much two vectors point in the same direction.
    Dividing by both vector lengths makes the result independent of scale.
    """
    # Dot product: multiply matching elements and sum them up
    dot_product = sum(a * b for a, b in zip(v1, v2))

    # Length (magnitude) of each vector: square root of sum of squares
    length_v1 = math.sqrt(sum(a ** 2 for a in v1))
    length_v2 = math.sqrt(sum(b ** 2 for b in v2))

    # If either vector is all zeros (no price movement), similarity is 0
    if length_v1 == 0 or length_v2 == 0:
        return 0.0

    return dot_product / (length_v1 * length_v2)


def compute_similarity_matrices(vectors):
    """
    For each (year, city) pair, compute cosine similarity for every pair of items.

    Returns: sim_matrices[year][city] = a pandas DataFrame (items x items)
             where each cell contains the cosine similarity between two items.
    """
    sim_matrices = {}

    for year in PROJECT_YEARS:
        sim_matrices[year] = {}

        for city, item_vectors in vectors[year].items():
            items = sorted(item_vectors.keys())
            n = len(items)

            # Build an empty n x n matrix to hold similarity scores
            sim_data = {}
            for item in items:
                sim_data[item] = {}

            # Compute similarity for every pair of items (i, j)
            for i in range(n):
                for j in range(n):
                    item_i = items[i]
                    item_j = items[j]

                    if i == j:
                        # An item is perfectly similar to itself
                        sim_data[item_i][item_j] = 1.0
                    else:
                        sim = cosine_similarity(
                            item_vectors[item_i],
                            item_vectors[item_j]
                        )
                        sim_data[item_i][item_j] = sim

            sim_matrices[year][city] = pd.DataFrame(sim_data, index=items, columns=items)

    return sim_matrices