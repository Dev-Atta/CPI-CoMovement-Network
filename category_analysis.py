# category_analysis.py
# Each item belongs to one of seven economic categories (Food, Health, Transport, etc.)
# This file checks whether items in the same category tend to co-move more than
# items from different categories.
#
# We count:
#   intra-category edges : both items are in the SAME category
#   inter-category edges : the two items are in DIFFERENT categories

import pandas as pd
from collections import Counter
from config import OUTPUT_DIR
from graph_builder import load_graphs


def compute_category_analysis(graphs):
    """
    For each yearly graph, count intra vs inter category edges
    and find the category pairs that are most connected.
    """
    rows = []

    for year in sorted(graphs):
        g = graphs[year]["unweighted"]

        intra_count = 0
        inter_count = 0
        pair_counts = Counter()  # counts edges between each pair of categories

        for item_a, item_b in g.edges():
            cat_a = g.nodes[item_a].get("category", "Unknown")
            cat_b = g.nodes[item_b].get("category", "Unknown")

            # Use a sorted tuple so (Food, Health) == (Health, Food)
            pair_key = tuple(sorted([cat_a, cat_b]))
            pair_counts[pair_key] += 1

            if cat_a == cat_b:
                intra_count += 1
            else:
                inter_count += 1

        # Get the top 5 most connected category pairs
        top5 = " | ".join(
            f"{a} <-> {b}: {count}"
            for (a, b), count in pair_counts.most_common(5)
        )

        rows.append({
            "year":                 year,
            "intra_category_edges": intra_count,
            "inter_category_edges": inter_count,
            "top_5_category_pairs": top5,
        })

    return pd.DataFrame(rows)


def save_category(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "category_analysis.csv"
    df.to_csv(path, index=False)
    return path


def run_category_analysis(graphs=None):
    if graphs is None:
        graphs = load_graphs()
    df = compute_category_analysis(graphs)
    print("\n  Category connectivity:")
    print(df.to_string(index=False))
    save_category(df)
    return df