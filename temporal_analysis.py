# temporal_analysis.py
# Compare the item co-movement networks across the three years.
#
# We want to know:
#   1. Which edges appeared or disappeared between years?
#      (Did certain item pairs start or stop co-moving?)
#   2. Which items stayed highly central across all three years?
#      (Are there persistent "hub" items that always drive price co-movement?)

import networkx as nx
import pandas as pd
from config import OUTPUT_DIR, PROJECT_YEARS
from graph_builder import load_graphs


def compute_temporal_analysis(graphs, centrality_df=None):
    """
    Analyse how the network changes from year to year.

    Returns a DataFrame summarising:
      - Edge appearance and disappearance between consecutive years
      - Which items are persistently in the top 5 for each centrality metric
    """
    rows = []
    years = [y for y in PROJECT_YEARS if y in graphs]

    # ── Part 1: Compare consecutive yearly graphs ──────────────────────────────
    for i in range(len(years) - 1):
        y0 = years[i]
        y1 = years[i + 1]

        # Convert edge lists to sets of (item_a, item_b) pairs for easy comparison
        # We sort each pair so (A, B) and (B, A) are treated the same
        edges_y0 = set()
        for u, v in graphs[y0]["unweighted"].edges():
            edges_y0.add(tuple(sorted([u, v])))

        edges_y1 = set()
        for u, v in graphs[y1]["unweighted"].edges():
            edges_y1.add(tuple(sorted([u, v])))

        appeared   = len(edges_y1 - edges_y0)   # edges in y1 but not in y0
        disappeared = len(edges_y0 - edges_y1)  # edges in y0 but not in y1

        rows.append({
            "transition":        f"{y0} to {y1}",
            "edges_appeared":    appeared,
            "edges_disappeared": disappeared,
            "net_change":        appeared - disappeared,
            "components_before": nx.number_connected_components(graphs[y0]["unweighted"]),
            "components_after":  nx.number_connected_components(graphs[y1]["unweighted"]),
        })

    # ── Part 2: Persistent top-5 items across all three years ──────────────────
    if centrality_df is not None and len(years) >= 3:
        metrics = ["degree_centrality", "closeness_centrality", "betweenness_centrality"]

        for metric in metrics:
            # Get the top-5 items for each year as a set
            top_sets = []
            for year in years[:3]:
                year_df = centrality_df[centrality_df["year"] == year]
                top5 = set(year_df.nlargest(5, metric)["item_name"].tolist())
                top_sets.append(top5)

            # Items in top-5 every single year
            persistent = top_sets[0] & top_sets[1] & top_sets[2]

            # Items in top-5 in at least two years
            near_persistent = set()
            near_persistent.update(top_sets[0] & top_sets[1])
            near_persistent.update(top_sets[1] & top_sets[2])
            near_persistent.update(top_sets[0] & top_sets[2])
            near_persistent -= persistent  # remove those already in persistent

            rows.append({
                "transition":        "all years",
                "metric":            metric,
                "persistent_top5":   " | ".join(sorted(persistent)) if persistent else "None",
                "near_persistent":   " | ".join(sorted(near_persistent)) if near_persistent else "None",
                "edges_appeared":    "",
                "edges_disappeared": "",
                "net_change":        "",
                "components_before": "",
                "components_after":  "",
            })

    return pd.DataFrame(rows)


def save_temporal(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "temporal_analysis.csv"
    df.to_csv(path, index=False)
    return path


def run_temporal_analysis(graphs=None, centrality_df=None):
    if graphs is None:
        graphs = load_graphs()
    if centrality_df is None:
        try:
            centrality_df = pd.read_csv(OUTPUT_DIR / "centrality_results.csv")
        except FileNotFoundError:
            pass

    df = compute_temporal_analysis(graphs, centrality_df)
    print("\n  Temporal analysis:")
    print(df.to_string(index=False))
    save_temporal(df)
    return df