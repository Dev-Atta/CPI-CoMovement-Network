# centrality.py
# Compute three centrality measures for each item in each yearly graph.
#
# Degree centrality    : what fraction of all other items does this item co-move with?
#                        High value = this item's price moves with many others.
#
# Closeness centrality : how quickly can this item "reach" all other items in the network?
#                        High value = this item is close to everyone in terms of connections.
#
# Betweenness centrality: does this item sit on the shortest path between many other pairs?
#                          High value = this item acts as a bridge between different groups.

import networkx as nx
import pandas as pd
from pathlib import Path
from config import OUTPUT_DIR
from graph_builder import load_graphs


def compute_centrality(graphs):
    """
    Compute all three centrality measures for every item in every yearly graph.
    We use the unweighted graph so the scores are purely based on network structure.

    Returns a pandas DataFrame with one row per (year, item).
    """
    rows = []

    for year in sorted(graphs):
        g = graphs[year]["unweighted"]

        # NetworkX computes all three measures for us
        degree      = nx.degree_centrality(g)
        closeness   = nx.closeness_centrality(g)
        betweenness = nx.betweenness_centrality(g, normalized=True)

        for item in g.nodes():
            rows.append({
                "year":                   year,
                "item_name":              item,
                "category":               g.nodes[item].get("category", "Unknown"),
                "degree_centrality":      round(degree[item],      4),
                "closeness_centrality":   round(closeness[item],   4),
                "betweenness_centrality": round(betweenness[item], 4),
            })

    df = pd.DataFrame(rows)
    df = df.sort_values(["year", "item_name"]).reset_index(drop=True)
    return df


def print_top_centrality(df, top_n=5):
    """Print the top items per centrality metric for each year."""
    metrics = ["degree_centrality", "closeness_centrality", "betweenness_centrality"]

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year]
        print(f"\n  Year {year} — Top {top_n} items per centrality metric:")

        for metric in metrics:
            top = year_df.nlargest(top_n, metric)[["item_name", metric]]
            print(f"    {metric}:")
            for _, row in top.iterrows():
                print(f"      {row['item_name']:<35}  {row[metric]:.4f}")


def save_centrality(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "centrality_results.csv"
    df.to_csv(path, index=False)
    return path