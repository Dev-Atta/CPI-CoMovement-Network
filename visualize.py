# visualize.py
# Generate all plots for the project report and demo video.

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from collections import Counter
from config import OUTPUT_DIR, CATEGORIES, SIMILARITY_THRESHOLD, MIN_CITIES
from graph_builder import load_graphs

PLOTS_DIR = OUTPUT_DIR / "plots"

# One colour per category — used consistently across all plots
CATEGORY_COLORS = {
    "Food & Non-Alcoholic Beverages": "#1f77b4",
    "Alcoholic Beverages, Tobacco":   "#ff7f0e",
    "Clothing and Footwear":          "#2ca02c",
    "Housing, Water, Electricity":    "#d62728",
    "Furnishing & HH Equipment":      "#9467bd",
    "Health":                         "#8c564b",
    "Transport":                      "#e377c2",
    "Unknown":                        "#7f7f7f",
}


def _layout(g):
    """
    Compute node positions using a spring layout.
    Connected nodes are spread out. Isolated nodes go on an outer ring.
    """
    connected = [n for n in g.nodes() if g.degree(n) > 0]
    isolated  = [n for n in g.nodes() if g.degree(n) == 0]

    pos = {}

    if connected:
        sub = g.subgraph(connected)
        k_val = 2.5 / max(len(connected) ** 0.5, 1)
        pos.update(nx.spring_layout(sub, seed=42, k=k_val, iterations=100))

    if isolated:
        angles = np.linspace(0, 2 * 3.14159, len(isolated), endpoint=False)
        for node, angle in zip(isolated, angles):
            pos[node] = (2.5 * np.cos(angle), 2.5 * np.sin(angle))

    return pos


def _category_legend(ax, fontsize=8):
    """Build a category colour legend for a network plot."""
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=CATEGORY_COLORS.get(cat, "#7f7f7f"),
                   label=cat, markersize=10)
        for cat in CATEGORIES if cat in CATEGORY_COLORS
    ]
    ax.legend(handles=handles, title="Category", loc="upper left",
              fontsize=fontsize, framealpha=0.9)


# ── Plot 1: Yearly network graphs ─────────────────────────────────────────────

def plot_network_by_year(graphs):
    """One category-coloured network plot per year (unweighted graph)."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for year in sorted(graphs):
        g = graphs[year]["unweighted"]
        pos = _layout(g)

        node_colors = [
            CATEGORY_COLORS.get(g.nodes[n].get("category", "Unknown"), "#7f7f7f")
            for n in g.nodes()
        ]
        # Only label nodes that have at least one connection
        labels = {n: str(n)[:20] for n in g.nodes() if g.degree(n) > 0}

        fig, ax = plt.subplots(figsize=(20, 16))
        nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.20, width=0.7, edge_color="#888888")
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=220, alpha=0.92)
        nx.draw_networkx_labels(g, pos, labels=labels, ax=ax, font_size=6)
        _category_legend(ax)
        ax.set_title(
            f"Item Co-Movement Network ({year})  |  tau={SIMILARITY_THRESHOLD}, K={MIN_CITIES}",
            fontsize=14,
        )
        ax.axis("off")
        fig.tight_layout()

        path = PLOTS_DIR / f"network_{year}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output_paths.append(path)

    return output_paths


# ── Plot 2: Weighted network comparison ───────────────────────────────────────

def plot_weighted_comparison(graphs):
    """
    Side-by-side comparison of the two weighted graph variants per year.
    Left:  edge width proportional to city-count weight.
    Right: edge colour scaled by average similarity weight.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for year in sorted(graphs):
        g_cc = graphs[year]["weighted_city_count"]
        g_as = graphs[year]["weighted_avg_similarity"]
        g_un = graphs[year]["unweighted"]
        pos = _layout(g_un)

        node_colors = [
            CATEGORY_COLORS.get(g_un.nodes[n].get("category", "Unknown"), "#7f7f7f")
            for n in g_un.nodes()
        ]
        labels = {n: str(n)[:20] for n in g_un.nodes() if g_un.degree(n) > 0}

        fig, axes = plt.subplots(1, 2, figsize=(28, 14))

        # Left panel: city-count weights as edge width
        cc_weights = [d.get("weight", 1.0) for _, _, d in g_cc.edges(data=True)]
        max_cc = max(cc_weights) if cc_weights else 1.0
        cc_widths = [(w / max_cc) * 4.0 for w in cc_weights]

        if cc_widths:
            nx.draw_networkx_edges(g_cc, pos, ax=axes[0], width=cc_widths,
                                   alpha=0.40, edge_color="#333333")
        nx.draw_networkx_nodes(g_cc, pos, ax=axes[0], node_color=node_colors,
                               node_size=200, alpha=0.92)
        nx.draw_networkx_labels(g_cc, pos, labels=labels, ax=axes[0], font_size=5.5)
        axes[0].set_title(f"Weighted by City Count ({year})", fontsize=12)
        axes[0].axis("off")

        # Right panel: avg similarity weights as edge colour
        as_weights = [d.get("weight", 0.0) for _, _, d in g_as.edges(data=True)]
        if as_weights:
            vmin, vmax = min(as_weights), max(as_weights)
            if vmax <= vmin:
                vmax = vmin + 0.0001
            nx.draw_networkx_edges(g_as, pos, ax=axes[1], width=1.5, alpha=0.75,
                                   edge_color=as_weights,
                                   edge_cmap=plt.cm.Blues,
                                   edge_vmin=vmin, edge_vmax=vmax)
            sm = plt.cm.ScalarMappable(
                cmap=plt.cm.Blues,
                norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
            )
            sm.set_array([])
            fig.colorbar(sm, ax=axes[1], fraction=0.04, pad=0.04, label="Avg Similarity")

        nx.draw_networkx_nodes(g_as, pos, ax=axes[1], node_color=node_colors,
                               node_size=200, alpha=0.92)
        nx.draw_networkx_labels(g_as, pos, labels=labels, ax=axes[1], font_size=5.5)
        axes[1].set_title(f"Weighted by Avg Similarity ({year})", fontsize=12)
        axes[1].axis("off")

        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=CATEGORY_COLORS.get(cat, "#7f7f7f"),
                       label=cat, markersize=9)
            for cat in CATEGORIES if cat in CATEGORY_COLORS
        ]
        fig.legend(handles=legend_handles, title="Category", loc="lower center",
                   ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.02), framealpha=0.9)
        fig.suptitle(f"Weighted Network Comparison ({year})", fontsize=14, y=1.01)
        fig.tight_layout()

        path = PLOTS_DIR / f"network_weighted_comparison_{year}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        output_paths.append(path)

    return output_paths


# ── Plot 3: Centrality bar charts (all three metrics) ─────────────────────────

def _plot_centrality_bars(centrality_df, metric, label, color, filename_prefix):
    """
    Reusable helper: draws top-10 horizontal bar charts for any centrality metric.
    Bars are coloured by the item's category.
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for year in sorted(centrality_df["year"].unique()):
        year_df = centrality_df[centrality_df["year"] == year]
        top = year_df.nlargest(10, metric)
        if top.empty:
            continue

        item_labels = [str(r.item_name)[:35] for r in top.itertuples()]
        values      = top[metric].tolist()
        bar_colors  = [CATEGORY_COLORS.get(r.category, "#7f7f7f") for r in top.itertuples()]

        fig, ax = plt.subplots(figsize=(12, 7))
        bars = ax.barh(item_labels, values, color=bar_colors, edgecolor="white")
        ax.invert_yaxis()
        ax.set_xlabel(label)
        ax.set_title(f"Top 10 Items by {label} ({year})")
        ax.grid(axis="x", alpha=0.3)

        # Show the numeric value at the end of each bar
        for bar, val in zip(bars, values):
            ax.text(val + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", fontsize=9)

        # Category colour legend
        legend_handles = [
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=CATEGORY_COLORS.get(cat, "#7f7f7f"),
                       label=cat, markersize=9)
            for cat in CATEGORIES if cat in CATEGORY_COLORS
        ]
        ax.legend(handles=legend_handles, title="Category",
                  loc="lower right", fontsize=7, framealpha=0.9)
        fig.tight_layout()

        path = PLOTS_DIR / f"{filename_prefix}_{year}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        output_paths.append(path)

    return output_paths


def plot_top_degree_bars(centrality_df):
    """Top-10 items by degree centrality for each year."""
    return _plot_centrality_bars(
        centrality_df, "degree_centrality",
        "Degree Centrality", "#1f77b4", "top10_degree"
    )


def plot_top_closeness_bars(centrality_df):
    """Top-10 items by closeness centrality for each year."""
    return _plot_centrality_bars(
        centrality_df, "closeness_centrality",
        "Closeness Centrality", "#2ca02c", "top10_closeness"
    )


def plot_top_betweenness_bars(centrality_df):
    """Top-10 items by betweenness centrality for each year."""
    return _plot_centrality_bars(
        centrality_df, "betweenness_centrality",
        "Betweenness Centrality", "#d62728", "top10_betweenness"
    )


# ── Plot 4: Temporal edge dynamics ────────────────────────────────────────────

def plot_temporal_edge_overlap(graphs):
    """
    Stacked bar chart showing how edges change between consecutive years:
      - Green: edges that persisted (exist in both years)
      - Red:   edges that disappeared
      - Blue:  new edges that appeared
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted(graphs.keys())
    transitions = []

    for i in range(len(years) - 1):
        y0 = years[i]
        y1 = years[i + 1]

        # Build edge sets as sorted tuples so (A,B) == (B,A)
        edges_y0 = set(tuple(sorted([u, v])) for u, v in graphs[y0]["unweighted"].edges())
        edges_y1 = set(tuple(sorted([u, v])) for u, v in graphs[y1]["unweighted"].edges())

        transitions.append({
            "label":       f"{y0} to {y1}",
            "persistent":  len(edges_y0 & edges_y1),
            "disappeared": len(edges_y0 - edges_y1),
            "appeared":    len(edges_y1 - edges_y0),
        })

    labels      = [t["label"]       for t in transitions]
    persistent  = [t["persistent"]  for t in transitions]
    disappeared = [t["disappeared"] for t in transitions]
    appeared    = [t["appeared"]    for t in transitions]

    x     = np.arange(len(labels))
    width = 0.5

    fig, ax = plt.subplots(figsize=(10, 7))
    b1 = ax.bar(x, persistent,  width, label="Persistent edges",   color="#2ca02c", edgecolor="white")
    b2 = ax.bar(x, disappeared, width, bottom=persistent,
                label="Disappeared edges", color="#d62728", edgecolor="white")
    b3 = ax.bar(x, appeared, width,
                bottom=[p + d for p, d in zip(persistent, disappeared)],
                label="New edges",         color="#1f77b4", edgecolor="white")

    # Put the count number inside each bar segment
    for bar, val in zip(b1, persistent):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, val / 2,
                    str(val), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
    for bar, base, val in zip(b2, persistent, disappeared):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base + val / 2,
                    str(val), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")
    for bar, base, d, val in zip(b3, persistent, disappeared, appeared):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, base + d + val / 2,
                    str(val), ha="center", va="center",
                    fontsize=11, color="white", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Number of Edges")
    ax.set_title("Temporal Edge Dynamics Across Yearly Networks")
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = PLOTS_DIR / "temporal_edge_overlap.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


# ── Plot 5: Category connectivity ─────────────────────────────────────────────

def plot_category_connectivity(graphs):
    """
    Two-panel chart:
      Left:  intra vs inter category edge counts per year (grouped bars)
      Right: top-5 most connected category pairs per year (horizontal bars)
    """
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    years = sorted(graphs.keys())
    intra_counts = []
    inter_counts = []
    top_pairs_per_year = {}

    for year in years:
        g = graphs[year]["unweighted"]
        intra = 0
        inter = 0
        pair_counts = Counter()

        for item_a, item_b in g.edges():
            cat_a = g.nodes[item_a].get("category", "Unknown")
            cat_b = g.nodes[item_b].get("category", "Unknown")
            pair_key = tuple(sorted([cat_a, cat_b]))
            pair_counts[pair_key] += 1
            if cat_a == cat_b:
                intra += 1
            else:
                inter += 1

        intra_counts.append(intra)
        inter_counts.append(inter)
        top_pairs_per_year[year] = pair_counts.most_common(5)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left panel: grouped bar chart
    x     = np.arange(len(years))
    width = 0.35
    axes[0].bar(x - width / 2, intra_counts, width,
                label="Intra-category", color="#2ca02c", edgecolor="white")
    axes[0].bar(x + width / 2, inter_counts, width,
                label="Inter-category", color="#d62728", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(y) for y in years], fontsize=12)
    axes[0].set_ylabel("Number of Edges")
    axes[0].set_title("Intra- vs Inter-Category Edges per Year")
    axes[0].legend(fontsize=10)
    axes[0].grid(axis="y", alpha=0.3)
    for xi, (iv, xv) in enumerate(zip(intra_counts, inter_counts)):
        axes[0].text(xi - width / 2, iv + 2, str(iv), ha="center",
                     fontsize=10, fontweight="bold")
        axes[0].text(xi + width / 2, xv + 2, str(xv), ha="center",
                     fontsize=10, fontweight="bold")

    # Right panel: top-5 category pairs per year
    bar_h    = 0.25
    y_pos    = np.arange(5)
    yr_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for idx, year in enumerate(years):
        pairs  = top_pairs_per_year[year]
        counts = [c for _, c in pairs]
        axes[1].barh(y_pos - idx * bar_h, counts, bar_h,
                     label=str(year), color=yr_colors[idx], edgecolor="white")

    ref_labels = [
        f"{a.split()[0]} <-> {b.split()[0]}"
        for (a, b), _ in top_pairs_per_year[years[-1]]
    ]
    axes[1].set_yticks(y_pos - bar_h)
    axes[1].set_yticklabels(ref_labels, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Edge Count")
    axes[1].set_title("Top 5 Category Pairs by Edge Count")
    axes[1].legend(fontsize=10)
    axes[1].grid(axis="x", alpha=0.3)

    fig.suptitle("Category Connectivity Analysis", fontsize=14, fontweight="bold")
    fig.tight_layout()

    path = PLOTS_DIR / "category_connectivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


# ── Plot 6: Sensitivity lines ─────────────────────────────────────────────────

def plot_sensitivity_lines(sensitivity_df):
    """Line charts showing how edge count changes as tau or K is varied."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    output_paths = []

    uw = sensitivity_df[sensitivity_df["weight_scheme"] == "unweighted"]

    # Edges vs tau
    fig, ax = plt.subplots(figsize=(10, 6))
    for year in sorted(uw["year"].unique()):
        data = uw[uw["year"] == year].groupby("tau")["edges"].mean()
        ax.plot(data.index, data.values, marker="o", label=str(year))
    ax.set_xlabel("Similarity Threshold (tau)")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Sensitivity: Edge Count vs Tau")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "sensitivity_edges_vs_tau.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    # Edges vs K
    fig, ax = plt.subplots(figsize=(10, 6))
    for year in sorted(uw["year"].unique()):
        data = uw[uw["year"] == year].groupby("k")["edges"].mean()
        ax.plot(data.index, data.values, marker="o", label=str(year))
    ax.set_xlabel("City Count Threshold (K)")
    ax.set_ylabel("Number of Edges")
    ax.set_title("Sensitivity: Edge Count vs K")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = PLOTS_DIR / "sensitivity_edges_vs_k.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    output_paths.append(path)

    return output_paths


# ── Entry point ───────────────────────────────────────────────────────────────

def run_visualizations(centrality_df=None, sensitivity_df=None):
    """Generate all plots and return a list of file paths."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    graphs = load_graphs()

    if centrality_df is None:
        centrality_df = pd.read_csv(OUTPUT_DIR / "centrality_results.csv")
    if sensitivity_df is None:
        sensitivity_df = pd.read_csv(OUTPUT_DIR / "sensitivity_analysis.csv")

    paths = []
    paths.extend(plot_network_by_year(graphs))
    paths.extend(plot_weighted_comparison(graphs))
    paths.extend(plot_top_degree_bars(centrality_df))
    paths.extend(plot_top_closeness_bars(centrality_df))
    paths.extend(plot_top_betweenness_bars(centrality_df))
    paths.extend(plot_temporal_edge_overlap(graphs))
    paths.extend(plot_category_connectivity(graphs))
    paths.extend(plot_sensitivity_lines(sensitivity_df))
    return paths