# sensitivity_analysis.py
# Test how sensitive the graph structure is to our choice of tau and MIN_CITIES.
#
# tau controls how similar two items must be in a city (higher = stricter).
# MIN_CITIES controls how many cities must agree (higher = stricter).
#
# By varying these, we can show that our chosen values (tau=0.75, MIN_CITIES=5)
# produce a meaningful, non-trivial graph that is not an accident of the thresholds.

import pandas as pd
from config import OUTPUT_DIR, PROJECT_YEARS, TAU_VALUES, K_VALUES, WEIGHT_SCHEMES
from graph_builder import build_graph_from_sim


def compute_sensitivity(sim_matrices):
    """
    For every combination of year, tau, MIN_CITIES, and weight scheme:
      - Build a graph
      - Record how many edges it has and how dense it is

    This lets us see how the graph changes as we tighten or loosen the thresholds.
    """
    rows = []

    for year in PROJECT_YEARS:
        city_map = sim_matrices.get(year, {})
        if not city_map:
            continue

        for tau in TAU_VALUES:
            for k in K_VALUES:
                for scheme in WEIGHT_SCHEMES:
                    # Build the graph for this combination of parameters
                    g = build_graph_from_sim(city_map, tau, k, scheme)

                    n_nodes = g.number_of_nodes()
                    n_edges = g.number_of_edges()

                    # Density = actual edges / maximum possible edges
                    if n_nodes > 1:
                        max_edges = n_nodes * (n_nodes - 1) / 2
                        density = round(n_edges / max_edges, 4)
                    else:
                        density = 0.0

                    rows.append({
                        "year":         year,
                        "tau":          tau,
                        "k":            k,
                        "weight_scheme": scheme,
                        "edges":        n_edges,
                        "density":      density,
                    })

    return pd.DataFrame(rows)


def save_sensitivity(df, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "sensitivity_analysis.csv"
    df.to_csv(path, index=False)
    return path