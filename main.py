# main.py
# Runs the full CPI Co-Movement Network project pipeline.
# Each step is clearly labelled. Steps run in order and pass results forward.

import pickle
from config import SIMILARITY_THRESHOLD, MIN_CITIES, OUTPUT_DIR

from data_loader          import load_data, save_master
from vectors              import compute_price_vectors
from similarity           import compute_similarity_matrices
from graph_builder        import build_graphs, save_graphs
from centrality           import compute_centrality, print_top_centrality, save_centrality
from temporal_analysis    import run_temporal_analysis
from category_analysis    import run_category_analysis
from sensitivity_analysis import compute_sensitivity, save_sensitivity
from visualize            import run_visualizations


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    #Step 1 Load the raw CPI data 
    print("\n[1] Loading CPI data...")
    df = load_data()
    save_master(df)
    print(f"    Loaded {len(df)} rows | {df['item_name'].nunique()} items | "
          f"{df['city'].nunique()} cities | years: {sorted(df['year'].unique().tolist())}")

    # Step 2 Compute price-change vectors
    # For each item in each city, build an 11-element vector of monthly % price changes
    print("\n[2] Computing price-change vectors...")
    vectors = compute_price_vectors(df)
    total = sum(
        len(item_map)
        for year_map in vectors.values()
        for item_map in year_map.values()
    )
    print(f"    Built {total} price-change vectors (years x cities x items)")

    # Step 3 Compute cosine similarity matrices 
    # For each city and year, measure how similar every pair of items is
    print("\n[3] Computing cosine similarity matrices...")
    sim_matrices = compute_similarity_matrices(vectors)

    # Save similarity matrices so sensitivity analysis can reload them without recomputing
    sim_path = OUTPUT_DIR / "similarity_matrices.pkl"
    with open(sim_path, "wb") as f:
        pickle.dump(sim_matrices, f)
    print(f"    Similarity matrices saved to {sim_path}")

    # Step 4 Build graphs 
    # Two items get an edge if their similarity >= SIMILARITY_THRESHOLD in at least MIN_CITIES cities
    print(f"\n[4] Building graphs (tau={SIMILARITY_THRESHOLD}, K={MIN_CITIES})...")
    graphs = build_graphs(df, sim_matrices)
    save_graphs(graphs)

    # Step 5 Compute centrality measures
    # Degree, closeness, and betweenness centrality for every item every year
    print("\n[5] Computing centrality measures...")
    centrality_df = compute_centrality(graphs)
    print_top_centrality(centrality_df, top_n=5)
    save_centrality(centrality_df)

    #Step 6 Temporal analysis
    # Track which edges appeared/disappeared and which items stay central over time
    print("\n[6] Running temporal analysis...")
    run_temporal_analysis(graphs=graphs, centrality_df=centrality_df)

    #Step 7 Category connectivity analysis
    # Count intra vs inter category edges — do items co-move within their own category?
    print("\n[7] Running category connectivity analysis...")
    run_category_analysis(graphs=graphs)

    #Step 8 Sensitivity analysis 
    # Vary tau and MIN_CITIES to show how the graph structure changes with the thresholds
    print("\n[8] Running sensitivity analysis...")
    with open(sim_path, "rb") as f:
        sim_matrices_reload = pickle.load(f)
    sensitivity_df = compute_sensitivity(sim_matrices_reload)
    save_sensitivity(sensitivity_df)
    print(f"    Tested {len(sensitivity_df)} parameter combinations")

    #Step 9 Generate all visualisations
    print("\n[9] Generating visualisations...")
    plot_paths = run_visualizations(
        centrality_df=centrality_df,
        sensitivity_df=sensitivity_df
    )
    print(f"    Saved {len(plot_paths)} plots to {OUTPUT_DIR / 'plots'}")

    print("\n Pipeline complete. All outputs saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()