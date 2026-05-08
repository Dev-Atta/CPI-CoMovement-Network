# graph_builder.py
# Build item co-movement graphs for each year.
#
# Two items get an edge between them if their price-change patterns are
# similar (cosine similarity >= tau) in at least MIN_CITIES cities.
#
# This is the core of the project: we are modelling a RELATION between items
# using a graph, where an edge means "these two items tend to move together in price."

import networkx as nx
import pickle
from pathlib import Path
from config import OUTPUT_DIR, PROJECT_YEARS, SIMILARITY_THRESHOLD, MIN_CITIES


def build_graph_from_sim(city_map, tau, k, scheme):
    """
    Build a single graph from a city->similarity_matrix mapping.

    Steps:
      1. For each item pair, count how many cities have similarity >= tau.
         This count is called N(i, j) in the project spec.
    2. Add an edge between item i and item j if N(i, j) >= MIN_CITIES.
      3. For weighted graphs, store the weight on the edge.

    scheme can be:
      "unweighted"              - just edges, no weights
      "weighted_city_count"     - weight = number of cities where sim >= tau
      "weighted_avg_similarity" - weight = average similarity across all cities
    """
    if not city_map:
        return nx.Graph()

    # Get the list of all items (same across all cities)
    items = sorted(city_map[next(iter(city_map))].index)
    cities = list(city_map.keys())
    n = len(items)

    g = nx.Graph()
    g.add_nodes_from(items)

    # Check every pair of items (we only need to check each pair once)
    for i in range(n):
        for j in range(i + 1, n):
            item_i = items[i]
            item_j = items[j]

            # Count how many cities show similarity >= tau for this pair
            city_count = 0
            total_sim  = 0.0

            for city in cities:
                sim_value = city_map[city].loc[item_i, item_j]
                total_sim += sim_value
                if sim_value >= tau:
                    city_count += 1

            # Only add an edge if enough cities agree
            if city_count >= k:
                avg_sim = total_sim / len(cities)

                if scheme == "unweighted":
                    g.add_edge(item_i, item_j)
                elif scheme == "weighted_city_count":
                    g.add_edge(item_i, item_j, weight=float(city_count))
                elif scheme == "weighted_avg_similarity":
                    g.add_edge(item_i, item_j, weight=round(avg_sim, 4))

    return g


def build_graphs(df, sim_matrices, tau=SIMILARITY_THRESHOLD, k=MIN_CITIES):
    """
    Build three graph variants for each year:
      - unweighted:              edge exists or not
      - weighted_city_count:     edge weight = how many cities agreed
      - weighted_avg_similarity: edge weight = average cosine similarity

    Also attaches each item's category as a node attribute,
    which is used later for category analysis and graph colouring.
    """
    # Build a lookup from item name to its category
    item_category = {}
    for _, row in df[["item_name", "category"]].drop_duplicates().iterrows():
        item_category[row["item_name"]] = row["category"]

    graphs = {}

    for year in PROJECT_YEARS:
        city_map = sim_matrices.get(year, {})
        if not city_map:
            continue

        # Build all three variants using the shared helper above
        g_un = build_graph_from_sim(city_map, tau, k, "unweighted")
        g_cc = build_graph_from_sim(city_map, tau, k, "weighted_city_count")
        g_as = build_graph_from_sim(city_map, tau, k, "weighted_avg_similarity")

        # Attach category to every node in all three graphs
        for g in [g_un, g_cc, g_as]:
            for node in g.nodes():
                g.nodes[node]["category"] = item_category.get(node, "Unknown")

        graphs[year] = {
            "unweighted":              g_un,
            "weighted_city_count":     g_cc,
            "weighted_avg_similarity": g_as,
        }

        n_edges = g_un.number_of_edges()
        density = round(nx.density(g_un), 4)
        print(f"  {year}: {g_un.number_of_nodes()} nodes, {n_edges} edges, density={density}")

    return graphs


def save_graphs(graphs, output_dir=OUTPUT_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "graphs.pkl"
    with open(path, "wb") as f:
        pickle.dump(graphs, f)
    return path


def load_graphs(path=None):
    if path is None:
        path = OUTPUT_DIR / "graphs.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)