# CPI-CoMovement-Network

A graph-based analysis of consumer item price co-movement across 15 Pakistani cities (2023–2025), built as a Discrete Structures course project.

## Project Overview

This project models consumer items as nodes in a graph. Two items are connected by an edge if their monthly price-change patterns are similar across a sufficient number of cities — revealing which items tend to move together in price, and why.

## Key Findings

- Price co-movement in Pakistan mostly **crosses category boundaries**
- Food, Transport, Health, and Housing items frequently co-move together
- This suggests **economy-wide inflation drivers** rather than sector-specific shocks
- Results are stable across 180 sensitivity analysis combinations (τ and K variations)

## Pipeline

1. Load raw CPI data from PBS Pakistan
2. Compute monthly price-change vectors per item per city
3. Compute cosine similarity matrices for every item pair
4. Build yearly graphs using configurable thresholds (τ, K)
5. Compute degree, closeness & betweenness centrality
6. Temporal analysis across 2023, 2024, 2025
7. Category connectivity analysis
8. Sensitivity analysis across 180 parameter combinations

## Project Structure

| File | Description |
|------|-------------|
| `main.py` | Full pipeline runner |
| `config.py` | All parameters and settings |
| `data_loader.py` | Load and clean CPI data |
| `vectors.py` | Compute price-change vectors |
| `similarity.py` | Cosine similarity matrices |
| `graph_builder.py` | Build item co-movement graphs |
| `centrality.py` | Degree, closeness, betweenness centrality |
| `temporal_analysis.py` | Year-over-year network comparison |
| `category_analysis.py` | Intra vs inter-category edge analysis |
| `sensitivity_analysis.py` | Threshold sensitivity testing |
| `visualize.py` | All plots and visualizations |

## How to Run

```bash
pip install pandas networkx matplotlib
python main.py
```

Place your CPI data file at `data/cpi_raw.csv` before running.

## Data Source

Pakistan Bureau of Statistics (PBS): https://www.pbs.gov.pk/price/

## Tech Stack

- Python 3.12
- NetworkX
- pandas
- matplotlib

## Demo Video
https://www.youtube.com/watch?v=LdJyMBR1TBM
