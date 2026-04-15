# Supply Chain Route Optimizer using Genetic Algorithms

Solves the Vehicle Routing Problem (VRP) for 50 delivery nodes using a custom Genetic Algorithm (DEAP), benchmarked against Simulated Annealing — deployed as an interactive Streamlit + Folium map dashboard.

## Features
- **28%+ route distance reduction** over naive baseline
- **Genetic Algorithm** (DEAP) with ordered crossover and shuffle mutation
- **Simulated Annealing** with 2-opt swap for benchmarking
- **Folium map** showing naive vs optimized routes side-by-side
- **Distance comparison charts** and full route detail tables

## Setup

```bash
git clone https://github.com/Akshat1322/Supply-chain-route-optimizer
cd Supply-chain-route-optimizer
uv add deap folium streamlit-folium streamlit numpy pandas matplotlib
```

## Run

```bash
# Step 1 — Optimize routes
uv run python optimize.py

# Step 2 — Launch dashboard
uv run streamlit run app.py
```

## Tech Stack
`Python` `DEAP` `Simulated Annealing` `Folium` `Streamlit` `NumPy` `pandas` `Matplotlib`

## Project Structure
```
supply-chain-optimizer/
├── optimize.py        ← GA + SA solver
├── app.py             ← Streamlit + Folium dashboard
├── requirements.txt
├── README.md
└── results/
    └── route_results.pkl
```

---
Built by [Akshat Sharma](https://github.com/Akshat1322)