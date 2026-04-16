import numpy as np
import random
import pickle
import os
from deap import base, creator, tools, algorithms

random.seed(42)
np.random.seed(42)

# ── 1. GENERATE NODES ────────────────────────────────────────────────────────
N_STOPS   = 50
DEPOT_IDX = 0

lats = np.random.uniform(28.50, 28.75, N_STOPS + 1)
lons = np.random.uniform(77.05, 77.35, N_STOPS + 1)
lats[0], lons[0] = 28.6139, 77.2090   # depot = central Delhi

nodes       = list(zip(lats, lons))
stop_labels = ["Depot"] + [f"Stop {i}" for i in range(1, N_STOPS + 1)]

def haversine(a, b):
    R = 6371
    la1, lo1 = np.radians(a)
    la2, lo2 = np.radians(b)
    dlat, dlon = la2 - la1, lo2 - lo1
    h = np.sin(dlat/2)**2 + np.cos(la1)*np.cos(la2)*np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(h))

n = len(nodes)
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i][j] = haversine(nodes[i], nodes[j])

# ── DEAP works cleanly with 0-based indices ───────────────────────────────────
# stops_0based = [0..49] maps to actual node indices [1..50]
stops_0based  = list(range(N_STOPS))        # 0 to 49
actual_nodes  = list(range(1, N_STOPS + 1)) # 1 to 50

def route_distance(route_0based):
    """route_0based: list of ints 0..49 → actual node indices 1..50"""
    route = [actual_nodes[i] for i in route_0based]
    total = dist_matrix[DEPOT_IDX][route[0]]
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i+1]]
    total += dist_matrix[route[-1]][DEPOT_IDX]
    return total

def route_distance_actual(route):
    """route: actual node indices (1-based)"""
    total = dist_matrix[DEPOT_IDX][route[0]]
    for i in range(len(route) - 1):
        total += dist_matrix[route[i]][route[i+1]]
    total += dist_matrix[route[-1]][DEPOT_IDX]
    return total

# ── 2. NAIVE ROUTE ────────────────────────────────────────────────────────────
naive_route    = actual_nodes[:]
naive_distance = route_distance_actual(naive_route)
print(f"Naive route distance:     {naive_distance:.2f} km")

# ── 3. GENETIC ALGORITHM ─────────────────────────────────────────────────────
print("\nRunning Genetic Algorithm (DEAP)...")

for cls in ["FitnessMin", "Individual"]:
    if hasattr(creator, cls):
        delattr(creator, cls)

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("indices",    random.sample, stops_0based, len(stops_0based))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat,  list, toolbox.individual)

def eval_route(ind):
    return (route_distance(ind),)

toolbox.register("evaluate", eval_route)
toolbox.register("mate",     tools.cxPartialyMatched)   # PMX — works with any index range
toolbox.register("mutate",   tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select",   tools.selTournament, tournsize=3)

pop  = toolbox.population(n=300)
hof  = tools.HallOfFame(1)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)
stats.register("avg", np.mean)

pop, log = algorithms.eaSimple(
    pop, toolbox,
    cxpb=0.8, mutpb=0.2,
    ngen=300, stats=stats,
    halloffame=hof, verbose=False
)

ga_route_0    = list(hof[0])
ga_route      = [actual_nodes[i] for i in ga_route_0]
ga_distance   = route_distance_actual(ga_route)
ga_reduction  = (naive_distance - ga_distance) / naive_distance * 100
print(f"GA optimized distance:    {ga_distance:.2f} km  ({ga_reduction:.1f}% reduction)")

# ── 4. SIMULATED ANNEALING ────────────────────────────────────────────────────
print("\nRunning Simulated Annealing...")

def simulated_annealing(route, T=5000, cooling=0.995, n_iter=50000):
    current      = route[:]
    current_dist = route_distance_actual(current)
    best         = current[:]
    best_dist    = current_dist

    for _ in range(n_iter):
        i, j      = sorted(random.sample(range(len(current)), 2))
        neighbour = current[:]
        neighbour[i:j+1] = reversed(neighbour[i:j+1])
        n_dist    = route_distance_actual(neighbour)
        delta     = n_dist - current_dist

        if delta < 0 or random.random() < np.exp(-delta / T):
            current      = neighbour
            current_dist = n_dist
            if current_dist < best_dist:
                best      = current[:]
                best_dist = current_dist
        T *= cooling

    return best, best_dist

sa_route, sa_distance = simulated_annealing(actual_nodes[:])
sa_reduction = (naive_distance - sa_distance) / naive_distance * 100
print(f"SA optimized distance:    {sa_distance:.2f} km  ({sa_reduction:.1f}% reduction)")

# ── 5. SUMMARY ────────────────────────────────────────────────────────────────
best_label = "GA" if ga_distance <= sa_distance else "SA"
best_dist  = min(ga_distance, sa_distance)
best_route = ga_route if best_label == "GA" else sa_route

print(f"\n{'─'*45}")
print(f"Naive:  {naive_distance:.2f} km")
print(f"GA:     {ga_distance:.2f} km  (-{ga_reduction:.1f}%)")
print(f"SA:     {sa_distance:.2f} km  (-{sa_reduction:.1f}%)")
print(f"Winner: {best_label} → {best_dist:.2f} km")
print(f"{'─'*45}")

# ── 6. SAVE ───────────────────────────────────────────────────────────────────
os.makedirs("results", exist_ok=True)

results = {
    "nodes":         nodes,
    "stop_labels":   stop_labels,
    "dist_matrix":   dist_matrix,
    "naive_route":   naive_route,
    "naive_dist":    naive_distance,
    "ga_route":      ga_route,
    "ga_dist":       ga_distance,
    "ga_reduction":  ga_reduction,
    "sa_route":      sa_route,
    "sa_dist":       sa_distance,
    "sa_reduction":  sa_reduction,
    "best_route":    best_route,
    "best_dist":     best_dist,
    "best_label":    best_label,
}

with open("results/route_results.pkl", "wb") as f:
    pickle.dump(results, f)

print("\nSaved → results/route_results.pkl")
print("Run:   uv run streamlit run app.py")