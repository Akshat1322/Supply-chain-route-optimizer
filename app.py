import streamlit as st
import pickle
import folium
from streamlit_folium import st_folium
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

st.set_page_config(
    page_title="Supply Chain Route Optimizer",
    page_icon="🚚",
    layout="wide"
)

# ── LOAD RESULTS ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_results():
    with open("results/route_results.pkl", "rb") as f:
        return pickle.load(f)

try:
    r = load_results()
except FileNotFoundError:
    st.error("Results not found. Please run `python optimize.py` first.")
    st.stop()

nodes       = r["nodes"]
labels      = r["stop_labels"]
naive_route = r["naive_route"]
naive_dist  = r["naive_dist"]
ga_route    = r["ga_route"]
ga_dist     = r["ga_dist"]
ga_red      = r["ga_reduction"]
sa_route    = r["sa_route"]
sa_dist     = r["sa_dist"]
sa_red      = r["sa_reduction"]
best_label  = r["best_label"]
best_dist   = r["best_dist"]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.title("🚚 Supply Chain Route Optimizer")
st.caption("Genetic Algorithm (DEAP) vs Simulated Annealing | 50-node VRP")
st.markdown("---")

# ── METRIC CARDS ─────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Naive Route",      f"{naive_dist:.1f} km")
c2.metric("GA Optimized",     f"{ga_dist:.1f} km",   f"-{ga_red:.1f}%")
c3.metric("SA Optimized",     f"{sa_dist:.1f} km",   f"-{sa_red:.1f}%")
c4.metric("Best Algorithm",   best_label,             f"{best_dist:.1f} km")

st.markdown("---")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🗺️ Route Maps", "📊 Algorithm Comparison", "📋 Route Details"])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — MAPS
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    col_l, col_r = st.columns(2)

    def build_map(route, color, title):
        depot = nodes[0]
        m = folium.Map(location=depot, zoom_start=12, tiles="CartoDB positron")

        # Depot marker
        folium.Marker(
            depot, popup="🏭 Depot",
            icon=folium.Icon(color="red", icon="home", prefix="fa")
        ).add_to(m)

        # Stop markers
        for idx in route:
            folium.CircleMarker(
                location=nodes[idx],
                radius=6, color=color, fill=True, fill_opacity=0.8,
                popup=labels[idx]
            ).add_to(m)

        # Route lines: depot → stops → depot
        full_route = [0] + route + [0]
        coords = [nodes[i] for i in full_route]
        folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8).add_to(m)
        return m

    with col_l:
        st.markdown("**Naive route (sequential)**")
        naive_map = build_map(naive_route, "#e74c3c", "Naive")
        st_folium(naive_map, width=480, height=400, key="naive_map")

    with col_r:
        st.markdown(f"**Optimized route ({best_label})**")
        opt_route = ga_route if best_label == "GA" else sa_route
        opt_map   = build_map(opt_route, "#27ae60", "Optimized")
        st_folium(opt_map, width=480, height=400, key="opt_map")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — COMPARISON
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Algorithm Comparison")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Distance comparison (km)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        algos  = ["Naive", "Genetic Algorithm", "Simulated Annealing"]
        dists  = [naive_dist, ga_dist, sa_dist]
        colors = ["#e74c3c", "#3498db", "#f39c12"]
        bars   = ax.bar(algos, dists, color=colors, edgecolor="white", linewidth=0.5)
        for bar, d in zip(bars, dists):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f"{d:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_ylabel("Total Distance (km)")
        ax.set_ylim(0, max(dists) * 1.15)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_b:
        st.markdown("**Distance reduction vs naive (%)**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        reds   = [0, ga_red, sa_red]
        colors2 = ["#e74c3c", "#3498db", "#f39c12"]
        bars2  = ax2.bar(algos, reds, color=colors2, edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars2, reds):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                     f"{v:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax2.set_ylabel("Reduction (%)")
        ax2.set_ylim(0, max(reds) * 1.25 + 2)
        ax2.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    st.markdown("---")
    st.markdown("**Summary table**")
    summary = pd.DataFrame({
        "Algorithm":          ["Naive", "Genetic Algorithm (DEAP)", "Simulated Annealing"],
        "Distance (km)":      [f"{naive_dist:.2f}", f"{ga_dist:.2f}", f"{sa_dist:.2f}"],
        "Reduction vs Naive": ["—", f"{ga_red:.1f}%", f"{sa_red:.1f}%"],
        "Winner":             ["❌", "✅" if best_label == "GA" else "➖", "✅" if best_label == "SA" else "➖"]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — ROUTE DETAILS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Route Details")

    col_x, col_y = st.columns(2)

    def route_df(route):
        rows = []
        prev = 0
        for idx in route:
            rows.append({
                "Stop":      labels[idx],
                "Lat":       round(nodes[idx][0], 5),
                "Lon":       round(nodes[idx][1], 5),
                "Dist from prev (km)": round(r["dist_matrix"][prev][idx], 2)
            })
            prev = idx
        return pd.DataFrame(rows)

    with col_x:
        st.markdown("**GA optimized route**")
        st.dataframe(route_df(ga_route), use_container_width=True, height=400)

    with col_y:
        st.markdown("**SA optimized route**")
        st.dataframe(route_df(sa_route), use_container_width=True, height=400)

st.markdown("---")
st.caption("Built by Akshat Sharma · DEAP Genetic Algorithm + Simulated Annealing + Folium + Streamlit")