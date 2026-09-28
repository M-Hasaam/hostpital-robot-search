import heapq
import math
import matplotlib.pyplot as plt
import networkx as nx
import streamlit as st

# Set Streamlit Page Config
st.set_page_config(
    page_title="Hospital Emergency Robot Search", layout="wide", page_icon="🏥"
)

# ---------------------------------------------------------
# 1. TASK 3 GRAPH & LOCATIONS SETUP (Hospital)
# ---------------------------------------------------------
locations = {
    "Pharmacy": (0, 0),
    "Main_Corridor": (2, 1),
    "Patient_Wing": (1, 4),
    "Nursing_Station": (4, 2),
    "Laboratory": (5, 5),
    "Emergency_Ward": (8, 6),
}

hospital_graph = {
    "Pharmacy": {"Main_Corridor": 2.2, "Patient_Wing": 4.1},
    "Main_Corridor": {"Nursing_Station": 2.2},
    "Patient_Wing": {"Laboratory": 5.0},
    "Nursing_Station": {"Laboratory": 3.2, "Emergency_Ward": 6.0},
    "Laboratory": {"Emergency_Ward": 3.2},
    "Emergency_Ward": {},
}


# ---------------------------------------------------------
# 2. HEURISTIC FUNCTION (Euclidean Distance)
# ---------------------------------------------------------
def heuristic(node, goal):
    x1, y1 = locations[node]
    x2, y2 = locations[goal]
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


# ---------------------------------------------------------
# 3. SEARCH ALGORITHMS
# ---------------------------------------------------------
def greedy_best_first_search(start, goal, graph):
    counter = 0
    pq = [(heuristic(start, goal), counter, start, [start])]
    visited = set()
    expansion_order = []

    while pq:
        _, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            # Calculate actual path cost
            total_cost = sum(
                graph[path[i]][path[i + 1]] for i in range(len(path) - 1)
            )
            return path, total_cost, expansion_order

        for neighbor, _ in graph[current].items():
            if neighbor not in visited:
                counter += 1
                h_val = heuristic(neighbor, goal)
                heapq.heappush(pq, (h_val, counter, neighbor, path + [neighbor]))

    return [], float("inf"), expansion_order


def a_star_search(start, goal, graph):
    counter = 0
    pq = [(heuristic(start, goal), counter, start, [start])]
    g_cost = {node: float("inf") for node in locations}
    g_cost[start] = 0

    visited = set()
    expansion_order = []

    while pq:
        _, _, current, path = heapq.heappop(pq)

        if current in visited:
            continue

        visited.add(current)
        expansion_order.append(current)

        if current == goal:
            return path, g_cost[goal], expansion_order

        for neighbor, edge_cost in graph[current].items():
            tentative_g = g_cost[current] + edge_cost
            if tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f_cost = tentative_g + heuristic(neighbor, goal)
                counter += 1
                heapq.heappush(
                    pq, (f_cost, counter, neighbor, path + [neighbor])
                )

    return [], float("inf"), expansion_order


# ---------------------------------------------------------
# 4. STREAMLIT UI & SIDEBAR CONTROLS
# ---------------------------------------------------------
st.title("🏥 Emergency Supply Robot: Search Visualizer")
st.markdown(
    "Compare **Greedy Best-First Search (GBFS)** and **A\* Search** on the hospital corridor layout."
)

st.sidebar.header("Navigation Settings")

# Dropdown menu selections
node_list = list(locations.keys())
start_node = st.sidebar.selectbox("Select Initial Node", node_list, index=0)
goal_node = st.sidebar.selectbox(
    "Select Goal Node", node_list, index=len(node_list) - 1
)
algorithm = st.sidebar.selectbox(
    "Select Search Algorithm", ["Greedy Best-First Search (GBFS)", "A* Search"]
)

# ---------------------------------------------------------
# 5. RUN SEARCH & DRAW GRAPH
# ---------------------------------------------------------
if start_node == goal_node:
    st.warning("⚠️ Start and Goal nodes are the same location!")
else:
    # Execute algorithm
    if "GBFS" in algorithm:
        path, total_cost, expansion_order = greedy_best_first_search(
            start_node, goal_node, hospital_graph
        )
    else:
        path, total_cost, expansion_order = a_star_search(
            start_node, goal_node, hospital_graph
        )

    # Create NetworkX DiGraph
    G = nx.DiGraph()
    for node, position in locations.items():
        G.add_node(node, pos=position)

    for node, neighbors in hospital_graph.items():
        for neighbor, cost in neighbors.items():
            G.add_edge(node, neighbor, weight=cost)

    pos = locations

    # Plot Matplotlib Figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Identify solution path edges
    path_edges = (
        list(zip(path[:-1], path[1:])) if len(path) > 1 else []
    )
    other_edges = [e for e in G.edges() if e not in path_edges]

    # Draw nodes and normal edges
    nx.draw_networkx_nodes(
        G, pos, node_color="skyblue", node_size=1400, ax=ax
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=other_edges,
        edge_color="gray",
        arrows=True,
        arrowsize=18,
        ax=ax,
    )

    # Highlight solution path in red
    if path_edges:
        nx.draw_networkx_edges(
            G,
            pos,
            edgelist=path_edges,
            edge_color="red",
            width=3.0,
            arrows=True,
            arrowsize=20,
            ax=ax,
        )

    # Highlight Start (Green) and Goal (Gold)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[start_node],
        node_color="lightgreen",
        node_size=1500,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=[goal_node],
        node_color="gold",
        node_size=1500,
        ax=ax,
    )

    # Labels
    nx.draw_networkx_labels(
        G, pos, font_size=9, font_weight="bold", ax=ax
    )
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, font_size=8, ax=ax
    )

    plt.axis("off")
    st.pyplot(fig)

    # ---------------------------------------------------------
    # 6. DISPLAY METRICS & PATH
    # ---------------------------------------------------------
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Algorithm", algorithm.split(" ")[0])
    col2.metric(
        "Total Path Cost",
        f"{total_cost:.2f}" if path else "N/A",
    )
    col3.metric("Nodes Expanded", len(expansion_order))

    if path:
        st.success(f"**Solution Path:** {' → '.join(path)}")
        st.info(f"**Expansion Order:** {' → '.join(expansion_order)}")
    else:
        st.error("No valid route exists between the selected locations.")