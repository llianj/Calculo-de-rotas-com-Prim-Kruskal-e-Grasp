import streamlit as st
from streamlit_folium import st_folium
import folium
import numpy as np
import pandas as pd
from geopy.distance import geodesic
import time
import math
import random
import itertools
from collections import defaultdict
import networkx as nx
import osmnx as ox

def randomized_prim(ids, idx, D, rcl_size=3, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(ids)
    in_mst = set()
    edges = []
    start = 0
    in_mst.add(start)
    frontier = []
    for j in range(1, n):
        frontier.append((D[start, j], start, j))
    while len(in_mst) < n:
        if not frontier:
            break
        frontier.sort(key=lambda e: e[0])
        take = frontier[:rcl_size] if rcl_size > 0 else frontier
        if not take:
            break
        w, u, v = random.choice(take)
        if v in in_mst:
            frontier = [e for e in frontier if e[2] != v]
            continue
        edges.append((u, v, w))
        in_mst.add(v)
        frontier = [e for e in frontier if e[2] not in in_mst]
        for j in range(n):
            if j not in in_mst:
                frontier.append((D[v, j], v, j))
    return edges

def randomized_kruskal(ids, idx, D, rcl_size=50, seed=None):
    if seed is not None:
        random.seed(seed)
    n = len(ids)
    edges_all = []
    for i in range(n):
        for j in range(i + 1, n):
            edges_all.append((D[i, j], i, j))
    edges_all.sort(key=lambda e: e[0])
    parent = list(range(n))
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    mst = []
    pos = 0
    while len(mst) < n - 1 and pos < len(edges_all):
        rcl = edges_all[pos:pos + rcl_size] if rcl_size > 0 else edges_all[pos:]
        if not rcl:
            break
        picked = None
        candidates = rcl[:]
        random.shuffle(candidates)
        for w, u, v in candidates:
            if find(u) != find(v):
                picked = (w, u, v)
                break
            else:
                try:
                    edges_all.remove((w, u, v))
                except ValueError:
                    pass
        if picked is None:
            while pos < len(edges_all):
                w, u, v = edges_all[pos]
                pos += 1
                if find(u) != find(v):
                    picked = (w, u, v)
                    break
            if picked is None:
                break
        if picked:
            w, u, v = picked
            ru, rv = find(u), find(v)
            parent[ru] = rv
            mst.append((u, v, w))
            try:
                edges_all.remove(picked)
            except ValueError:
                pass
        pos += 1
    return mst

def split_mst_edges_to_routes(mst_edges, n_nodes, k):
    if k <= 1 or not mst_edges:
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_weighted_edges_from(mst_edges)
        comps = list(nx.connected_components(G))
        return [list(c) for c in comps]
    edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)
    remove_set = set()
    for i in range(min(k - 1, len(edges_sorted))):
        u, v, _ = edges_sorted[i]
        remove_set.add(tuple(sorted((u, v))))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u, v, w in mst_edges:
        if tuple(sorted((u, v))) not in remove_set:
            G.add_edge(u, v, weight=w)
    comps = list(nx.connected_components(G))
    return [list(c) for c in comps]

def route_distance(route, D, depot_start=0, depot_end=None):
    if depot_end is None:
        seq = [depot_start] + route + [depot_start]
    else:
        seq = [depot_start] + route + [depot_end]
    if len(seq) <= 1:
        return 0.0
    dist = 0.0
    for i in range(len(seq) - 1):
        dist += D[seq[i], seq[i + 1]]
    return dist

def two_opt(route, D, depot_start=0, depot_end=None):
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(0, len(best) - 1):
            for j in range(i + 1, len(best)):
                new_route = best[:i] + best[i:j + 1][::-1] + best[j + 1:]
                old_dist = route_distance(best, D, depot_start, depot_end)
                new_dist = route_distance(new_route, D, depot_start, depot_end)
                if new_dist < old_dist:
                    best = new_route
                    improved = True
        route = best
    return best

def relocate_between_routes(routes, D, depot_start=0, depot_end=None):
    improved = True
    best_routes = [r[:] for r in routes]
    n_routes = len(routes)
    while improved:
        improved = False
        for a_idx in range(n_routes):
            for b_idx in range(n_routes):
                if a_idx == b_idx:
                    continue
                for node_idx_in_a, node in enumerate(best_routes[a_idx]):
                    orig_cost = (route_distance(best_routes[a_idx], D, depot_start, depot_end) +
                                 route_distance(best_routes[b_idx], D, depot_start, depot_end))
                    route_a_new = best_routes[a_idx][:node_idx_in_a] + best_routes[a_idx][node_idx_in_a + 1:]
                    for pos_in_b in range(len(best_routes[b_idx]) + 1):
                        route_b_new = best_routes[b_idx][:pos_in_b] + [node] + best_routes[b_idx][pos_in_b:]
                        new_cost = (route_distance(route_a_new, D, depot_start, depot_end) +
                                    route_distance(route_b_new, D, depot_start, depot_end))
                        if new_cost < orig_cost:
                            best_routes[a_idx] = route_a_new
                            best_routes[b_idx] = route_b_new
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes

def build_routes_from_mst_random(n_nodes, method, ids, idx, D, k_routes=3, rcl_size=3, seed=None):
    if method == 'prim':
        mst = randomized_prim(ids, idx, D, rcl_size=rcl_size, seed=seed)
    else:
        mst = randomized_kruskal(ids, idx, D, rcl_size=rcl_size, seed=seed)
    comps = split_mst_edges_to_routes(mst, n_nodes, k_routes)
    routes = [list(c) for c in comps]
    return routes, mst

def grasp(ids, idx, D, coords, k_routes=3, iterations=100, method='prim', rcl_size=3, seed=None, depot_start=0, depot_end=None):
    best_solution = None
    best_cost = float('inf')
    n = len(ids)
    if seed is None:
        seed = int(time.time())
    for it in range(iterations):
        s, _ = build_routes_from_mst_random(n, method, ids, idx, D, k_routes=k_routes, rcl_size=rcl_size, seed=seed + it)
        processed_routes = []
        for comp in s:
            route = [v for v in comp if v != depot_start and (depot_end is None or v != depot_end)]
            processed_routes.append(route)
        s2 = [two_opt(r, D, depot_start=depot_start, depot_end=depot_end) for r in processed_routes]
        s3 = relocate_between_routes(s2, D, depot_start=depot_start, depot_end=depot_end)
        total_cost = sum(route_distance(r, D, depot_start=depot_start, depot_end=depot_end) for r in s3)
        if total_cost < best_cost:
            best_cost = total_cost
            best_solution = s3
    return best_solution, best_cost

@st.cache_data
def load_graph(city_name):
    st.info(f"Baixando dados de ruas de {city_name}")
    G = ox.graph_from_place(city_name, network_type="drive")
    st.success(f"Mapa de ruas de {city_name} carregado!")
    return G

def haversine_m(a, b):
    return geodesic(a, b).meters

def haversine_np(lat1, lon1, lat2_arr, lon2_arr):
    R = 6371000.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2_arr)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def build_real_distance_matrix(graph, points_list):
    n = len(points_list)
    pids = [p[0] for p in points_list]
    coords = {p[0]: (p[1], p[2]) for p in points_list}
    idx_map = {pids[i]: i for i in range(n)}
    node_map = {}
    node_ids = np.array(list(graph.nodes()))
    node_lats = np.array([graph.nodes[n]['y'] for n in node_ids], dtype=float)
    node_lons = np.array([graph.nodes[n]['x'] for n in node_ids], dtype=float)
    for pid, lat, lon, label in points_list:
        dists = haversine_np(lat, lon, node_lats, node_lons)
        nearest_idx = int(np.argmin(dists))
        node = node_ids[nearest_idx]
        node_map[pid] = node
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            pid_i = pids[i]
            pid_j = pids[j]
            node_i = node_map[pid_i]
            node_j = node_map[pid_j]
            if node_i == node_j:
                d = 0
            else:
                try:
                    d = nx.shortest_path_length(graph, node_i, node_j, weight="length")
                except nx.NetworkXNoPath:
                    d = haversine_m(coords[pid_i], coords[pid_j])
            D[i, j] = D[j, i] = d
    return pids, idx_map, D, coords, node_map

st.set_page_config(layout="wide")
st.title("Planejador de Rotas de Entrega GRASP - Santarém (PA)")

SANTARÉM_CENTER = (-2.4381, -54.7000)

if "points" not in st.session_state:
    st.session_state.points = []
if "next_id" not in st.session_state:
    st.session_state.next_id = 0
if "start_id" not in st.session_state:
    st.session_state.start_id = None
if "end_id" not in st.session_state:
    st.session_state.end_id = None
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "last_draw_hash" not in st.session_state:
    st.session_state.last_draw_hash = None

col1, col2 = st.columns([3, 1.5])

with col1:
    st.header("1. Adicione os Pontos no Mapa")
    m_draw = folium.Map(location=SANTARÉM_CENTER, zoom_start=13, tiles="OpenStreetMap")
    map_data = st_folium(m_draw, height=600, width="100%", returned_objects=['last_clicked'])
    if map_data and map_data.get("last_clicked"):
        clicked_data = map_data["last_clicked"]
        lat, lon = clicked_data["lat"], clicked_data["lng"]
        click_hash = hash(f"{lat}-{lon}")
        if "last_click_hash" not in st.session_state or st.session_state.last_click_hash != click_hash:
            st.session_state.last_click_hash = click_hash
            pid = st.session_state.next_id
            st.session_state.next_id += 1
            label = f"Ponto {pid}"
            if len(st.session_state.points) == 0:
                st.session_state.start_id = pid
            st.session_state.points.append((pid, lat, lon, label))
            st.rerun()
    st.header("3. Visualize as Rotas Calculadas")
    m_viz = folium.Map(location=SANTARÉM_CENTER, zoom_start=13, tiles="cartodbdarkmatter")
    for pid, lat, lon, label in st.session_state.points:
        icon = 'truck'
        color = 'blue'
        if pid == st.session_state.start_id:
            icon = 'home'
            color = 'red'
        elif pid == st.session_state.end_id:
            icon = 'flag'
            color = 'green'
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>{label} (ID: {pid})</b>",
            tooltip=label,
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(m_viz)
    if st.session_state.last_result:
        G_santarem = load_graph("Santarém, Pará, Brazil")
        res = st.session_state.last_result
        routes = res["routes"]
        ids = res["ids"]
        idx_map = res["idx_map"]
        node_map = res["node_map"]
        depot_start_idx = idx_map[st.session_state.start_id]
        depot_start_node = node_map[st.session_state.start_id]
        depot_end_idx = idx_map.get(st.session_state.end_id, None)
        depot_end_node = node_map.get(st.session_state.end_id, None)
        route_colors = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#46F0F0", "#F032E6"]
        for i, route_indices in enumerate(routes):
            color = route_colors[i % len(route_colors)]
            if depot_end_idx is None:
                seq_indices = [depot_start_idx] + route_indices + [depot_start_idx]
            else:
                seq_indices = [depot_start_idx] + route_indices + [depot_end_idx]
            seq_nodes = [node_map[ids[idx]] for idx in seq_indices]
            for j in range(len(seq_nodes) - 1):
                node_a = seq_nodes[j]
                node_b = seq_nodes[j+1]
                if node_a == node_b:
                    continue
                try:
                    path_nodes = nx.shortest_path(G_santarem, node_a, node_b, weight="length")
                    path_coords = [(G_santarem.nodes[n]['y'], G_santarem.nodes[n]['x']) for n in path_nodes]
                    folium.PolyLine(locations=path_coords, color='black', weight=8, opacity=0.6).add_to(m_viz)
                    folium.PolyLine(locations=path_coords, color=color, weight=5, opacity=0.8).add_to(m_viz)
                except nx.NetworkXNoPath:
                    pass
    st_folium(m_viz, height=600, width="100%")

with col2:
    st.header("2. Gerencie os Pontos")
    if not st.session_state.points:
        st.info("Coloque o primeiro ponto (Inicio) usando o mapa.")
    else:
        df = pd.DataFrame(st.session_state.points, columns=["id", "lat", "lon", "label"])
        for i, row in df.iterrows():
            pid = row['id']
            with st.expander(f"Ponto {pid}: {row['label']}", expanded=pid == st.session_state.next_id - 1):
                new_label = st.text_input("Nome:", value=row['label'], key=f"label_{pid}")
                if new_label != row['label']:
                    st.session_state.points[i] = (pid, row['lat'], row['lon'], new_label)
                    st.rerun()
                is_start = (pid == st.session_state.start_id)
                is_end = (pid == st.session_state.end_id)
                tipo = st.radio("Tipo:", ["Início", "Destino", "Fim"], index=0 if is_start else (2 if is_end else 1), key=f"tipo_{pid}", horizontal=True)
                if tipo == "Início" and not is_start:
                    st.session_state.start_id = pid
                    if is_end:
                        st.session_state.end_id = None
                    st.rerun()
                elif tipo == "Fim" and not is_end:
                    st.session_state.end_id = pid
                    if is_start:
                        st.session_state.start_id = None
                    st.rerun()
                elif tipo == "Destino" and (is_start or is_end):
                    if is_start:
                        st.session_state.start_id = None
                    if is_end:
                        st.session_state.end_id = None
                    st.rerun()
                if st.button("Apagar Ponto", key=f"del_{pid}", type="primary"):
                    st.session_state.points = [p for p in st.session_state.points if p[0] != pid]
                    if is_start:
                        st.session_state.start_id = None
                    if is_end:
                        st.session_state.end_id = None
                    st.rerun()
        if st.button("🧹 Limpar Todos os Pontos"):
            st.session_state.points = []
            st.session_state.start_id = None
            st.session_state.end_id = None
            st.session_state.next_id = 0
            st.session_state.last_result = None
            st.rerun()
    st.markdown("---")
    st.header("Plotagem")
    k_routes = st.number_input("Número de rotas (veículos):", min_value=1, max_value=1, value=1)
    iterations = st.number_input("Iterações do GRASP:", min_value=10, max_value=1000, value=100)
    metodo = st.selectbox("Métodos a serem usados:", ["prim", "kruskal"])
    rcl = st.slider("Tamanho RCL (Aleatoriedade):", 1, 50, 5)
    if st.button("Calcular Rotas", type="primary", use_container_width=True):
        if len(st.session_state.points) < 2:
            st.error("Adicione pelo menos 2 pontos.")
        elif st.session_state.start_id is None:
            st.error("Defina um ponto de 'Início'.")
        else:
            with st.spinner("Calculando... "):
                G_santarem = load_graph("Santarém, Pará, Brazil")
                pts_list = st.session_state.points
                ids, idx_map, D, coords, node_map = build_real_distance_matrix(G_santarem, pts_list)
                depot_start_idx = idx_map[st.session_state.start_id]
                depot_end_idx = idx_map.get(st.session_state.end_id, None)
                start_t = time.time()
                best_routes, best_cost = grasp(
                    ids, idx_map, D, coords,
                    k_routes=k_routes,
                    iterations=iterations,
                    method=metodo,
                    rcl_size=rcl,
                    seed=123,
                    depot_start=depot_start_idx,
                    depot_end=depot_end_idx,
                )
                t = time.time() - start_t
                st.session_state.last_result = {
                    "routes": best_routes,
                    "cost": best_cost,
                    "ids": ids,
                    "idx_map": idx_map,
                    "coords": coords,
                    "node_map": node_map
                }
                st.success(f"Rotas calculadas em {t:.2f}s")
                st.metric("Custo Total (Distância de Rua)", f"{best_cost / 1000:.2f} km")
                st.markdown("---")
                st.subheader("Distâncias por Rota")
                route_colors_hex = ["#E6194B", "#3CB44B", "#FFE119", "#4363D8", "#F58231", "#911EB4", "#46F0F0", "#F032E6"]
                color_names = ["Vermelha", "Verde", "Amarela", "Azul", "Laranja", "Roxa", "Ciano", "Magenta"]
                route_details = []
                min_dist = float('inf')
                best_route_idx = -1
                for i, route_indices in enumerate(best_routes):
                    dist = route_distance(route_indices, D, depot_start_idx, depot_end_idx)
                    route_details.append((i, dist))
                    if dist < min_dist:
                        min_dist = dist
                        best_route_idx = i
                for i, dist in route_details:
                    color_hex = route_colors_hex[i % len(route_colors_hex)]
                    color_name = color_names[i % len(color_names)]
                    color_block = f'<div style="width:15px; height:15px; background-color:{color_hex}; border-radius:3px; display:inline-block; margin-right:8px; border: 1px solid #555;"></div>'
                    st.markdown(f"{color_block} Rota {i} ({color_name}): {dist / 1000:.2f} km", unsafe_allow_html=True)
                if best_route_idx != -1:
                    best_color_name = color_names[best_route_idx % len(color_names)]
                    st.success(f"Rota Mais Rápida: Rota {best_route_idx} ({best_color_name}) com {min_dist / 1000:.2f} km")
