import math, random, time, itertools
from collections import defaultdict
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# -------------------------
# utils: points, distances
# -------------------------
def generate_random_points(n, seed=40, bounds=(0,1000)):
    random.seed(seed)
    xs = [random.uniform(*bounds) for _ in range(n)]
    ys = [random.uniform(*bounds) for _ in range(n)]
    ids = list(range(n))
    return [(i, xs[i], ys[i]) for i in ids]

def load_points_csv(path):
    df = pd.read_csv(path)
    # Expect columns id,x,y or first three columns
    if {'id','x','y'}.issubset(df.columns):
        rows = [(int(r['id']), float(r['x']), float(r['y'])) for _,r in df.iterrows()]
    else:
        rows = [(int(r[0]), float(r[1]), float(r[2])) for r in df.values]
    return rows

def euclid(a,b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def build_distance_matrix(points):
    n = len(points)
    coords = {pid:(x,y) for pid,x,y in points}
    ids = [pid for pid,_,_ in points]
    idx = {pid:i for i,pid in enumerate(ids)}
    D = np.zeros((n,n))
    for i in range(n):
        for j in range(i+1,n):
            d = euclid(coords[ids[i]], coords[ids[j]])
            D[i,j] = D[j,i] = d
    return ids, idx, D, coords

# -------------------------
# Randomized MST constructors
# -------------------------
def randomized_prim(ids, idx, D, rcl_size=3, seed=None):
    """Prim but choose from RCL (rcl_size smallest candidate edges) randomly."""
    if seed is not None:
        random.seed(seed)
    n = len(ids)
    in_mst = set()
    edges = []
    start = 0
    in_mst.add(start)
    frontier = []
    for j in range(1,n):
        frontier.append((D[start,j], start, j))
    while len(in_mst) < n:
        frontier.sort(key=lambda e: e[0])
        take = frontier[:rcl_size] if rcl_size>0 else frontier
        w,u,v = random.choice(take)
        edges.append((u,v,w))
        in_mst.add(v)
        # remove edges leading to v and add new frontiers
        frontier = [e for e in frontier if e[2] not in in_mst]
        for j in range(n):
            if j not in in_mst:
                frontier.append((D[v,j], v, j))
    return edges

def randomized_kruskal(ids, idx, D, rcl_size=50, seed=None):
    """Kruskal but at each step choose randomly among the rcl_size smallest available edges."""
    if seed is not None:
        random.seed(seed)
    n = len(ids)
    edges_all = []
    for i in range(n):
        for j in range(i+1,n):
            edges_all.append((D[i,j], i, j))
    edges_all.sort(key=lambda e: e[0])
    parent = list(range(n))
    def find(a):
        while parent[a]!=a:
            parent[a]=parent[parent[a]]
            a=parent[a]
        return a
    mst = []
    pos = 0
    while len(mst) < n-1:
        # form RCL from remaining smallest edges
        rcl = edges_all[pos:pos+rcl_size] if rcl_size>0 else edges_all[pos:]
        # pick random from rcl that doesn't create cycle
        picked = None
        for attempt in range(len(rcl)):
            cand = random.choice(rcl)
            if find(cand[1]) != find(cand[2]):
                picked = cand
                break
            else:
                # remove that edge from edges_all to speed up
                try:
                    edges_all.remove(cand)
                except ValueError:
                    pass
        if picked is None:
            # fallback: scan sequentially
            while pos < len(edges_all):
                w,u,v = edges_all[pos]
                pos += 1
                if find(u)!=find(v):
                    picked=(w,u,v)
                    break
            if picked is None:
                break
        w,u,v = picked
        ru,rv = find(u), find(v)
        parent[ru]=rv
        mst.append((u,v,w))
        # ensure we don't pick same edge again
        try:
            edges_all.remove(picked)
        except ValueError:
            pass
    return mst

# -------------------------
# split MST into k routes
# -------------------------
def split_mst_edges_to_routes(mst_edges, n_nodes, k):
    """Given MST edges as (u,v,w) with nodes labeled 0..n-1, remove k-1 largest edges to create k components (routes)."""
    if k<=1:
        G = nx.Graph()
        G.add_nodes_from(range(n_nodes))
        G.add_weighted_edges_from([(u,v,w) for u,v,w in mst_edges])
        comps = list(nx.connected_components(G))
        return [list(c) for c in comps]
    # sort edges by weight descending and remove k-1 largest edges
    edges_sorted = sorted(mst_edges, key=lambda e: e[2], reverse=True)
    remove_set = set((edges_sorted[i][0], edges_sorted[i][1]) for i in range(min(k-1,len(edges_sorted))))
    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    for u,v,w in mst_edges:
        if (u,v) in remove_set or (v,u) in remove_set:
            continue
        G.add_edge(u,v,weight=w)
    comps = list(nx.connected_components(G))
    return [list(c) for c in comps]

# -------------------------
# route utilities: route distance, 2-opt, relocate
# -------------------------
def route_distance(route, D, depot_start=0, depot_end=None):
    """
    route: list of node indices (visits) - these are indices 0..n-1 excluding start/end
    depot_start: index of start node (0..n-1)
    depot_end: index of end node (0..n-1) - if None, assume return to start
    """
    if depot_end is None:
        # closed tour to start
        seq = [depot_start] + route + [depot_start]
    else:
        seq = [depot_start] + route + [depot_end]
    if len(seq) <= 1:
        return 0.0
    dist = 0.0
    for i in range(len(seq)-1):
        dist += D[seq[i], seq[i+1]]
    return dist

def two_opt(route, D, depot_start=0, depot_end=None):
    # 2-opt for list of visits only (start/end handled in route_distance)
    best = route[:]
    improved = True
    while improved:
        improved = False
        for i in range(0, len(best)-1):
            for j in range(i+1, len(best)):
                new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                if route_distance(new, D, depot_start, depot_end) < route_distance(best, D, depot_start, depot_end):
                    best = new
                    improved = True
        route = best
    return best

def relocate_between_routes(routes, D, depot_start=0, depot_end=None):
    # try moving a node from route A to route B if it improves total cost
    improved = True
    best_routes = [r[:] for r in routes]
    n_routes = len(routes)
    while improved:
        improved = False
        for a in range(n_routes):
            for b in range(n_routes):
                if a==b: continue
                for i, node in enumerate(best_routes[a]):
                    # try inserting node in every position in route b
                    origA = best_routes[a][:]
                    origB = best_routes[b][:]
                    del origA[i]
                    best_gain = 0
                    best_insert = None
                    for pos in range(len(origB)+1):
                        newB = origB[:pos] + [node] + origB[pos:]
                        old_cost = route_distance(best_routes[a], D, depot_start, depot_end) + route_distance(best_routes[b], D, depot_start, depot_end)
                        new_cost = route_distance(origA, D, depot_start, depot_end) + route_distance(newB, D, depot_start, depot_end)
                        if new_cost + 1e-6 < old_cost:
                            gain = old_cost - new_cost
                            if gain > best_gain:
                                best_gain = gain
                                best_insert = (origA, newB)
                    if best_insert:
                        best_routes[a], best_routes[b] = best_insert
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
    return best_routes

# -------------------------
# GRASP main
# -------------------------
def build_routes_from_mst_random(n_nodes, method, ids, idx, D, k_routes=3, rcl_size=3, seed=None):
    if method=='prim':
        mst = randomized_prim(ids, idx, D, rcl_size=rcl_size, seed=seed)
    else:
        mst = randomized_kruskal(ids, idx, D, rcl_size=rcl_size, seed=seed)
    comps = split_mst_edges_to_routes(mst, n_nodes, k_routes)
    routes = [list(c) for c in comps]
    # ensure start/end removal won't accidentally remove indices: handled later
    return routes, mst  # return also mst so we can desenhar ela inteira

def grasp(ids, idx, D, coords, k_routes=3, iterations=100, method='prim', rcl_size=3, seed=None, depot_start=0, depot_end=None):
    best = None
    best_cost = float('inf')
    n = len(ids)
    for it in range(iterations):
        s, _ = build_routes_from_mst_random(n, method, ids, idx, D, k_routes=k_routes, rcl_size=rcl_size, seed=(seed or 0)+it)
        # s: list of components (each is list of node indices)
        # remove start/end from inner lists so they are treated as terminals
        processed = []
        for comp in s:
            comp = [v for v in comp if v != depot_start and (depot_end is None or v != depot_end)]
            processed.append(comp)
        # local search: 2-opt each route then relocate between routes (pass depots)
        s2 = [two_opt(r, D, depot_start=depot_start, depot_end=depot_end) for r in processed]
        s3 = relocate_between_routes(s2, D, depot_start=depot_start, depot_end=depot_end)
        total = sum(route_distance(r, D, depot_start=depot_start, depot_end=depot_end) for r in s3)
        if total < best_cost:
            best_cost = total
            best = s3
    return best, best_cost

# -------------------------
# plotting
# -------------------------
def plot_solution(routes, coords, ids, D, mst_edges=None, depot_start=0, depot_end=None, show_edge_labels=True):
    plt.figure(figsize=(9,9))
    pos = {i: coords[ids[i]] for i in range(len(ids))}

    # desenha MST completa em cinza claro (para mostrar interligações)
    if mst_edges:
        for u,v,w in mst_edges:
            x1,y1 = pos[u]
            x2,y2 = pos[v]
            plt.plot([x1,x2],[y1,y2], color='lightgray', linewidth=1, zorder=1, linestyle='--')

    # desenha vértices (círculos) e números
    for i, (x, y) in pos.items():
        plt.scatter(x, y, color='white', edgecolor='black', s=140, zorder=4)
        plt.text(x, y, str(i), fontsize=9, ha='center', va='center', color='black', fontweight='bold', zorder=5)

    # cores para rotas
    cmap = plt.colormaps.get_cmap('tab10')
    for r_idx, route in enumerate(routes):
        color = cmap(r_idx % cmap.N)
        # sequence from start to end (if end is None we return to start)
        if depot_end is None:
            seq = [depot_start] + route + [depot_start]
        else:
            seq = [depot_start] + route + [depot_end]
        # desenha linhas da rota
        for i in range(len(seq)-1):
            x1, y1 = pos[seq[i]]
            x2, y2 = pos[seq[i+1]]
            plt.plot([x1, x2], [y1, y2], color=color, linewidth=2.5, zorder=3)
            # rótulo da distância (se desejado)
            if show_edge_labels:
                dist = D[seq[i], seq[i+1]]
                xm, ym = (x1+x2)/2, (y1+y2)/2
                plt.text(xm, ym, f"{dist:.1f}", color=color, fontsize=8, ha='center', va='center', zorder=6, bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=0.5))

    # destaque: start (vermelho) e end (verde)
    xs, ys = pos[depot_start]
    plt.scatter(xs, ys, color='red', s=220, zorder=7, label=f'Start ({depot_start})')
    if depot_end is None:
        plt.scatter(xs, ys, color='red', s=220, zorder=7)
    else:
        xe, ye = pos[depot_end]
        plt.scatter(xe, ye, color='green', s=220, zorder=7, label=f'End ({depot_end})')

    plt.title("Rotas (start!=end) — MST pontilhada em cinza, rotas em cor")
    plt.legend()
    plt.axis('equal')
    plt.show()


# -------------------------
# example usage
# -------------------------
if __name__=='__main__':
    # Configuração
    n_points = 10
    n_routes = 4
    iterations = 200
    seed = 123

    pts = generate_random_points(n_points, seed=seed, bounds=(0,1000))
    ids, idx, D, coords = build_distance_matrix(pts)

    #pts = [
    #(0, 100, 100),   # Ponto inicial (depósito)
    #(1, 200, 300),
    #(2, 400, 250),
    #(3, 500, 500),
    #(4, 700, 600),
    #(5, 800, 200),
    #(6, 300, 700),
    #(7, 100, 900),
    #(8, 900, 900),
    #(9, 600, 100)
#]

    # Sorteia start e end aleatórios (diferentes)
    rng = random.Random(seed)
    depot_start = rng.randrange(len(ids))
    depot_end = rng.randrange(len(ids))
    while depot_end == depot_start:
        depot_end = rng.randrange(len(ids))

    print(f"start (início) sorteado: {depot_start}")
    print(f"end (chegada) sorteado: {depot_end}")


   


    #desenhho
    mst_edges = randomized_prim(ids, idx, D, rcl_size=3, seed=seed)

    start_time = time.time()
    best_routes, best_cost = grasp(ids, idx, D, coords,
                                  k_routes=n_routes,
                                  iterations=iterations,
                                  method='prim',
                                  rcl_size=5,
                                  seed=seed,
                                  depot_start=depot_start,
                                  depot_end=depot_end)
    elapsed = time.time() - start_time

    print(f"Melhor custo total (considerando start->visitas->end por rota): {best_cost:.2f}  em {elapsed:.2f}s")
    for i,r in enumerate(best_routes):
        print(f"Rota {i} ({len(r)} pontos): {r}")

         # Mostrar a distância de cada rota separadamente
    print("\nDistâncias por rota:")
    distances = []
    for i, r in enumerate(best_routes):
        dist = route_distance(r, D, depot_start=depot_start, depot_end=depot_end)
        distances.append((i, dist))
        print(f"  Rota {i}: {dist:.2f} unidades")

    # Mostrar qual é a melhor (menor distância)
    best_route_idx, best_dist = min(distances, key=lambda x: x[1])
    print(f"\nMelhor rota: Rota {best_route_idx} com distância {best_dist:.2f}")

    # Plota (passa mst_edges p/ mostrar interligações)
    plot_solution(best_routes, coords, ids, D, mst_edges=mst_edges, depot_start=depot_start, depot_end=depot_end)
