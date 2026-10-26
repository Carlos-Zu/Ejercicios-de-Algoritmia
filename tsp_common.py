"""Funciones compartidas por los ejercicios 3, 4 y 5 (TSP)."""
import itertools
import math
import random


def tour_length(points, tour) -> float:
    return sum(math.dist(points[tour[i]], points[tour[(i + 1) % len(tour)]]) for i in range(len(tour)))


def two_opt_move(route: list[int], i: int, k: int) -> list[int]:
    """Reverse route[i..k]: removes edges (i-1, i) and (k, k+1)."""
    return route[:i] + route[i:k + 1][::-1] + route[k + 1:]


def mst_prim(points: list[tuple[float, float]]) -> dict[int, list[int]]:
    """Prim's algorithm from city 0; returns the tree as {node: children}."""
    best = {v: (math.dist(points[0], points[v]), 0) for v in range(1, len(points))}
    children = {v: [] for v in range(len(points))}
    while best:
        v = min(best, key=lambda u: best[u][0])
        children[best.pop(v)[1]].append(v)          # add the cheapest edge to the tree
        for u in best:
            d = math.dist(points[v], points[u])
            if d < best[u][0]:
                best[u] = (d, v)
    return children


def preorder(children: dict[int, list[int]], root: int = 0) -> list[int]:
    """Walk around the tree; visiting each city once = shortcutting repeated cities."""
    return [root] + [c for child in children[root] for c in preorder(children, child)]


def optimal_tour(points) -> tuple[int, ...]:
    """Fuerza bruta (como en el PDF): (n-1)! tours fijando la ciudad 0."""
    return min(((0,) + p for p in itertools.permutations(range(1, len(points)))),
               key=lambda t: tour_length(points, t))


def random_tour(n: int) -> list[int]:
    tour = list(range(n))
    random.shuffle(tour)
    return tour


def nearest_neighbour(points, start: int | None = None) -> list[int]:
    """Vecino Mas Cercano; si `start` es None, inicia en un nodo aleatorio."""
    if start is None:
        start = random.randrange(len(points))
    unvisited = set(range(len(points))) - {start}
    tour = [start]
    while unvisited:
        nxt = min(unvisited, key=lambda c: math.dist(points[tour[-1]], points[c]))
        tour.append(nxt)
        unvisited.remove(nxt)
    return tour


def two_opt(points, tour, eps: float = 1e-9):
    """
    Repite el movimiento 2-opt para cada par (i, k) hasta que ninguna reversion acorte el tour
    (se aplica la primera mejora que se encuentra). Devuelve (tour, numero_de_mejoras).
    """
    tour = list(tour)
    n = len(tour)
    mejoras = 0
    mejoro = True
    while mejoro:
        mejoro = False
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                a, b = points[tour[i - 1]], points[tour[i]]
                c, d = points[tour[k]], points[tour[(k + 1) % n]]
                delta = math.dist(a, c) + math.dist(b, d) - math.dist(a, b) - math.dist(c, d)
                if delta < -eps:
                    tour = two_opt_move(tour, i, k)
                    mejoras += 1
                    mejoro = True
    return tour, mejoras


def linear_schedule(T0: float, n_steps: int, T_min: float = 0.01):
    """T baja en linea recta de T0 a 0 en n_steps iteraciones."""
    def schedule(t):
        T = T0 * (1 - t / n_steps)
        return T if T > T_min else 0
    return schedule


def exponential_schedule(T0: float, alpha: float, T_min: float = 0.01):
    """T = T0 * alpha^t (enfriamiento geometrico)."""
    def schedule(t):
        T = T0 * alpha ** t
        return T if T > T_min else 0
    return schedule


def simulated_annealing(points, schedule, initial_state, trace_every: int = 0):
    """
    function SIMULATED-ANNEALING(problem, schedule): VALUE = -longitud del tour (el pseudocodigo maximiza),
    RANDOM_SUCCESSOR = un movimiento 2-opt al azar. Devuelve (current, iteraciones, traza).
    La traza guarda (iteracion, T, costo_actual) cada `trace_every` iteraciones.
    """
    current = list(initial_state)
    cur_val = -tour_length(points, current)
    traza = []
    t = 1
    while True:                                           # for t = 1 to infinity
        T = schedule(t)
        if T == 0:                                        # if T = 0 then return current
            return current, t - 1, traza
        i, k = sorted(random.sample(range(1, len(current)), 2))
        nxt = two_opt_move(current, i, k)                 # next = RANDOM_SUCCESSOR(current)
        nxt_val = -tour_length(points, nxt)
        delta_E = nxt_val - cur_val
        if delta_E > 0:
            current, cur_val = nxt, nxt_val
        elif random.random() < math.exp(delta_E / T):     # aceptar un movimiento malo con prob e^(dE/T)
            current, cur_val = nxt, nxt_val
        if trace_every and t % trace_every == 0:
            traza.append((t, T, -cur_val))
        t += 1
