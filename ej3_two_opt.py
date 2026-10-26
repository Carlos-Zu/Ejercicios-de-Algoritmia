import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsp_common import tour_length, random_tour, nearest_neighbour, two_opt

if __name__ == "__main__":
    random.seed(3)
    n_ciudades, n_instancias, n_corridas = 40, 30, 20   # 20 corridas de cada tipo por instancia

    mejoras = {"Aleatorio": [], "Vecino más cercano": []}
    calidad = {"Aleatorio": [], "Vecino más cercano": []}     # longitud final / mejor longitud de esa instancia

    for _ in range(n_instancias):
        pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_ciudades)]
        corridas = {"Aleatorio": [], "Vecino más cercano": []}
        for _ in range(n_corridas):
            for nombre, inicio in [("Aleatorio", random_tour(n_ciudades)),
                                   ("Vecino más cercano", nearest_neighbour(pts))]:   # nodo inicial aleatorio
                tour, m = two_opt(pts, inicio)
                corridas[nombre].append((m, tour_length(pts, tour)))
        mejor = min(L for lista in corridas.values() for _, L in lista)   # referencia para comparar instancias
        for nombre, lista in corridas.items():
            mejoras[nombre] += [m for m, _ in lista]
            calidad[nombre] += [L / mejor for _, L in lista]

    print(f"{n_instancias} instancias de {n_ciudades} ciudades x {n_corridas} corridas por tipo de inicio")
    print(f"{'inicio':20s} {'mejoras (media)':>16} {'mediana':>8} {'mín-máx':>10} {'calidad media (L/L_mejor)':>26}")
    for nombre in mejoras:
        m = mejoras[nombre]
        print(f"{nombre:20s} {statistics.mean(m):>16.1f} {statistics.median(m):>8.0f} "
              f"{min(m):>4d}-{max(m):<5d} {statistics.mean(calidad[nombre]):>26.4f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    ax[0].boxplot([mejoras[k] for k in mejoras])
    ax[0].set_xticklabels(list(mejoras))
    ax[0].set_ylabel("número de mejoras 2-opt hasta el óptimo local")
    ax[0].set_title("Mejoras necesarias para converger")
    ax[1].boxplot([calidad[k] for k in calidad])
    ax[1].set_xticklabels(list(calidad))
    ax[1].set_ylabel("longitud final / mejor longitud de la instancia")
    ax[1].set_title("Calidad del óptimo local (1 = el mejor encontrado)")
    for a in ax:
        a.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig("ej3_boxplots.png", dpi=130)
    print("Gráfica guardada en ej3_boxplots.png")
