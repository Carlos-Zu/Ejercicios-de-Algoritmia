import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsp_common import (tour_length, mst_prim, preorder, optimal_tour, random_tour, nearest_neighbour,
                        two_opt, simulated_annealing, exponential_schedule)

METODOS = ["MST puro", "MST + 2-opt", "Búsqueda local (inicio aleatorio)",
           "Búsqueda local (Vecino más cercano)", "Recocido simulado"]

if __name__ == "__main__":
    random.seed(7)                      # misma semilla que el PDF: la instancia 1 es la del PDF (razón 1.298)
    n_ciudades, n_instancias = 9, 100
    sched_sa = exponential_schedule(5.0, 0.995)   # SA: T0 = 5 (instancias en [0,10]^2), enfriamiento exponencial

    costos = {m: [] for m in METODOS}
    optimos = []
    for _ in range(n_instancias):
        pts = [(random.uniform(0, 10), random.uniform(0, 10)) for _ in range(n_ciudades)]
        optimos.append(tour_length(pts, optimal_tour(pts)))          # fuerza bruta: 8! = 40,320 tours

        mst = preorder(mst_prim(pts))                                # aproximacion MST
        costos["MST puro"].append(tour_length(pts, mst))
        costos["MST + 2-opt"].append(tour_length(pts, two_opt(pts, mst)[0]))
        costos["Búsqueda local (inicio aleatorio)"].append(tour_length(pts, two_opt(pts, random_tour(n_ciudades))[0]))
        costos["Búsqueda local (Vecino más cercano)"].append(tour_length(pts, two_opt(pts, nearest_neighbour(pts))[0]))
        sa, _, _ = simulated_annealing(pts, sched_sa, random_tour(n_ciudades))
        costos["Recocido simulado"].append(tour_length(pts, sa))

    razones = {m: [c / o for c, o in zip(costos[m], optimos)] for m in METODOS}

    print(f"{n_instancias} instancias de {n_ciudades} ciudades (razón = C_obtenido / C_óptimo)")
    print(f"{'método':37s} {'costo medio':>12} {'razón media':>12} {'razón máx.':>11} {'= óptimo':>9}")
    for m in METODOS:
        r = razones[m]
        print(f"{m:37s} {statistics.mean(costos[m]):>12.2f} {statistics.mean(r):>12.3f} {max(r):>11.3f} "
              f"{sum(x < 1 + 1e-7 for x in r):>6d}/{n_instancias}")
    print(f"Costo óptimo medio: {statistics.mean(optimos):.2f}")
    print(f"Instancias donde MST + 2-opt es peor que MST puro: "
          f"{sum(b > a + 1e-9 for a, b in zip(costos['MST puro'], costos['MST + 2-opt']))}")

    # 1) Histograma de la razon de aproximacion (MST puro) con linea en el limite teorico de 2
    plt.figure(figsize=(7, 4.5))
    plt.hist(razones["MST puro"], bins=15, color="tab:blue", edgecolor="white")
    plt.axvline(2, color="red", ls="--", label="límite teórico = 2")
    plt.xlim(0.95, 2.1)
    plt.xlabel("razón de aproximación  C_obtenido / C_óptimo"); plt.ylabel("instancias")
    plt.title("Razón de aproximación del MST (100 instancias de 9 ciudades)")
    plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("ej5_razon_mst.png", dpi=130)

    # 2) Boxplots del costo obtenido por metodo
    plt.figure(figsize=(8, 4.8))
    plt.boxplot([costos[m] for m in METODOS])
    plt.xticks(range(1, 6), ["MST\npuro", "MST\n+ 2-opt", "Local\n(aleatorio)", "Local\n(NN)", "Recocido\nsimulado"])
    plt.ylabel("costo obtenido (longitud del tour)")
    plt.title("Costo obtenido por método")
    plt.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig("ej5_comparacion_metodos.png", dpi=130)
    print("Gráficas guardadas en ej5_razon_mst.png y ej5_comparacion_metodos.png")
