import random
import statistics
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tsp_common import (tour_length, random_tour, simulated_annealing,
                        linear_schedule, exponential_schedule)

T0 = 50.0
ESQUEMAS = {
    "Lineal (5000 pasos)": linear_schedule(T0, 5000),
    "Exponencial α=0.99": exponential_schedule(T0, 0.99),
    "Exponencial α=0.995": exponential_schedule(T0, 0.995),
    "Exponencial α=0.999": exponential_schedule(T0, 0.999),
}

if __name__ == "__main__":
    n_ciudades, n_corridas, paso_traza = 30, 20, 25
    random.seed(11)
    pts = [(random.uniform(0, 100), random.uniform(0, 100)) for _ in range(n_ciudades)]

    resultados = {}
    for nombre, sched in ESQUEMAS.items():
        finales, tiempos, trazas, iters = [], [], [], 0
        for r in range(n_corridas):
            random.seed(100 + r)
            inicio = random_tour(n_ciudades)                       # solucion inicial aleatoria
            t0 = time.perf_counter()
            tour, iters, traza = simulated_annealing(pts, sched, inicio, trace_every=paso_traza)
            tiempos.append(time.perf_counter() - t0)
            finales.append(tour_length(pts, tour))
            trazas.append(traza)
        resultados[nombre] = {"finales": finales, "tiempos": tiempos, "iters": iters, "trazas": trazas}

    print(f"{n_ciudades} ciudades, {n_corridas} corridas por esquema, T0 = {T0}")
    print(f"{'esquema':22s} {'iteraciones':>11} {'costo medio':>12} {'desv.':>7} {'mejor':>8} {'tiempo medio (s)':>17}")
    for nombre, r in resultados.items():
        print(f"{nombre:22s} {r['iters']:>11,} {statistics.mean(r['finales']):>12.1f} "
              f"{statistics.stdev(r['finales']):>7.1f} {min(r['finales']):>8.1f} {statistics.mean(r['tiempos']):>17.3f}")

    # 1) Curvas de convergencia: costo vs iteraciones y costo vs temperatura (promedio de las corridas)
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for nombre, r in resultados.items():
        its = [p[0] for p in r["trazas"][0]]
        Ts = [p[1] for p in r["trazas"][0]]
        costo_medio = [statistics.mean(tr[j][2] for tr in r["trazas"]) for j in range(len(its))]
        ax[0].plot(its, costo_medio, label=nombre)
        ax[1].plot(Ts, costo_medio, label=nombre)
    ax[0].set_xlabel("iteración"); ax[0].set_ylabel("costo (longitud del tour actual)")
    ax[0].set_title("Convergencia: costo vs iteraciones"); ax[0].set_xscale("log")
    ax[1].set_xlabel("temperatura T (el enfriamiento va de izquierda a derecha)")
    ax[1].set_xscale("log"); ax[1].invert_xaxis()
    ax[1].set_title("Convergencia: costo vs temperatura")
    for a in ax:
        a.grid(alpha=.3, which="both"); a.legend()
    plt.tight_layout()
    plt.savefig("ej4_convergencia.png", dpi=130)

    # 2) Tiempo de ejecucion vs calidad de la solucion final
    plt.figure(figsize=(7, 5))
    for nombre, r in resultados.items():
        sc = plt.scatter(r["tiempos"], r["finales"], alpha=.4, s=18)
        plt.scatter(statistics.mean(r["tiempos"]), statistics.mean(r["finales"]), marker="X", s=140,
                    color=sc.get_facecolor()[0][:3], edgecolor="black", label=f"{nombre} (media)")
    plt.xlabel("tiempo de ejecución (s)"); plt.ylabel("costo final (longitud del tour)")
    plt.title("Tiempo vs calidad (cada punto es una corrida)")
    plt.grid(alpha=.3); plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("ej4_tiempo_vs_calidad.png", dpi=130)
    print("Gráficas guardadas en ej4_convergencia.png y ej4_tiempo_vs_calidad.png")
