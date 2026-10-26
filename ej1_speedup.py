import multiprocessing as mp
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def estimate_pi(seed: int, samples: int = 1_000_000) -> float:
    """Monte Carlo: fraction of random points that fall inside the unit quarter circle."""
    rng = random.Random(seed)
    inside = sum(rng.random()**2 + rng.random()**2 <= 1.0 for _ in range(samples))
    return 4 * inside / samples


def run_with(n_procs: int, seeds: list[int]) -> float:
    """Wall-clock time de correr todas las semillas con n_procs procesos."""
    t0 = time.perf_counter()
    if n_procs == 1:
        [estimate_pi(s) for s in seeds]          # 1 proceso = ejecucion secuencial
    else:
        with mp.Pool(processes=n_procs) as pool:
            pool.map(estimate_pi, seeds)
    return time.perf_counter() - t0


if __name__ == "__main__":
    seeds = list(range(8))          # 8 corridas independientes: el trabajo total es el mismo siempre
    procs = [1, 2, 4, 8]
    repeticiones = 3                # se promedia para reducir ruido

    print(f"CPUs visibles: {mp.cpu_count()}")
    tiempos = [sum(run_with(p, seeds) for _ in range(repeticiones)) / repeticiones for p in procs]
    speedup = [tiempos[0] / t for t in tiempos]

    print(f"{'procesos':>8} {'tiempo (s)':>11} {'speedup':>8}")
    for p, t, s in zip(procs, tiempos, speedup):
        print(f"{p:>8} {t:>11.3f} {s:>8.2f}")

    plt.figure(figsize=(6, 4.5))
    plt.plot(procs, speedup, "o-", label="speedup medido")
    plt.plot(procs, procs, "k--", label="speedup ideal (lineal)")
    plt.xlabel("número de procesos"); plt.ylabel("speedup = T1 / Tp")
    plt.title("Aceleración paralela de estimate_pi")
    plt.xticks(procs); plt.legend(); plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig("ej1_speedup.png", dpi=130)
    print("Gráfica guardada en ej1_speedup.png")
