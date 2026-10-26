import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def greedy(values, weights, W, key):
    """Algoritmo voraz: ordena los objetos con `key` (menor primero) y mete cada uno que quepa."""
    order = sorted(range(len(values)), key=key)
    chosen, cap = [], W
    for i in order:
        if weights[i] <= cap:
            chosen.append(i)
            cap -= weights[i]
    return sorted(chosen)


def greedy_ratio(values, weights, W):
    """Metodo base: mayor valor/peso primero."""
    return greedy(values, weights, W, key=lambda i: -values[i] / weights[i])


def greedy_min_weight(values, weights, W):
    """MODIFICACION del ejercicio: menor peso primero."""
    return greedy(values, weights, W, key=lambda i: weights[i])


def resumen(names, values, weights, chosen):
    return {"objetos": [names[i] for i in chosen],
            "capacidad": sum(weights[i] for i in chosen),
            "valor": sum(values[i] for i in chosen)}


if __name__ == "__main__":
    # Instancia pequeña: menor peso mete tres objetos ligeros y el valioso (peso 8) ya no cabe
    names, values, weights, W = ["A", "B", "C", "D"], [1, 2, 3, 60], [1, 2, 3, 8], 10

    filas = {"Valor/peso": resumen(names, values, weights, greedy_ratio(values, weights, W)),
             "Menor peso": resumen(names, values, weights, greedy_min_weight(values, weights, W))}

    print(f"Capacidad W = {W}; objetos (valor, peso): {dict(zip(names, zip(values, weights)))}")
    print(f"{'método':12s} {'objetos seleccionados':24s} {'capacidad usada':>16} {'valor total':>12}")
    for m, f in filas.items():
        print(f"{m:12s} {str(f['objetos']):24s} {f['capacidad']:>10d}/{W:<5d} {f['valor']:>12d}")

    metodos = list(filas)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, campo, titulo in [(ax[0], "valor", "Valor total"), (ax[1], "capacidad", "Capacidad usada")]:
        vals = [filas[m][campo] for m in metodos]
        barras = a.bar(metodos, vals, color=["tab:blue", "tab:orange"])
        for b, m in zip(barras, metodos):
            a.text(b.get_x() + b.get_width() / 2, b.get_height(), "objetos: " + "".join(filas[m]["objetos"]),
                   ha="center", va="bottom", fontsize=9)
        if campo == "capacidad":
            a.axhline(W, color="red", ls="--", label=f"capacidad W = {W}")
            a.legend()
        a.set_title(titulo)
        a.set_ylim(0, max(vals + [W if campo == "capacidad" else 0]) * 1.2)
        a.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig("ej2_comparacion.png", dpi=130)
    print("Gráfica guardada en ej2_comparacion.png")
