import math
import matplotlib.pyplot as plt
import numpy as np
import os

def limpiar():
    os.system('cls' if os.name == 'nt' else 'clear')

# Binomial

def prob_binomial(n, k, p):
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

# PMF CDF

def graficar(x, pmf, titulo, k_destacado=None):
    cdf = list(np.cumsum(pmf))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(titulo, fontsize=12, fontweight="bold")

    colores = ["red" if i == k_destacado else "steelblue" for i in range(len(x))]

    # PMF
    ax1.bar(x, pmf, color=colores, edgecolor="navy", width=0.6)
    ax1.set_title("Función de Probabilidad (PMF)")
    ax1.set_xlabel("k")
    ax1.set_ylabel("P(X = k)")
    ax1.set_xticks(x)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # CDF
    ax2.step(x, cdf, where="post", color="green", linewidth=2)
    ax2.scatter(x, cdf, color="green", s=40)
    ax2.set_title("Función Acumulada (CDF)")
    ax2.set_xlabel("k")
    ax2.set_ylabel("P(X ≤ k)")
    ax2.set_xticks(x)
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
#Linea discontinua
    if k_destacado is not None:
        ax2.axvline(x=x[k_destacado], color="red", linestyle="--", linewidth=1.2)
        ax2.axhline(y=cdf[k_destacado], color="red", linestyle=":", linewidth=1.2)

    plt.tight_layout()
    plt.show()
#grafica binomial
def graficar_binomial(n, p, k_destacado=None):
    x   = list(range(n + 1))
    pmf = [prob_binomial(n, k, p) for k in x]
    graficar(x, pmf, f"Distribución Binomial  (n={n}, p={p})", k_destacado)

def graficar_uniforme(total, titulo):
    if total > 40:
        bloque = max(1, total // 20)
        x   = list(range(1, total + 1, bloque))
        pmf = [bloque / total] * len(x)
    else:
        x   = list(range(1, total + 1))
        pmf = [1 / total] * total
    graficar(x, pmf, titulo)

#Menú

def main():
    while True:
        limpiar()
        print("\n SISTEMA DE PROBABILIDAD - VARIABLES DISCRETAS")
        print("1. Combinaciones C(n,r)")
        print("2. Variaciones V(n,r)")
        print("3. Permutaciones P(n)")
        print("4. Probabilidad Binomial")
        print("5. Probabilidad con sustitución")
        print("6. Probabilidad sin sustitución")
        print("7. Salir")
        opcion = input("\nSeleccione una opción: ")

        if opcion == "1":
            n = int(input("Ingrese n: "))
            r = int(input("Ingrese r: "))
            resultado = math.comb(n, r)
            print(f"C({n},{r}) = {resultado}")
            if resultado > 0:
                input("\nPresione Enter para ver la gráfica...")
                graficar_uniforme(resultado, f"C({n},{r}) = {resultado} combinaciones")
            else:
                print("No hay combinaciones posibles (r > n).")

        elif opcion == "2":
            n = int(input("Ingrese n: "))
            r = int(input("Ingrese r: "))
            resultado = math.perm(n, r)
            print(f"V({n},{r}) = {resultado}")
            if resultado > 0:
                input("\nPresione Enter para ver la gráfica...")
                graficar_uniforme(resultado, f"V({n},{r}) = {resultado} variaciones")
            else:
                print("No hay variaciones posibles (r > n).")

        elif opcion == "3":
            n = int(input("Ingrese n: "))
            resultado = math.factorial(n)
            print(f"P({n}) = {resultado}")
            input("\nPresione Enter para ver la gráfica...")
            graficar_uniforme(resultado, f"P({n}) = {resultado} permutaciones")

        elif opcion == "4":
            n = int(input("Número de ensayos: "))
            k = int(input("Número de éxitos: "))
            p = float(input("Probabilidad de éxito (decimal): "))
            resultado = prob_binomial(n, k, p)
            print(f"P(X = {k}) = {resultado:.6f}")
            input("\nPresione Enter para ver la gráfica...")
            graficar_binomial(n, p, k_destacado=k)

        elif opcion == "5":
            favorables = float(input("Cantidad favorable: "))
            total      = float(input("Total elementos: "))
            veces      = int(input("Número de selecciones: "))
            p    = favorables / total
            prob = p ** veces
            print(f"Probabilidad: {prob:.6f}")
            input("\nPresione Enter para ver la gráfica...")
            graficar_binomial(veces, p, k_destacado=veces)

        elif opcion == "6":
            N          = int(input("Total elementos: "))
            n          = int(input("Cantidad seleccionada: "))
            K          = int(input("Favorables dentro del total: "))
            k_deseados = int(input("Cantidad favorable deseada: "))
            prob = math.comb(K, k_deseados) * math.comb(N - K, n - k_deseados) / math.comb(N, n)
            print(f"Probabilidad: {prob:.6f}")
            input("\nPresione Enter para ver la gráfica...")
            graficar_binomial(n, K / N, k_destacado=k_deseados)

        elif opcion == "7":
            print("Saliendo...")
            break

        else:
            print("Opción inválida.")

        input("\nPresione Enter para continuar...")

if __name__ == "__main__":
    main()


