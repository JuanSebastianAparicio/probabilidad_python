"""
 PROBABILIDAD CONJUNTA DISCRETA
 Problema: Control de Calidad en Línea de Ensamblaje
 Variables: X (defectos visuales), Y (fallas eléctricas)
 Rango: X, Y ∈ {0, 1, 2}
"""

import numpy as np
import math


#  1. REPRESENTACIÓN DE LA DISTRIBUCIÓN CONJUNTA


def crear_distribucion_conjunta():
    """
    Construye la matriz de probabilidad conjunta f(x, y).
    Filas  → X = {0, 1, 2}
    Columnas → Y = {0, 1, 2}
    """
    # f[i][j] = P(X = i, Y = j)
    f = np.array([
        [0.20, 0.10, 0.05],   # X = 0
        [0.15, 0.20, 0.05],   # X = 1
        [0.05, 0.10, 0.10],   # X = 2
    ])
    return f


def imprimir_tabla_conjunta(f, titulo="Distribución de Probabilidad Conjunta f(x,y)"):
    """Imprime la tabla conjunta con encabezados de filas y columnas."""
    n = f.shape[0]
    sep = "=" * 54
    print(f"\n{sep}")
    print(f"  {titulo}")
    print(sep)
    encabezado = f"{'X\\Y':>6}" + "".join(f"{'Y='+str(j):>10}" for j in range(n)) + f"{'f_X(x)':>10}"
    print(encabezado)
    print("-" * 54)
    for i in range(n):
        marginal_x = np.sum(f[i, :])
        fila = f"{'X='+str(i):>6}" + "".join(f"{f[i, j]:>10.4f}" for j in range(n)) + f"{marginal_x:>10.4f}"
        print(fila)
    print("-" * 54)
    marginal_y = np.sum(f, axis=0)
    pie = f"{'f_Y(y)':>6}" + "".join(f"{marginal_y[j]:>10.4f}" for j in range(n)) + f"{np.sum(f):>10.4f}"
    print(pie)
    print(sep)



#  2. VERIFICACIÓN DE NORMALIZACIÓN


def verificar_normalizacion(f):
    """Verifica que la suma de todas las probabilidades sea 1."""
    total = np.sum(f)
    print("\n--- Verificación de Normalización ---")
    print(f"  Suma total de f(x,y) = {total:.6f}")
    es_valida = math.isclose(total, 1.0, abs_tol=1e-9)
    print(f"  ¿Es distribución válida? {'Sí ✓' if es_valida else 'No ✗'}")
    return es_valida



#  3. PROBABILIDADES ESPECÍFICAS


def calcular_probabilidades_especificas(f):
    """Calcula probabilidades de eventos específicos."""
    n = f.shape[0]
    print("\n" + "=" * 54)
    print("  PROBABILIDADES ESPECÍFICAS")
    print("=" * 54)

    # P(X + Y <= 2)
    p_suma_leq_2 = sum(
        f[x, y]
        for x in range(n)
        for y in range(n)
        if x + y <= 2
    )
    print(f"\n  P(X + Y ≤ 2) = {p_suma_leq_2:.4f}")
    print(f"  Pares incluidos: {[(x, y) for x in range(n) for y in range(n) if x+y<=2]}")

    # P(X = Y)
    p_x_igual_y = sum(f[k, k] for k in range(n))
    print(f"\n  P(X = Y)    = {p_x_igual_y:.4f}")
    print(f"  Pares incluidos: {[(k, k) for k in range(n)]}")

    # P(X >= 1, Y >= 1)
    p_ambos_mayor1 = sum(
        f[x, y]
        for x in range(n)
        for y in range(n)
        if x >= 1 and y >= 1
    )
    print(f"\n  P(X ≥ 1, Y ≥ 1) = {p_ambos_mayor1:.4f}")
    print(f"  Pares incluidos: {[(x, y) for x in range(n) for y in range(n) if x>=1 and y>=1]}")

    return p_suma_leq_2, p_x_igual_y, p_ambos_mayor1


#  4. DISTRIBUCIONES MARGINALES


def calcular_marginales(f):
    """
    Calcula las distribuciones marginales de X e Y.
    f_X(x) = Σ_y f(x, y)
    f_Y(y) = Σ_x f(x, y)
    """
    f_X = np.sum(f, axis=1)   # suma sobre columnas → marginal de X
    f_Y = np.sum(f, axis=0)   # suma sobre filas    → marginal de Y

    print("\n" + "=" * 54)
    print("  DISTRIBUCIONES MARGINALES")
    print("=" * 54)

    print("\n  Marginal de X  [f_X(x) = Σ_y f(x,y)]:")
    for x, px in enumerate(f_X):
        print(f"    f_X({x}) = {px:.4f}")
    print(f"  Suma: {np.sum(f_X):.4f}  ✓" if math.isclose(sum(f_X), 1.0, abs_tol=1e-9) else "  ¡Error!")

    print("\n  Marginal de Y  [f_Y(y) = Σ_x f(x,y)]:")
    for y, py in enumerate(f_Y):
        print(f"    f_Y({y}) = {py:.4f}")
    print(f"  Suma: {np.sum(f_Y):.4f}  ✓" if math.isclose(sum(f_Y), 1.0, abs_tol=1e-9) else "  ¡Error!")

    return f_X, f_Y



#  5. ESPERANZA MATEMÁTICA


def calcular_esperanza(f, f_X, f_Y):
    """
    Calcula E(X), E(Y), E(X+Y) y E(XY).
    E(X) = Σ_x x * f_X(x)
    E(Y) = Σ_y y * f_Y(y)
    E(XY) = Σ_{x,y} x * y * f(x,y)
    """
    n = f.shape[0]
    valores = np.arange(n)

    E_X  = np.dot(valores, f_X)
    E_Y  = np.dot(valores, f_Y)
    E_XY = sum(
        x * y * f[x, y]
        for x in range(n)
        for y in range(n)
    )

    print("\n" + "=" * 54)
    print("  ESPERANZA MATEMÁTICA")
    print("=" * 54)
    print(f"\n  E(X)    = {E_X:.4f}")
    print(f"  E(Y)    = {E_Y:.4f}")
    print(f"  E(X+Y)  = E(X) + E(Y) = {E_X + E_Y:.4f}  [linealidad]")
    print(f"  E(XY)   = {E_XY:.4f}")

    return E_X, E_Y, E_XY



#  6. VARIANZA Y COVARIANZA


def calcular_varianza_covarianza(f, f_X, f_Y, E_X, E_Y, E_XY):
    """
    Var(X)     = E(X²) - [E(X)]²
    Var(Y)     = E(Y²) - [E(Y)]²
    Cov(X,Y)   = E(XY) - E(X)*E(Y)
    ρ(X,Y)     = Cov(X,Y) / sqrt(Var(X) * Var(Y))
    """
    n = f.shape[0]
    valores = np.arange(n)

    E_X2 = np.dot(valores**2, f_X)
    E_Y2 = np.dot(valores**2, f_Y)

    Var_X = E_X2 - E_X**2
    Var_Y = E_Y2 - E_Y**2
    Cov   = E_XY - E_X * E_Y
    rho   = Cov / math.sqrt(Var_X * Var_Y)

    print("\n" + "=" * 54)
    print("  VARIANZA Y COVARIANZA")
    print("=" * 54)
    print(f"\n  E(X²)         = {E_X2:.4f}")
    print(f"  E(Y²)         = {E_Y2:.4f}")
    print(f"\n  Var(X)        = E(X²) - [E(X)]² = {E_X2:.4f} - {E_X**2:.4f} = {Var_X:.4f}")
    print(f"  Var(Y)        = E(Y²) - [E(Y)]² = {E_Y2:.4f} - {E_Y**2:.4f} = {Var_Y:.4f}")
    print(f"\n  Cov(X,Y)      = E(XY) - E(X)·E(Y) = {E_XY:.4f} - {E_X*E_Y:.4f} = {Cov:.4f}")
    print(f"  ρ(X,Y)        = Cov / √(Var(X)·Var(Y)) = {rho:.6f}")

    return Var_X, Var_Y, Cov, rho



#  7. VERIFICACIÓN DE INDEPENDENCIA


def verificar_independencia(f, f_X, f_Y, tolerancia=1e-9):
    """
    X e Y son independientes ⟺ f(x,y) = f_X(x) * f_Y(y)  ∀(x,y)
    Verifica celda a celda.
    """
    n = f.shape[0]
    print("\n" + "=" * 54)
    print("  VERIFICACIÓN DE INDEPENDENCIA")
    print("=" * 54)
    print(f"\n  Condición: f(x,y) = f_X(x) · f_Y(y) para todo (x,y)\n")

    encabezado = f"  {'(x,y)':>6}  {'f(x,y)':>8}  {'fX·fY':>8}  {'Diferencia':>12}  {'¿Igual?':>8}"
    print(encabezado)
    print("  " + "-" * 52)

    todas_iguales = True
    for x in range(n):
        for y in range(n):
            producto = f_X[x] * f_Y[y]
            diferencia = f[x, y] - producto
            igual = math.isclose(f[x, y], producto, abs_tol=tolerancia)
            if not igual:
                todas_iguales = False
            marca = "✓" if igual else "✗"
            print(f"  ({x},{y}):  {f[x,y]:>8.4f}  {producto:>8.4f}  {diferencia:>12.6f}  {marca:>8}")

    print()
    if todas_iguales:
        print("  ➜ CONCLUSIÓN: X e Y son INDEPENDIENTES ✓")
    else:
        print("  ➜ CONCLUSIÓN: X e Y NO son independientes ✗")
        print("  (La condición falla en al menos una celda)")

    return todas_iguales



#  8. FUNCIÓN PRINCIPAL


def main():
    print("\n" + "#" * 54)
    print("#   PROBABILIDAD CONJUNTA DISCRETA                    #")
    print("#   Control de Calidad -- Variables X e Y             #")
    print("#" * 54)

    # --- Crear distribución ---
    f = crear_distribucion_conjunta()
    imprimir_tabla_conjunta(f)

    # --- Normalización ---
    verificar_normalizacion(f)

    # --- Probabilidades específicas ---
    p1, p2, p3 = calcular_probabilidades_especificas(f)

    # --- Marginales ---
    f_X, f_Y = calcular_marginales(f)

    # --- Esperanza ---
    E_X, E_Y, E_XY = calcular_esperanza(f, f_X, f_Y)

    # --- Varianza y covarianza ---
    Var_X, Var_Y, Cov, rho = calcular_varianza_covarianza(f, f_X, f_Y, E_X, E_Y, E_XY)

    # --- Independencia ---
    independientes = verificar_independencia(f, f_X, f_Y)

    # --- Resumen final ---
    print("\n" + "=" * 54)
    print("  RESUMEN FINAL DE RESULTADOS")
    print("=" * 54)
    print(f"\n  E(X)              = {E_X:.4f}")
    print(f"  E(Y)              = {E_Y:.4f}")
    print(f"  E(X+Y)            = {E_X + E_Y:.4f}")
    print(f"  Var(X)            = {Var_X:.4f}")
    print(f"  Var(Y)            = {Var_Y:.4f}")
    print(f"  Cov(X,Y)          = {Cov:.4f}")
    print(f"  ρ(X,Y)            = {rho:.6f}")
    print(f"  P(X+Y ≤ 2)        = {p1:.4f}")
    print(f"  P(X = Y)          = {p2:.4f}")
    print(f"  P(X≥1, Y≥1)       = {p3:.4f}")
    print(f"  Independientes     = {'Sí' if independientes else 'No'}")
    print("\n" + "=" * 54 + "\n")


if __name__ == "__main__":
    main()
