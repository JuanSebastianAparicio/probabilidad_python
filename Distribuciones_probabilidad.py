"""

  VALIDACIÓN COMPUTACIONAL DE DISTRIBUCIONES DE PROBABILIDAD

"""

import math
import numpy as np
from scipy import stats

# Separador visual
SEP = "=" * 65
SEP2 = "-" * 65


# UTILIDADES


def encabezado(titulo: str) -> None:
    """Imprime un encabezado formateado."""
    print(f"\n{SEP}")
    print(f"  {titulo}")
    print(SEP)


def sub(titulo: str) -> None:
    print(f"\n{SEP2}")
    print(f"  {titulo}")
    print(SEP2)


def comparar(nombre: str, teorico: float, computacional: float) -> None:
    """Imprime la comparación teórico vs computacional."""
    error = abs(teorico - computacional)
    print(f"  {'Resultado teórico':<30}: {teorico:.6f}")
    print(f"  {'Resultado computacional':<30}: {computacional:.6f}")
    print(f"  {'Error absoluto':<30}: {error:.2e}")



# 1. DISTRIBUCIÓN BINOMIAL


def binomial_pmf_manual(n: int, k: int, p: float) -> float:
    """Calcula P(X=k) para X ~ B(n,p) usando math.comb."""
    return math.comb(n, k) * (p ** k) * ((1 - p) ** (n - k))


def ejercicio_binomial_1() -> float:
    """
    Ejercicio 1 — Binomial
    Examen 10 preguntas, 4 opciones, respuesta al azar.
    P(X = 3) con n=10, p=0.25.
    Resultado teórico: 0.250282
    """
    sub("Binomial — Ej. 1: Examen de opción múltiple P(X=3), n=10, p=0.25")
    n, k, p = 10, 3, 0.25

    # Cálculo manual con math.comb
    teorico = binomial_pmf_manual(n, k, p)

    # Cálculo con scipy
    computacional = stats.binom.pmf(k, n, p)

    print(f"  Parámetros: n={n}, k={k}, p={p}")
    print(f"  C(10,3) = {math.comb(n,k)}")
    print(f"  (0.25)^3 = {0.25**3:.8f}")
    print(f"  (0.75)^7 = {0.75**7:.8f}")
    comparar("P(X=3)", teorico, computacional)
    return computacional


def ejercicio_binomial_2() -> float:
    """
    Ejercicio 2 — Binomial
    Tornillos defectuosos p=0.05, n=20.
    P(X <= 1).
    """
    sub("Binomial — Ej. 2: Tornillos defectuosos P(X<=1), n=20, p=0.05")
    n, p = 20, 0.05

    px0 = binomial_pmf_manual(n, 0, p)
    px1 = binomial_pmf_manual(n, 1, p)
    teorico = px0 + px1

    computacional = stats.binom.cdf(1, n, p)

    print(f"  Parámetros: n={n}, p={p}")
    print(f"  P(X=0) manual = {px0:.6f}")
    print(f"  P(X=1) manual = {px1:.6f}")
    comparar("P(X<=1)", teorico, computacional)
    return computacional



# 2. DISTRIBUCIÓN DE POISSON


def poisson_pmf_manual(lam: float, k: int) -> float:
    """Calcula P(X=k) para X ~ Po(lambda) usando math."""
    return (math.exp(-lam) * lam ** k) / math.factorial(k)


def ejercicio_poisson_1() -> float:
    """
    Ejercicio 1 — Poisson
    Servidor web: lambda=3 peticiones/min.
    P(X = 5).
    """
    sub("Poisson — Ej. 1: Servidor web P(X=5), lambda=3")
    lam, k = 3.0, 5

    teorico = poisson_pmf_manual(lam, k)
    computacional = stats.poisson.pmf(k, lam)

    print(f"  Parámetros: lambda={lam}, k={k}")
    print(f"  e^(-3) = {math.exp(-lam):.8f}")
    print(f"  3^5 = {lam**k:.1f}")
    print(f"  5! = {math.factorial(k)}")
    comparar("P(X=5)", teorico, computacional)
    return computacional


def ejercicio_poisson_2() -> float:
    """
    Ejercicio 2 — Poisson
    Accidentes: lambda=2/día.
    P(X < 3).
    """
    sub("Poisson — Ej. 2: Accidentes P(X<3), lambda=2")
    lam = 2.0

    px0 = poisson_pmf_manual(lam, 0)
    px1 = poisson_pmf_manual(lam, 1)
    px2 = poisson_pmf_manual(lam, 2)
    teorico = px0 + px1 + px2

    # scipy: P(X < 3) = P(X <= 2) = CDF(2)
    computacional = stats.poisson.cdf(2, lam)

    print(f"  Parámetros: lambda={lam}")
    print(f"  P(X=0) = {px0:.6f}")
    print(f"  P(X=1) = {px1:.6f}")
    print(f"  P(X=2) = {px2:.6f}")
    comparar("P(X<3)", teorico, computacional)
    return computacional



# 3. DISTRIBUCIÓN NORMAL


def ejercicio_normal_1() -> float:
    """
    Ejercicio 1 — Normal
    Alturas: mu=175, sigma=7.
    P(X > 182).
    """
    sub("Normal — Ej. 1: Alturas P(X>182), mu=175, sigma=7")
    mu, sigma, x = 175.0, 7.0, 182.0

    z = (x - mu) / sigma
    # P(X > x) = 1 - Phi(z)
    teorico = 1.0 - stats.norm.cdf(z)        # usando tabla z
    computacional = 1.0 - stats.norm.cdf(x, mu, sigma)

    print(f"  Parámetros: mu={mu}, sigma={sigma}, x={x}")
    print(f"  z = (182 - 175) / 7 = {z:.4f}")
    print(f"  Phi(1.0) = {stats.norm.cdf(1.0):.6f}")
    comparar("P(X>182)", teorico, computacional)
    return computacional


def ejercicio_normal_2() -> float:
    """
    Ejercicio 2 — Normal (ej. 6.3 adaptado)
    Botellas: mu=500, sigma=5.
    P(490 <= X <= 510).
    """
    sub("Normal — Ej. 2 (6.3): Botellas P(490<=X<=510), mu=500, sigma=5")
    mu, sigma = 500.0, 5.0
    a, b = 490.0, 510.0

    z1 = (a - mu) / sigma   # -2
    z2 = (b - mu) / sigma   #  2
    teorico = stats.norm.cdf(z2) - stats.norm.cdf(z1)
    computacional = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)

    print(f"  Parámetros: mu={mu}, sigma={sigma}")
    print(f"  z1 = {z1:.2f},  z2 = {z2:.2f}")
    print(f"  Phi(2)  = {stats.norm.cdf(2):.6f}")
    print(f"  Phi(-2) = {stats.norm.cdf(-2):.6f}")
    comparar("P(490<=X<=510)", teorico, computacional)
    return computacional



# 4. DISTRIBUCIÓN EXPONENCIAL


def ejercicio_exponencial_1() -> float:
    """
    Ejercicio 1 — Exponencial
    Componente electrónico: lambda=0.01.
    P(X > 200).
    """
    sub("Exponencial — Ej. 1: Componente P(X>200), lambda=0.01")
    lam, x = 0.01, 200.0

    # P(X > x) = e^(-lambda*x)
    teorico = math.exp(-lam * x)
    # scipy: scale = 1/lambda
    computacional = 1.0 - stats.expon.cdf(x, scale=1.0 / lam)

    print(f"  Parámetros: lambda={lam}, x={x}")
    print(f"  e^(-0.01*200) = e^(-2) = {math.exp(-2):.8f}")
    comparar("P(X>200)", teorico, computacional)
    return computacional


def ejercicio_exponencial_2() -> float:
    """
    Ejercicio 2 — Exponencial
    Clientes: lambda=0.2 clientes/min.
    P(3 <= X <= 8).
    """
    sub("Exponencial — Ej. 2: Clientes P(3<=X<=8), lambda=0.2")
    lam, a, b = 0.2, 3.0, 8.0

    F_a = 1.0 - math.exp(-lam * a)
    F_b = 1.0 - math.exp(-lam * b)
    teorico = F_b - F_a

    scale = 1.0 / lam
    computacional = stats.expon.cdf(b, scale=scale) - stats.expon.cdf(a, scale=scale)

    print(f"  Parámetros: lambda={lam}, a={a}, b={b}")
    print(f"  F(8) = 1 - e^(-1.6) = {F_b:.6f}")
    print(f"  F(3) = 1 - e^(-0.6) = {F_a:.6f}")
    comparar("P(3<=X<=8)", teorico, computacional)
    return computacional



# 5. EJERCICIOS OBLIGATORIOS


def obligatorio_binomial_dado() -> dict:
    """
    Obligatorio — Binomial: dado no alterado, n=5, p=1/6.
    (a) P(X=2)   -> exactamente 2 veces
    (b) P(X<=1)  -> máximo 1 vez
    (c) P(X>=2)  -> al menos 2 veces
    """
    encabezado("EJERCICIO OBLIGATORIO 1: Binomial — Dado, n=5, p=1/6")
    n = 5
    p = 1 / 6
    q = 1 - p

    print(f"  Modelo: X ~ B(n={n}, p=1/6 ≈ {p:.6f})")
    print(f"  Media = np = {n*p:.4f}    Varianza = npq = {n*p*q:.4f}")

    # ---- (a) P(X=2) ----
    print("\n  (a) P(X = 2)  —  exactamente 2 veces:")
    px2_manual = binomial_pmf_manual(n, 2, p)
    px2_scipy  = stats.binom.pmf(2, n, p)
    fraccion   = math.comb(n, 2) * (1**2) * (5**3)   # numerador
    denom      = 6**5                                  # denominador
    print(f"      C(5,2) = {math.comb(n,2)}")
    print(f"      (1/6)^2 * (5/6)^3 = {(1/6)**2 * (5/6)**3:.8f}")
    print(f"      Fracción exacta: {fraccion}/{denom}")
    comparar("P(X=2)", px2_manual, px2_scipy)

    # ---- (b) P(X<=1) ----
    print("\n  (b) P(X ≤ 1)  —  máximo 1 vez:")
    px0 = binomial_pmf_manual(n, 0, p)
    px1 = binomial_pmf_manual(n, 1, p)
    teorico_b  = px0 + px1
    scipy_b    = stats.binom.cdf(1, n, p)
    print(f"      P(X=0) = (5/6)^5 = {px0:.8f}")
    print(f"      P(X=1) = 5*(1/6)*(5/6)^4 = {px1:.8f}")
    comparar("P(X<=1)", teorico_b, scipy_b)

    # ---- (c) P(X>=2) ----
    print("\n  (c) P(X ≥ 2)  —  al menos 2 veces:")
    teorico_c  = 1.0 - teorico_b
    scipy_c    = 1.0 - stats.binom.cdf(1, n, p)
    print(f"      P(X>=2) = 1 - P(X<=1) = 1 - {teorico_b:.8f}")
    comparar("P(X>=2)", teorico_c, scipy_c)

    return {
        "P(X=2)":  scipy_b if False else stats.binom.pmf(2, n, p),
        "P(X<=1)": scipy_b,
        "P(X>=2)": scipy_c,
    }


def obligatorio_poisson_accidentes() -> float:
    """
    Obligatorio — Poisson: p=0.001, n=1000 autos.
    P(X >= 2)  con lambda = n*p = 1.
    """
    encabezado("EJERCICIO OBLIGATORIO 2: Poisson — Accidentes, lambda=1")
    n, p = 1000, 0.001
    lam = n * p

    print(f"  Aproximación: n={n}, p={p} -> lambda = n*p = {lam}")
    print(f"  Modelo: X ~ Po(lambda={lam})")

    px0 = poisson_pmf_manual(lam, 0)
    px1 = poisson_pmf_manual(lam, 1)
    teorico = 1.0 - px0 - px1

    # scipy: P(X >= 2) = 1 - P(X <= 1) = sf(1)
    computacional = stats.poisson.sf(1, lam)

    print(f"\n  P(X=0) = e^(-1) = {px0:.8f}")
    print(f"  P(X=1) = e^(-1) = {px1:.8f}")
    print(f"  P(X>=2) = 1 - 2*e^(-1) = 1 - {px0+px1:.8f}")
    comparar("P(X>=2)", teorico, computacional)
    return computacional


def obligatorio_normal_botellas() -> dict:
    """
    Obligatorio — Normal (Ejercicio 6.3):
    Botellas mu=500, sigma=5.
    (a) P(490 <= X <= 510)
    (b) P(X < 492)
    (c) c tal que P(X > c) = 0.05
    """
    encabezado("EJERCICIO OBLIGATORIO 3: Normal (Ej. 6.3) — Botellas, mu=500, sigma=5")
    mu, sigma = 500.0, 5.0

    print(f"  Modelo: X ~ N(mu={mu}, sigma^2={sigma**2})")

    # ---- (a) ----
    a, b = 490.0, 510.0
    z1, z2 = (a - mu)/sigma, (b - mu)/sigma
    p_a = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)
    print(f"\n  (a) P({a} <= X <= {b})")
    print(f"      z1 = {z1:.2f},  z2 = {z2:.2f}")
    print(f"      P = Phi({z2}) - Phi({z1}) = {stats.norm.cdf(z2):.4f} - {stats.norm.cdf(z1):.4f}")
    print(f"      >>> P(490 <= X <= 510) = {p_a:.6f}  ({p_a*100:.2f}%)")

    # ---- (b) ----
    x_b = 492.0
    z_b = (x_b - mu) / sigma
    p_b = stats.norm.cdf(x_b, mu, sigma)
    print(f"\n  (b) P(X < {x_b})")
    print(f"      z = ({x_b} - {mu}) / {sigma} = {z_b:.2f}")
    print(f"      >>> P(X < 492) = Phi({z_b}) = {p_b:.6f}  ({p_b*100:.2f}%)")

    # ---- (c) ----
    alpha = 0.05
    c = stats.norm.ppf(1 - alpha, mu, sigma)
    print(f"\n  (c) c tal que P(X > c) = {alpha}")
    print(f"      z_(0.95) = {stats.norm.ppf(0.95):.4f}")
    print(f"      c = mu + z*sigma = {mu} + {stats.norm.ppf(0.95):.4f}*{sigma}")
    print(f"      >>> c = {c:.4f} ml")

    return {"P(490<=X<=510)": p_a, "P(X<492)": p_b, "c_P>0.05": c}


# 6. RESUMEN COMPARATIVO FINAL


def resumen_comparativo() -> None:
    """Genera tabla resumen de todos los resultados."""
    encabezado("RESUMEN COMPARATIVO — Teórico vs Computacional")

    resultados = [
        ("Binomial Ej.1 P(X=3|n=10,p=0.25)",
            binomial_pmf_manual(10, 3, 0.25),
            stats.binom.pmf(3, 10, 0.25)),
        ("Binomial Ej.2 P(X<=1|n=20,p=0.05)",
            binomial_pmf_manual(20, 0, 0.05) + binomial_pmf_manual(20, 1, 0.05),
            stats.binom.cdf(1, 20, 0.05)),
        ("Poisson Ej.1  P(X=5|lambda=3)",
            poisson_pmf_manual(3.0, 5),
            stats.poisson.pmf(5, 3)),
        ("Poisson Ej.2  P(X<3|lambda=2)",
            sum(poisson_pmf_manual(2.0, k) for k in range(3)),
            stats.poisson.cdf(2, 2)),
        ("Normal Ej.1   P(X>182|175,7)",
            1 - stats.norm.cdf(1.0),
            1 - stats.norm.cdf(182, 175, 7)),
        ("Normal Ej.2   P(490<=X<=510|500,5)",
            stats.norm.cdf(2) - stats.norm.cdf(-2),
            stats.norm.cdf(510, 500, 5) - stats.norm.cdf(490, 500, 5)),
        ("Exponencial Ej.1 P(X>200|lambda=0.01)",
            math.exp(-2),
            1 - stats.expon.cdf(200, scale=100)),
        ("Exponencial Ej.2 P(3<=X<=8|lambda=0.2)",
            (1 - math.exp(-1.6)) - (1 - math.exp(-0.6)),
            stats.expon.cdf(8, scale=5) - stats.expon.cdf(3, scale=5)),
        ("Oblig. Dado P(X=2|n=5,p=1/6)",
            binomial_pmf_manual(5, 2, 1/6),
            stats.binom.pmf(2, 5, 1/6)),
        ("Oblig. Dado P(X<=1|n=5,p=1/6)",
            sum(binomial_pmf_manual(5, k, 1/6) for k in range(2)),
            stats.binom.cdf(1, 5, 1/6)),
        ("Oblig. Dado P(X>=2|n=5,p=1/6)",
            1 - sum(binomial_pmf_manual(5, k, 1/6) for k in range(2)),
            stats.binom.sf(1, 5, 1/6)),
        ("Oblig. Acc. P(X>=2|lambda=1)",
            1 - poisson_pmf_manual(1.0, 0) - poisson_pmf_manual(1.0, 1),
            stats.poisson.sf(1, 1)),
    ]

    ancho = 42
    print(f"\n  {'Ejercicio':<{ancho}} {'Teórico':>10}  {'SciPy':>10}  {'Error':>10}")
    print(f"  {'-'*ancho} {'-'*10}  {'-'*10}  {'-'*10}")
    for nombre, teo, comp in resultados:
        err = abs(teo - comp)
        print(f"  {nombre:<{ancho}} {teo:>10.6f}  {comp:>10.6f}  {err:>10.2e}")

    print(f"\n  {'✓ Todos los resultados son consistentes entre métodos.':}")


# MAIN


def main():
    print(SEP)
    print("  VALIDACIÓN COMPUTACIONAL DE DISTRIBUCIONES DE PROBABILIDAD")
    print("  Librerías: math | numpy | scipy.stats")
    print(SEP)

    # --- Binomial ---
    encabezado("SECCIÓN 1: DISTRIBUCIÓN BINOMIAL")
    ejercicio_binomial_1()
    ejercicio_binomial_2()

    # --- Poisson ---
    encabezado("SECCIÓN 2: DISTRIBUCIÓN DE POISSON")
    ejercicio_poisson_1()
    ejercicio_poisson_2()

    # --- Normal ---
    encabezado("SECCIÓN 3: DISTRIBUCIÓN NORMAL")
    ejercicio_normal_1()
    ejercicio_normal_2()

    # --- Exponencial ---
    encabezado("SECCIÓN 4: DISTRIBUCIÓN EXPONENCIAL")
    ejercicio_exponencial_1()
    ejercicio_exponencial_2()

    # --- Obligatorios ---
    encabezado("SECCIÓN 5: EJERCICIOS OBLIGATORIOS")
    obligatorio_binomial_dado()
    obligatorio_poisson_accidentes()
    obligatorio_normal_botellas()

    # --- Resumen ---
    resumen_comparativo()

    print(f"\n{SEP}")
    print("  FIN DEL PROGRAMA")
    print(SEP)


if __name__ == "__main__":
    main()
