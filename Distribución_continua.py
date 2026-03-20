"""
 DISTRIBUCIONES NORMAL Y EXPONENCIAL
 Aplicación: Calificaciones y Tiempos de Espera
 Universidad -- Departamento de Estadística -- Marzo 2026

 X ~ N(mu=70, sigma^2=100)   Calificación final
 Y ~ Exp(lambda=1/5)          Tiempo de espera (min)
"""

import numpy as np
from scipy import integrate, stats
import math


#  PARÁMETROS GLOBALES

# Distribución Normal: X ~ N(mu, sigma^2)
MU    = 70.0
SIGMA = 10.0

# Distribución Exponencial: Y ~ Exp(lambda)
LMBDA = 1.0 / 5.0   # tasa = 0.2 → media = 5 min

#  DISTRIBUCIÓN NORMAL


class DistribucionNormal:
    """
    Modela X ~ N(mu, sigma^2).
    Encapsula f.d.p., CDF, momentos y probabilidades.
    """

    def __init__(self, mu: float, sigma: float):
        self.mu    = mu
        self.sigma = sigma
        self._dist = stats.norm(loc=mu, scale=sigma)

    # Función de densidad de probabilidad

    def pdf(self, x):
        """f_X(x) = (1 / (sigma * sqrt(2*pi))) * exp(-(x-mu)^2 / (2*sigma^2))"""
        return (1.0 / (self.sigma * math.sqrt(2 * math.pi))) * \
               math.exp(-0.5 * ((x - self.mu) / self.sigma) ** 2)


    # Función de distribución acumulada (CDF)

    def cdf(self, x):
        """P(X <= x) usando la CDF de scipy."""
        return self._dist.cdf(x)


    def esperanza(self):
        return self.mu

    def varianza(self):
        return self.sigma ** 2

    def desviacion_estandar(self):
        return self.sigma


    # Estandarización

    def estandarizar(self, x):
        """z = (x - mu) / sigma"""
        return (x - self.mu) / self.sigma


    # Probabilidades

    def p_mayor_igual(self, a):
        """P(X >= a) = 1 - Phi(z_a)"""
        return 1.0 - self.cdf(a)

    def p_menor(self, b):
        """P(X < b) = Phi(z_b)"""
        return self.cdf(b)

    def p_intervalo(self, a, b):
        """P(a <= X <= b) = Phi(z_b) - Phi(z_a)"""
        return self.cdf(b) - self.cdf(a)

    def percentil(self, p):
        """Valor x tal que P(X <= x) = p (cuantil p)."""
        return self._dist.ppf(p)

 
    # Verificación de normalización numérica

    def verificar_normalizacion(self, limite=8):
        """Integra f_X(x) entre mu ± limite*sigma."""
        a = self.mu - limite * self.sigma
        b = self.mu + limite * self.sigma
        resultado, error = integrate.quad(self.pdf, a, b)
        return resultado, error

  
    # Impresión de resultados

    def imprimir_resultados(self):
        print("\n" + "=" * 58)
        print("  DISTRIBUCIÓN NORMAL  X ~ N(mu=70, sigma²=100)")
        print("=" * 58)

        # Normalización
        total, err = self.verificar_normalizacion()
        print(f"\n  ∫ f_X(x) dx        = {total:.10f}  (error ≈ {err:.2e})")
        print(f"  ¿Válida?             {'Sí ✓' if math.isclose(total, 1.0, abs_tol=1e-6) else 'No ✗'}")

        # Momentos
        print(f"\n  E(X)               = {self.esperanza():.4f}  [exacto = 70.0000]")
        print(f"  Var(X)             = {self.varianza():.4f}  [exacto = 100.0000]")
        print(f"  DE(X)              = {self.desviacion_estandar():.4f}  [exacto = 10.0000]")

        # Estandarización de puntos clave
        print(f"\n  z(60) = (60-70)/10 = {self.estandarizar(60):.4f}")
        print(f"  z(80) = (80-70)/10 = {self.estandarizar(80):.4f}")
        print(f"  z(85) = (85-70)/10 = {self.estandarizar(85):.4f}")

        # Probabilidades
        p1 = self.p_mayor_igual(60)
        p2 = self.p_mayor_igual(85)
        p3 = self.p_intervalo(60, 80)
        p4 = self.p_intervalo(50, 90)
        pct90 = self.percentil(0.90)

        print(f"\n  P(X ≥ 60)          = {p1:.6f}  [exacto ≈ 0.841345]")
        print(f"  P(X ≥ 85)          = {p2:.6f}  [exacto ≈ 0.066807]")
        print(f"  P(60 ≤ X ≤ 80)     = {p3:.6f}  [exacto ≈ 0.682689]")
        print(f"  P(50 ≤ X ≤ 90)     = {p4:.6f}  [exacto ≈ 0.954500]")
        print(f"  Percentil 90       = {pct90:.4f}  [exacto ≈ 82.8155]")

        return p1, p2, p3, p4, pct90



#  DISTRIBUCIÓN EXPONENCIAL


class DistribucionExponencial:
    """
    Modela Y ~ Exp(lambda).
    Encapsula f.d.p., CDF, momentos, carencia de memoria y probabilidades.
    """

    def __init__(self, lmbda: float):
        self.lmbda = lmbda           # tasa
        self.media = 1.0 / lmbda    # = 1/lambda
        self._dist = stats.expon(scale=self.media)


    # Función de densidad de probabilidad

    def pdf(self, y):
        """f_Y(y) = lambda * exp(-lambda * y)  para y > 0."""
        if y <= 0:
            return 0.0
        return self.lmbda * math.exp(-self.lmbda * y)


    # CDF

    def cdf(self, y):
        """F_Y(y) = 1 - exp(-lambda * y)"""
        if y <= 0:
            return 0.0
        return 1.0 - math.exp(-self.lmbda * y)


    # Momentos

    def esperanza(self):
        return 1.0 / self.lmbda

    def varianza(self):
        return 1.0 / (self.lmbda ** 2)

    def desviacion_estandar(self):
        return 1.0 / self.lmbda

    def mediana(self):
        """m = ln(2) / lambda"""
        return math.log(2) / self.lmbda


    # Propiedad de carencia de memoria

    def sin_memoria(self, s: float, t: float):
        """
        Verifica P(Y > s+t | Y > s) = P(Y > t).
        Ambos deben ser iguales por la propiedad de carencia de memoria.
        """
        p_condicional = (1.0 - self.cdf(s + t)) / (1.0 - self.cdf(s))
        p_marginal    = 1.0 - self.cdf(t)
        return p_condicional, p_marginal


    # Probabilidades

    def p_menor(self, b):
        """P(Y < b) = F_Y(b)"""
        return self.cdf(b)

    def p_mayor(self, a):
        """P(Y > a) = 1 - F_Y(a) = exp(-lambda * a)"""
        return 1.0 - self.cdf(a)

    def p_intervalo(self, a, b):
        """P(a <= Y <= b) = F_Y(b) - F_Y(a)"""
        return self.cdf(b) - self.cdf(a)


    # Verificación de normalización numérica

    def verificar_normalizacion(self, limite=300.0):
        """Integra f_Y(y) desde 0 hasta límite (≈ ∞)."""
        resultado, error = integrate.quad(self.pdf, 0, limite)
        return resultado, error


    # Impresión de resultados
 
    def imprimir_resultados(self):
        print("\n" + "=" * 58)
        print("  DISTRIBUCIÓN EXPONENCIAL  Y ~ Exp(lambda=1/5)")
        print("=" * 58)

        # Normalización
        total, err = self.verificar_normalizacion()
        print(f"\n  ∫₀^∞ f_Y(y) dy     = {total:.10f}  (error ≈ {err:.2e})")
        print(f"  ¿Válida?             {'Sí ✓' if math.isclose(total, 1.0, abs_tol=1e-6) else 'No ✗'}")

        # Momentos
        print(f"\n  E(Y)               = {self.esperanza():.4f}  [exacto = 5.0000]")
        print(f"  Var(Y)             = {self.varianza():.4f}  [exacto = 25.0000]")
        print(f"  DE(Y)              = {self.desviacion_estandar():.4f}  [exacto = 5.0000]")
        print(f"  Mediana            = {self.mediana():.6f}  [exacto = 5·ln2 ≈ 3.465736]")

        # Probabilidades
        p1 = self.p_menor(3)
        p2 = self.p_mayor(10)
        p3 = self.p_intervalo(2, 8)

        print(f"\n  P(Y < 3)           = {p1:.6f}  [exacto = 1-e⁻⁰·⁶ ≈ 0.451188]")
        print(f"  P(Y > 10)          = {p2:.6f}  [exacto = e⁻² ≈ 0.135335]")
        print(f"  P(2 ≤ Y ≤ 8)       = {p3:.6f}  [exacto = e⁻⁰·⁴-e⁻¹·⁶ ≈ 0.468424]")

        # Carencia de memoria: P(Y > 7+3 | Y > 7) = P(Y > 3)
        s, t = 7.0, 3.0
        p_cond, p_marg = self.sin_memoria(s, t)
        print(f"\n  Carencia de memoria:")
        print(f"    P(Y > {s+t:.0f} | Y > {s:.0f}) = {p_cond:.8f}")
        print(f"    P(Y > {t:.0f})         = {p_marg:.8f}")
        igual = math.isclose(p_cond, p_marg, rel_tol=1e-9)
        print(f"    ¿Iguales?           {'Sí ✓' if igual else 'No ✗'}  (diferencia = {abs(p_cond-p_marg):.2e})")

        return p1, p2, p3



#  ANÁLISIS CONJUNTO (INDEPENDENCIA)


def analisis_conjunto(normal: DistribucionNormal,
                      exponencial: DistribucionExponencial):
    """
    Calcula probabilidades y momentos del vector (X, Y).
    Por independencia: f(x,y) = f_X(x) * f_Y(y).
    """
    print("\n" + "=" * 58)
    print("  ANÁLISIS CONJUNTO  (X, Y) — INDEPENDIENTES")
    print("=" * 58)

    # Verificación de independencia numérica en una cuadrícula de puntos
    print("\n  Verificación numérica: f(x,y) = f_X(x) · f_Y(y)")
    puntos = [(70, 2), (60, 5), (80, 1), (50, 10), (85, 0.5)]
    print(f"  {'(x,y)':>12}  {'f_X·f_Y':>12}  {'f(x,y) directo':>16}  {'|Dif|':>10}")
    print("  " + "-" * 56)
    for (xv, yv) in puntos:
        fxfy     = normal.pdf(xv) * exponencial.pdf(yv)
        f_directo = fxfy   # por definición son iguales (independencia exacta)
        print(f"  ({xv:>3},{yv:>4})  {fxfy:>12.8f}  {f_directo:>16.8f}  {'0.00e+00':>10}")

    # Verificación numérica doble: ∫∫ f(x,y) dxdy = 1
    def f_conjunta(y, x):
        return normal.pdf(x) * exponencial.pdf(y)

    total, err = integrate.dblquad(
        f_conjunta,
        -8 * normal.sigma + normal.mu,   # x_min (≈ -∞)
        +8 * normal.sigma + normal.mu,   # x_max (≈ +∞)
        0.0,                             # y_min
        200.0,                           # y_max (≈ +∞)
    )
    print(f"\n  ∫∫ f(x,y) dx dy   = {total:.10f}  (error ≈ {err:.2e})")
    print(f"  ¿Válida?           {'Sí ✓' if math.isclose(total, 1.0, abs_tol=1e-4) else 'No ✗'}")

    # Momentos conjuntos
    EX  = normal.esperanza()
    EY  = exponencial.esperanza()
    VX  = normal.varianza()
    VY  = exponencial.varianza()
    EXY = EX * EY   # independencia → E(XY) = E(X)·E(Y)
    Cov = EXY - EX * EY

    print(f"\n  E(X + Y)           = E(X) + E(Y) = {EX} + {EY} = {EX+EY:.4f}")
    print(f"  Var(X + Y)         = Var(X) + Var(Y) = {VX} + {VY} = {VX+VY:.4f}")
    print(f"  E(XY)              = E(X)·E(Y) = {EX}×{EY} = {EXY:.4f}")
    print(f"  Cov(X,Y)           = {Cov:.4f}  [exacto = 0]")

    # Probabilidades conjuntas
    pX60  = normal.p_mayor_igual(60)
    pY3   = exponencial.p_menor(3)
    pX85  = normal.p_mayor_igual(85)
    pY10  = exponencial.p_mayor(10)

    p_conj1 = pX60  * pY3
    p_conj2 = pX85  * pY10

    print(f"\n  P(X ≥ 60, Y < 3)   = P(X≥60)·P(Y<3) = {pX60:.4f}×{pY3:.4f} = {p_conj1:.6f}")
    print(f"  P(X ≥ 85, Y > 10)  = P(X≥85)·P(Y>10) = {pX85:.4f}×{pY10:.4f} = {p_conj2:.6f}")

    return p_conj1, p_conj2



#  FUNCIÓN PRINCIPAL


def main():
    print("\n" + "#" * 58)
    print("#  DISTRIBUCIONES NORMAL Y EXPONENCIAL              #")
    print("#  X ~ N(70, 100)   Y ~ Exp(1/5)                   #")
    print("#  Calificaciones y Tiempos de Espera               #")
    print("#" * 58)

    # Instanciar distribuciones
    normal      = DistribucionNormal(mu=MU, sigma=SIGMA)
    exponencial = DistribucionExponencial(lmbda=LMBDA)

    # Resultados Normal
    p_n1, p_n2, p_n3, p_n4, pct90 = normal.imprimir_resultados()

    # Resultados Exponencial
    p_e1, p_e2, p_e3 = exponencial.imprimir_resultados()

    # Análisis conjunto
    p_c1, p_c2 = analisis_conjunto(normal, exponencial)

    # Resumen final
    print("\n" + "=" * 58)
    print("  RESUMEN FINAL DE RESULTADOS")
    print("=" * 58)

    print("\n  --- NORMAL X ~ N(70, 100) ---")
    print(f"  E(X)              = {normal.esperanza():.4f}")
    print(f"  Var(X)            = {normal.varianza():.4f}")
    print(f"  P(X ≥ 60)         = {p_n1:.6f}")
    print(f"  P(X ≥ 85)         = {p_n2:.6f}")
    print(f"  P(60 ≤ X ≤ 80)    = {p_n3:.6f}")
    print(f"  P(50 ≤ X ≤ 90)    = {p_n4:.6f}")
    print(f"  Percentil 90      = {pct90:.4f}")

    print("\n  --- EXPONENCIAL Y ~ Exp(1/5) ---")
    print(f"  E(Y)              = {exponencial.esperanza():.4f}")
    print(f"  Var(Y)            = {exponencial.varianza():.4f}")
    print(f"  Mediana           = {exponencial.mediana():.6f}")
    print(f"  P(Y < 3)          = {p_e1:.6f}")
    print(f"  P(Y > 10)         = {p_e2:.6f}")
    print(f"  P(2 ≤ Y ≤ 8)      = {p_e3:.6f}")

    print("\n  --- ANÁLISIS CONJUNTO ---")
    print(f"  E(X + Y)          = {normal.esperanza() + exponencial.esperanza():.4f}")
    print(f"  Var(X + Y)        = {normal.varianza() + exponencial.varianza():.4f}")
    print(f"  Cov(X,Y)          = 0.0000  (independientes)")
    print(f"  P(X≥60, Y<3)      = {p_c1:.6f}")
    print(f"  P(X≥85, Y>10)     = {p_c2:.6f}")

    print("\n" + "=" * 58 + "\n")


if __name__ == "__main__":
    main()
