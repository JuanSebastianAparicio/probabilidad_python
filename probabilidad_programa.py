import math

def combinacion(n, r):
    return math.comb(n, r)

def variacion(n, r):
    return math.perm(n, r)

def permutacion(n):
    return math.factorial(n)

def prob_binomial(n, r, p):
    return math.comb(n, r) * (p ** r) * ((1 - p) ** (n - r))

def main():
    while True:
        print("\n=== SISTEMA DE PROBABILIDAD Y CONTEO ===")
        print("1. Combinaciones C(n,r)")
        print("2. Variaciones V(n,r)")
        print("3. Permutaciones P(n)")
        print("4. Probabilidad Binomial")
        print("5. Probabilidad con reemplazo")
        print("6. Probabilidad sin reemplazo")
        print("7. Salir")

        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            n = int(input("Ingrese n: "))
            r = int(input("Ingrese r: "))
            print("Resultado:", combinacion(n, r))

        elif opcion == "2":
            n = int(input("Ingrese n: "))
            r = int(input("Ingrese r: "))
            print("Resultado:", variacion(n, r))

        elif opcion == "3":
            n = int(input("Ingrese n: "))
            print("Resultado:", permutacion(n))

        elif opcion == "4":
            n = int(input("Número de ensayos: "))
            r = int(input("Número de éxitos: "))
            p = float(input("Probabilidad de éxito (ej 0.03): "))
            print("Probabilidad:", prob_binomial(n, r, p))

        elif opcion == "5":
            print("Probabilidad con reemplazo")
            favorables = float(input("Cantidad favorable: "))
            total = float(input("Total elementos: "))
            veces = int(input("Número de selecciones: "))
            prob = (favorables / total) ** veces
            print("Probabilidad:", prob)

        elif opcion == "6":
            print("Probabilidad sin reemplazo")
            total = int(input("Total elementos: "))
            seleccion = int(input("Cantidad seleccionada: "))
            favorables = int(input("Favorables dentro del total: "))
            exitos = int(input("Cantidad favorable deseada: "))

            numerador = math.comb(favorables, exitos) * \
                        math.comb(total - favorables, seleccion - exitos)
            denominador = math.comb(total, seleccion)

            print("Probabilidad:", numerador / denominador)

        elif opcion == "7":
            print("Saliendo del sistema...")
            break

        else:
            print("Opción inválida")

if __name__ == "__main__":
    main()