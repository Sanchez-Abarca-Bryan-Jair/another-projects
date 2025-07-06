#Librerias
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import math

#Funciones (Evaluacion y derivadas)
def f(func, val):
    val = np.asarray(val).flatten()  # Asegura que val sea un vector 1D
    return func.subs({x: float(val[0]), y: float(val[1])})

#---------------------------------------------------------- ----------------------------------------------------------------

def f_grad(func): #derivada simbolica y evaluacion de la derivada en xk\
    grad_sim = [sp.diff(func,var) for var in (x,y)]
    return grad_sim
#--------------------------------------------------------------------------------------------------------------------------
# Graficacion
def graficar_contorno(func, trayectoria,x0,y0,puntop):
    trayectoria = np.array(trayectoria)
    x_vals = np.linspace(-10, 10, 600)
    y_vals = np.linspace(-10, 10, 600)
    X, Y = np.meshgrid(x_vals, y_vals)
    f_lamb = sp.lambdify((x, y), func, 'numpy')
    Z = f_lamb(X, Y)

    plt.figure(figsize=(8, 6))
    
    # Contorno con más niveles y mejor mapa de colores
    contour = plt.contour(X, Y, Z, levels=400, cmap='Spectral', alpha=0.4)
    plt.colorbar(contour, label='Valor de la función')
    
    # Trayectoria (con transparencia para no tapar puntos)
    plt.plot(trayectoria[:, 0], trayectoria[:, 1], 'r.-', 
             linewidth=2, markersize=6, alpha=0.7, label='Trayectoria')
    
    # Puntos inicial y óptimo MÁS GRANDES y con bordes
    plt.scatter(x0, y0, c='blue', s=200, edgecolor='black', 
                linewidth=2, label='Punto Inicial', zorder=5)
    plt.scatter(puntop[0], puntop[1], c='green', s=200, 
                edgecolor='black', linewidth=2, label='Punto Óptimo', zorder=5)
    
    # Añadir etiquetas a los puntos importantes
    plt.annotate('Inicio', (x0, y0), textcoords="offset points", 
                 xytext=(10,10), ha='center', fontsize=10)
    plt.annotate('Óptimo', (puntop[0], puntop[1]), textcoords="offset points", 
                 xytext=(10,10), ha='center', fontsize=10)
    
    plt.title("Convergencia", pad=20)
    plt.xlabel("$x$", fontsize=12)
    plt.ylabel("$y$", fontsize=12)
    plt.legend(loc='upper right', framealpha=1)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()
#0------------------------------------------------------------------------------------------------------------------------
#Busqueda en linea
def min_alpha_lineal(func, xk, P):
    xk0 = xk[0].item() # convierte el valor de xk en un numero
    xk1 = xk[1].item()
    P0 = P[0].item()
    P1 = P[1].item()

    new_func = func.subs({x: xk0 + a * P0, y: xk1 + a * P1})  # evaluación simbólica de f(xk + a*P)
    new_func = sp.simplify(new_func)

    df_dnew = sp.diff(new_func, a)  #derivada respecto a a
    df_dnew = sp.simplify(df_dnew)

    sols = sp.solve(df_dnew, a)    # Soluciona df/da = 0
    if sols:
        return float(sols[0])
    else:
        return 0.01  # o un valor por defecto

#--------------------------------------------------------------------------------------------------------------------------
#BACTRACKING

def minAlpha(xk,pk,alpha0,p,mu,max_iter,func, df_dx,df_dy):
    alpha = alpha0
    i = 0
    while i < max_iter:
        grad_alpha = np.array([
            f(df_dx,xk), #evaluacion de la derivada parcial respecto a x en xk
            f(df_dy,xk)  #evaluacion de la derivada parcial respecto a y en xk
        ], dtype=np.float64)

        f_new = f(func,xk+alpha*pk) #evaluacion de la funcion en xk+alpha*pk
        f_old = f(func,xk) #evaluacion de la funcion en xk
        val_grad = np.dot(grad_alpha,pk) #producto punto entre el gradiente y pk

        if f_new < f_old + mu*alpha*val_grad: #si la funcion es mayor a la funcion anterior + mu*alpha*val_grad, se multiplica alpha por p
            return alpha #retorna alpha
        alpha *= p
        i += 1
    return alpha

#--------------------------------------------------------------------------------------------------------------------------

#Gradient Descent
def grad_desc(tol,nIter,xk,func,df_dx,df_dy):
    i = 0
    trayectoria = [xk.copy()] #lista para guardar la trayectoria
    decimales = int(abs(math.log10(tol)))

    while i < nIter:
        pk = -np.array([
            f(df_dx,xk), #evaluacion de la derivada parcial respecto a x en xk
            f(df_dy,xk)  #evaluacion de la derivada parcial respecto a y en xk
        ], dtype=np.float64)

        norm = np.linalg.norm(pk) #norma del gradiente
        if norm < tol: #si la norma es menor a la tolerancia, se termina el algoritmo
            return xk, np.array(trayectoria)
        alpha = minAlpha(xk,pk,alpha0=2,p=0.7,mu=1e-4,max_iter=1000, func=func, df_dx=df_dx,df_dy=df_dy) #calculo de alpha
        if i < 100 or i > 900:
            print('-------------------------------------------------------------------------------------------------')
            print(f"| iteracion {i} "
                f"| xk: {np.round(np.array(xk, dtype=float), decimales)} "
                f"| norm: {round(float(norm), decimales)} "
                f"| pk: {np.round(np.array(pk, dtype=float), decimales)} "
                f"| alpha: {round(float(alpha), decimales)} |")
        xk = xk + alpha*pk
        trayectoria.append(xk.copy())
        i+=1
    return xk, np.array(trayectoria)
#-------------------------------------------------------------------------------------------------------------------------
#Newton
def newton(func,xk, tol, nIter, df_dx, df_dy):
    """
    sp.diff(func,var1,var2) #derivada parcial de la funcion respecto a x y y(primero x, luego y)
    for var1 in(x,y) itera por cada variable en x e y en las columnas, ciclo interior
    for var2 in(x,y) itera por cada variable en x e y en las filas, ciclo exterior
    """
    decimales = int(abs(math.log10(tol)))
    #Calculo de la matriz hessiana
    hessian = sp.Matrix([[sp.diff(func,var1,var2) for var1 in(x,y)]for var2 in(x,y)]) #matriz hessiana
    print("\nMatriz Hessiana:")
    sp.pprint(hessian) #imprime la matriz hessiana
    print("\n")
    trayectoria = [xk.copy()] #lista para guardar la trayectoria
    i = 0
    while i< nIter: #iteraciones maximas
        hessian_eval = hessian.subs({x: xk[0], y: xk[1]}) #evaluacion de la matriz hessiana en xk
        print("----------------------------------------------------------------------------------")
        try:
            hessian_inv = hessian_eval.inv() #inversa de la matriz hessiana
            print(f"\nMatriz Hessiana Inversa: en xk {round(float(xk[0]), decimales)}, {round(float(xk[1]), decimales)}")
            sp.pprint(hessian_inv)
            print("\n")
        except Exception as e:
            print("Error: {e}")

        phi_val = np.array([
                f(df_dx,xk), #evaluacion de la derivada parcial respecto a x en xk
                f(df_dy,xk)  #evaluacion de la derivada parcial respecto a y en xk
            ], dtype=np.float64)

        pk = -np.dot(hessian_inv,phi_val) #producto punto entre la inversa de la matriz hessiana y el gradiente
        alpha = minAlpha(xk,pk,alpha0=1,p=0.7,mu=1e-4,max_iter=1000, func=func, df_dx=df_dx,df_dy=df_dy) #calculo de alpha
        xk = xk + alpha*pk#actualizacion de xk
        trayectoria.append(xk.copy())
        norm = np.linalg.norm(phi_val) #norma del gradiente

        print(f"| iteracion {i} "
            f"| xk: {np.round(np.array(xk, dtype=float), decimales)} "
            f"| phi: {phi_val} "
            f"| norm: {round(float(norm), decimales)} "
            f"| pk: {np.round(np.array(pk, dtype=float), decimales)} "
            f"| alpha: {round(float(alpha), decimales)} |")


        if norm < tol:
            return xk, np.array(trayectoria)
        i += 1
    return  xk, np.array(trayectoria)#retorna el punto optimo
#--------------------------------------------------------------------------------------------------------------------------
#Metodo Cuasi-Newton
def quasi_newton(func,xk, tol, nIter,df_dx,df_dy):
    i = 0
    print(df_dx,df_dy) #derivada simbolica y evaluacion de la derivada en xk
    Bk  = np.eye(2) #matriz identidad de 2x2
    trayectoria = [xk.copy()] #lista para guardar la trayectoria
    decimales = int(abs(math.log10(tol)))
    while i < nIter:
        der_eval = np.array([
            f(df_dx,xk), #evaluacion de la derivada parcial respecto a x en xk
            f(df_dy,xk)  #evaluacion de la derivada parcial respecto a y en xk
        ], dtype=np.float64)

        pk = -Bk @ der_eval
        norm = np.linalg.norm(der_eval) #norma del gradiente
        if norm < tol: #si la norma es menor a la tolerancia, se termina el algoritmo
            return xk,np.array(trayectoria)

        alpha = minAlpha(xk,pk,alpha0=1,p=0.7,mu=1e-4,max_iter=1000, func=func, df_dx=df_dx,df_dy=df_dy) #calculo de alpha
        xk_2 = xk+alpha*pk #calculo de la siguiente poscion xk
        sk = xk_2 - xk #diferencia entre la nueva posicion y la posicion anterior
        yk = np.array([ f(df_dx,xk_2),f(df_dy,xk_2)], dtype=np.float64) - der_eval #evaluacion de la derivada en xk_2

        #Actializacion de la matriz B
        yk = yk.reshape(-1,1) #reshape de yk para que sea una matriz columna
        sk = sk.reshape(-1,1)

        divisor = (sk.T @ yk).item() # item Convierte esa matriz 1x1 en un número escalar puro de tipo float.
        if divisor == 0:
            divisor = 0.01
        ykT_Bk_yk = (yk.T @ Bk @ yk).item()

        # Actualizacion de la matriz Bk
        term1 = 1 + (ykT_Bk_yk / divisor)
        term2 = (sk @ sk.T) / divisor
        term3 = (sk @ (yk.T @ Bk)) / divisor 
        term4 = (Bk @ yk @ sk.T) / divisor

        #if (yk @ sk.T)/np.linalg.norm(sk)**2 > 

        Bk = Bk + term1 * term2 - term3 - term4

        print(f"| iteracion {i} "
            f"| xk: {np.round(np.array(xk, dtype=float), decimales)} "
            f"| phi: {der_eval} "
            f"| norm: {round(float(norm), decimales)} "
            f"| pk: {np.round(np.array(pk, dtype=float), decimales)} "
            f"| alpha: {round(float(alpha), decimales)}"
            f"| xk_2: {np.round(np.array(xk_2, dtype=float), decimales)}\n"
            f"| sk: {np.round(np.array(sk, dtype=float), decimales)} \n"
            f"| yk: {np.round(np.array(yk, dtype=float), decimales)} \n"
            f"| bk: {np.round(np.array(Bk, dtype=float), decimales)} |\n"
        )
        xk = xk_2.copy() #actualizacion de xk
        trayectoria.append(xk.copy())
        i += 1
    return xk,np.array(trayectoria) #retorna el punto optimo
#------------------------------------------------------------------------------------------------------------------------
#Busqueda conjugada de Powell
def powell(func, x0, tol=1e-4, max_iter=1000):
    n = len(x0)
    x = np.array(x0, dtype=float).reshape(-1, 1)
    S = np.eye(n)  # Direcciones iniciales (base canónica)
    s1 = np.zeros((n, 1))  # Primera dirección histórica
    s2 = np.zeros((n, 1))  # Segunda dirección histórica
    trayectoria = [x.flatten().copy()]

    for iter_count in range(max_iter):
        x_old = x.copy()

        # --- Fase 1: Búsqueda en direcciones base ---
        for k in range(n):
            P = S[:, k].reshape(-1, 1) #Selecciona la k-esima columna de S como direccion de busqueda
            alpha = min_alpha_lineal(func, x, P)
            x = x + alpha * P
            trayectoria.append(x.flatten().copy())

        # --- Fase 2: Búsqueda en direcciones históricas s1 y s2 ---
        for d in [s1, s2]:
            if np.linalg.norm(d) > tol:  # Solo si la dirección no es nula
                alpha = min_alpha_lineal(func, x, d)
                x = x + alpha * d
                trayectoria.append(x.flatten().copy())

        # --- Fase 3: Actualización de direcciones ---
        d_new = x - x_old #direccionde movimiento
        if np.linalg.norm(d_new) > tol:
            # Actualizar s2 con la dirección anterior de s1
            s2 = s1.copy()
            # Actualizar s1 con la nueva dirección
            s1 = d_new.copy()

            # Reemplazar la dirección menos útil en S
            S[:, :-1] = S[:, 1:]  # Desplazar direcciones
            S[:, -1] = s1.flatten()  # Añadir la nueva dirección

        # Criterio de convergencia
        if np.linalg.norm(x - x_old) < tol:
            break

    print(f"\nOptimización completada en {iter_count+1} iteraciones")
    print(f"Punto óptimo: {x.flatten()}")
    return x.flatten(), np.array(trayectoria)

#------------------------------------------------------------------------------------------------------------------------
def menu(func, df_dx, df_dy):
    # Pedir punto inicial al usuario
    x0_input = input("Ingrese el punto inicial x (deje vacío para usar 0.0): ") or "0.0"
    y0_input = input("Ingrese el punto inicial y (deje vacío para usar 0.0): ") or "0.0"

    try:
        x0 = float(x0_input)
        y0 = float(y0_input)
    except ValueError:
        print("Valores inválidos, usando (0.0, 0.0) como punto inicial")
        x0, y0 = 0.0, 0.0

    xk = np.array([x0, y0], dtype=np.float64)

    # Pedir tolerancia al usuario
    tol_input = input("Ingrese la tolerancia (deje vacío para usar 1e-4): ") or "1e-4"
    try:
        tol = float(tol_input)
    except ValueError:
        print("Tolerancia inválida, usando 1e-4")
        tol = 1e-4
    decimales = int(abs(math.log10(tol))) #decimales para redondear los resultados

    while True:
        print("\nAlgoritmos de optimizacion\n")
        print(f"Función: {func}")
        print(f"Punto inicial: ({x0}, {y0})")
        print(f"Tolerancia: {tol}\n")
        print(f"Derivadas parciales:\n{df_dx}\n{df_dy}\n")

        opcion = input("Seleccione el algoritmo a utilizar:\n1. Gradiente Descendente\n2. Newton\n3. Cuasi-Newton\n4. Powell\n5. Cambiar parámetros\n6. Salir\nOpción: ")

        if opcion == '1':
            print("\nEjecutando el algoritmo de gradiente descendente")
            puntop, trayectoria = grad_desc(tol=tol, nIter=1000, xk=xk, func=func, df_dx=df_dx, df_dy=df_dy)
            print(f"El punto óptimo es: {round(float(puntop[0]), decimales)}, {round(float(puntop[1]), decimales)}")
            graficar_contorno(func, trayectoria,x0=x0,y0=y0,puntop=puntop)
        elif opcion == '2':
            print("\nEjecutando el algoritmo de Newton")
            puntop, trayectoria = newton(func, xk, tol=tol, nIter=1000,df_dx=df_dx, df_dy=df_dy)
            print(f"El punto óptimo es: {round(float(puntop[0]), decimales)}, {round(float(puntop[1]), decimales)}")
            graficar_contorno(func, trayectoria,x0=x0,y0=y0,puntop=puntop)
        elif opcion == '3':
            print("\nEjecutando el algoritmo de Cuasi-Newton")
            puntop, trayectoria = quasi_newton(func, xk, tol=tol, nIter=1000, df_dx=df_dx, df_dy=df_dy)
            print(f"El punto óptimo es: {round(float(puntop[0]), decimales)}, {round(float(puntop[1]), decimales)}")
            graficar_contorno(func, trayectoria,x0=x0,y0=y0,puntop=puntop)
        elif opcion == '4':
            print("\nEjecutando el algoritmo de Powell")
            puntop, trayectoria = powell(func, xk, tol=tol)
            print(f"El punto óptimo es: {round(float(puntop[0]), decimales)}, {round(float(puntop[1]), decimales)}")
            graficar_contorno(func, trayectoria,x0=x0,y0=y0,puntop=puntop)
        elif opcion == '5':
            # Permitir cambiar los parámetros
            x0_input = input(f"Ingrese el nuevo punto inicial x (actual: {x0}): ") or str(x0)
            y0_input = input(f"Ingrese el nuevo punto inicial y (actual: {y0}): ") or str(y0)
            tol_input = input(f"Ingrese la nueva tolerancia (actual: {tol}): ") or str(tol)

            try:
                x0 = float(x0_input)
                y0 = float(y0_input)
                tol = float(tol_input)
                xk = np.array([x0, y0], dtype=np.float64)
            except ValueError:
                print("Valores inválidos, manteniendo los parámetros anteriores")
        elif opcion == '6':
            print("¡Adiós!")
            break
        else:
            print("Opción no válida")

if __name__ == "__main__":
    x, y, a = sp.symbols('x y a')
    #func = x-y+(2*x*y)+(2*x**2)+(y**2) # funciones
    func = 100*(y-(x**3))**2 + ((1-x)**2) # funcion
    df_dx,df_dy = f_grad(func)
    menu(func,df_dx,df_dy) #ejecuta el menú con la función seleccionada