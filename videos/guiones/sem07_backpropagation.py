"""Video de la semana 7: Nivel_Intermedio/clases/sem07_backpropagation.ipynb."""

from motor import (AZUL, GRIS, MORADO, NARANJA, ROJO, TINTA, VERDE, Barras, Caja, Circulo, Codigo, Dispersion, Escena,
                   Flecha, Formula, Marco, Matriz, Paso, Regiones, Seccion, Tabla, Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 7 · Backpropagation · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}

# Datos del notebook, copiados aquí porque el entorno de los videos no tiene sklearn.
# XOR con ruido: la parte de prueba (x1, x2 en centésimas) y su clase; es la misma partición de la semana 6.
# XOR_*: la red 2 → 8 → 2 que entrena el notebook con su propia vuelta (2000 pasos, tasa 0.5).
# *_PERDIDA: la entropía cruzada durante el entrenamiento, muestreada (XOR cada 50 pasos; dígitos cada 5 pasos
# hasta el 100 y cada 25 después).
XOR_X_CENT = [
    -12, 15, 6, 102, -23, 66, 107, -15, 85, 4, 66, 97, 48, -34, -3, 74, 35, 86, -6, 41, -10, -18, -4, -7, 24,
    -25, -6, 6, 93, 136, 89, 99, 6, 79, 115, 40, 82, 94, 111, 2, -2, -16, 126, -19, 113, 7, -48, 109, 9, -2,
    -19, 104, 119, -15, 23, -38, 96, 16, -11, 97, -6, 62, 82, 117, 77, -27, 133, -28, -21, 33, 84, 97, -4, 87,
    86, 9, 90, -11, -15, 123, 96, 113, -2, 2, 115, 86, -30, 89, -2, 29, 115, -10, 130, -10, 79, -19, 94, 3, 13,
    150, -12, -24, 95, -31, 98, 122, 103, 110, 97, 109, -32, 95, 91, 106, 94, -18, 2, 86, 107, 94, 14, 14, 9,
    -5, -2, 0, 105, 100, 36, 102, 112, 109, 84, 115, -5, -7, 19, -9, 135, 25, -21, 113, -35, 112, -18, 103, 22,
    83, 76, 21, -38, -15, -14, -24, 15, 99, 113, 108, 92, 1, 0, 102, 7, 95, 113, -11, 78, -7, 137, 106, 100,
    109, -32, 7, 71, 117, -17, -36, 45, 99, -6, 21, 98, 101, 108, 103, 111, 135, -30, 127, 98, 116, 22, -19, 93,
    76, 5, 23, 119, 113,
]
XOR_Y = "0111100111000000110101110110111011001111000101111101000101100000100001111110010111110000010000100000"
XOR_W1 = [[4.294, -2.473], [-1.054, 1.463], [-0.478, 0.018], [3.665, 1.507], [-1.209, -1.567], [-0.751, -0.904],
          [-2.491, 0.459], [-1.349, -1.645]]
XOR_B1 = [-1.343, 0.592, -0.297, -2.233, 1.366, 0.813, 0.244, 1.472]
XOR_W2 = [[-3.553, -1.32, 0.087, 3.079, 1.214, 1.347, -1.139, 1.558],
          [3.733, 1.209, -0.253, -3.019, -1.507, -0.553, 0.301, -1.487]]
XOR_B2 = [-0.501, 0.501]
XOR_PERDIDA = [0.838, 0.277, 0.138, 0.1, 0.083, 0.073, 0.066, 0.061, 0.058, 0.056, 0.054, 0.052, 0.051, 0.05, 0.049,
               0.048, 0.048, 0.047, 0.047, 0.046, 0.046, 0.046, 0.045, 0.045, 0.045, 0.044, 0.044, 0.044, 0.044,
               0.043, 0.043, 0.043, 0.043, 0.043, 0.042, 0.042, 0.042, 0.042, 0.042, 0.041, 0.041]
DIG_PERDIDA = [2.332, 1.81, 1.336, 0.956, 0.71, 0.557, 0.457, 0.389, 0.34, 0.305, 0.28, 0.261, 0.247, 0.225, 0.208,
               0.193, 0.182, 0.173, 0.165, 0.158, 0.152, 0.128, 0.112, 0.1, 0.091, 0.083, 0.076, 0.071, 0.066, 0.062,
               0.058, 0.054, 0.051, 0.049, 0.046, 0.044, 0.042]

DOS = [AZUL, NARANJA]
xor_x = np.array(XOR_X_CENT, float).reshape(-1, 2) / 100
xor_y = np.array([int(c) for c in XOR_Y])
XOR_SERIES = [(tuple(xor_x[xor_y == c, 0]), tuple(xor_x[xor_y == c, 1])) for c in (0, 1)]


def relu(z):
    return np.maximum(z, 0)


def red_xor(A, B):
    """La red 2 → 8 → 2 del notebook: la clase con el puntaje más alto."""
    X = np.c_[A.ravel(), B.ravel()]
    H = relu(X @ np.array(XOR_W1).T + XOR_B1)
    return (H @ np.array(XOR_W2).T + XOR_B2).argmax(axis=1).reshape(A.shape)


XOR_PASOS = np.arange(len(XOR_PERDIDA)) * 50
XOR_PASOS[-1] = 1999
DIG_PASOS = np.array(list(range(0, 100, 5)) + list(range(100, 500, 25)) + [499])


def curva(pasos, valores):
    return lambda t: float(np.interp(t, pasos, valores))


def enlace(a, b, color=GRIS, grosor=4, etiqueta=None, t=0.32, tam=30, dx=0, dy=None):
    """Flecha entre dos círculos (x, y, r), con una etiqueta opcional; dx y dy la apartan de la línea."""
    (x1, y1, r1), (x2, y2, r2) = a, b
    d = np.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / d, (y2 - y1) / d
    salida = [Flecha(x1 + ux * r1, y1 + uy * r1, x2 - ux * (r2 + 4), y2 - uy * (r2 + 4), color, grosor)]
    if etiqueta:
        dy = -tam * 1.15 if dy is None else dy
        salida.append(Texto(x1 + (x2 - x1) * t + dx, y1 + (y2 - y1) * t + dy, etiqueta, tam, TINTA, alinear="centro"))
    return salida


def grad(x, y, texto, tam=36, id=None):
    """Un gradiente escrito en rojo, centrado en x."""
    return Texto(x, y, f"**{texto}**", tam, ROJO, ROJO, alinear="centro", id=id)


# Posiciones de la compra: (x, y, r) de cada nodo.
PRECIO, ENVIO, CANTIDAD = (300, 330, 62), (300, 560, 62), (300, 790, 62)
SUMA, PROD, TOTAL = (760, 445, 62), (1200, 620, 62), (1620, 620, 70)

# Posiciones de la neurona.
N_W, N_X, N_B = (190, 330, 56), (190, 560, 56), (720, 680, 56)
N_MUL, N_SUM, N_SIG, N_ERR = (460, 445, 56), (720, 445, 56), (1000, 445, 56), (1300, 445, 70)
N_L = (1620, 445, 64)

# Posiciones de la red de la semana 6.
R_X1, R_X2, R_H1, R_H2, R_Z = (130, 370, 58), (130, 690, 58), (510, 370, 62), (510, 690, 62), (880, 530, 62)


def compra():
    """El grafo de la compra, con los valores hacia adelante."""
    return [
        *enlace(PRECIO, SUMA),
        *enlace(ENVIO, SUMA),
        *enlace(SUMA, PROD, etiqueta="subtotal $= 5$", t=0.5, dx=60, dy=-58),
        *enlace(CANTIDAD, PROD),
        *enlace(PROD, TOTAL),
        Circulo(*PRECIO, "$3$", AZUL, 44),
        Circulo(*ENVIO, "$2$", AZUL, 44),
        Circulo(*CANTIDAD, "$4$", AZUL, 44),
        Texto(140, 305, "precio", 32, GRIS, alinear="centro"),
        Texto(140, 535, "envío", 32, GRIS, alinear="centro"),
        Texto(140, 765, "cantidad", 32, GRIS, alinear="centro"),
        Circulo(*SUMA, "$+$", MORADO, 60),
        Circulo(*PROD, r"$\times$", MORADO, 60),
        Circulo(*TOTAL, "$20$", VERDE, 46),
        Texto(TOTAL[0], 486, "total", 32, GRIS, alinear="centro"),
    ]


def neurona():
    """La neurona de la semana 4 como grafo, con los valores hacia adelante."""
    return [
        *enlace(N_W, N_MUL),
        *enlace(N_X, N_MUL),
        *enlace(N_MUL, N_SUM, etiqueta="$1$", t=0.5),
        *enlace(N_B, N_SUM),
        *enlace(N_SUM, N_SIG, etiqueta="$z = 0$", t=0.5),
        *enlace(N_SIG, N_ERR, etiqueta="$a = 0.5$", t=0.5),
        *enlace(N_ERR, N_L),
        Circulo(*N_W, "$0.5$", AZUL, 34),
        Circulo(*N_X, "$2$", AZUL, 40),
        Circulo(*N_B, "$-1$", AZUL, 38),
        Texto(N_W[0] - 95, N_W[1] - 24, "$w$", 36, GRIS, alinear="centro"),
        Texto(N_X[0] - 95, N_X[1] - 24, "$x$", 36, GRIS, alinear="centro"),
        Texto(N_B[0] + 110, 660, "$b$", 36, GRIS, alinear="centro"),
        Circulo(*N_MUL, r"$\times$", MORADO, 54),
        Circulo(*N_SUM, "$+$", MORADO, 54),
        Circulo(*N_SIG, r"$\sigma$", MORADO, 50),
        Circulo(*N_ERR, "$(a - y)^2$", MORADO, 32),
        Circulo(*N_L, "$0.25$", VERDE, 34),
        Texto(N_L[0], 318, "$L$", 36, GRIS, alinear="centro"),
    ]


def red_ida():
    """La red de la semana 6 con la ida ya hecha para x = (2, −1)."""
    return [
        *enlace(R_X1, R_H1, etiqueta="$1$", t=0.5),
        *enlace(R_X1, R_H2, etiqueta="$-1$", t=0.25, dx=46, dy=-30),
        *enlace(R_X2, R_H1, etiqueta="$2$", t=0.25, dx=40, dy=0),
        *enlace(R_X2, R_H2, etiqueta="$1$", t=0.5),
        *enlace(R_H1, R_Z, etiqueta="$2$", t=0.5, dx=14, dy=-48),
        *enlace(R_H2, R_Z, etiqueta="$-1$", t=0.5, dx=-18, dy=-50),
        Circulo(*R_X1, "$2$", AZUL, 44),
        Circulo(*R_X2, "$-1$", AZUL, 44),
        Texto(130, 250, "$x$", 34, AZUL, alinear="centro"),
        Circulo(*R_H1, "$1$", VERDE, 48),
        Circulo(*R_H2, "$0$", GRIS, 48),
        Circulo(*R_Z, "$1.5$", MORADO, 40),
        Texto(510, 250, "$h$", 34, TINTA, alinear="centro"),
        Texto(880, 410, r"$p = \sigma(z)$", 30, TINTA, alinear="centro"),
        Texto(510, 768, "$a_2 = -3$: apagada", 28, GRIS, alinear="centro"),
    ]


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la séptima clase del nivel intermedio: backpropagation, "
            "el algoritmo con el que se entrena toda red neuronal.",
            Texto(960, 250, "SEMANA 7 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**Backpropagation**", 80, alinear="centro"),
            Texto(960, 450, "la regla de la cadena, de atrás hacia adelante", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Es la clase que no se debe delegar. Ten papel y lápiz a mano: vamos despacio, de una compra a una red entera.",
            Caja(110, 640, 400, 200, "**1**\nUna compra", AZUL, 38),
            Caja(560, 640, 400, 200, "**2**\nUna neurona", MORADO, 38),
            Caja(1010, 640, 400, 200, "**3**\nUna red en papel", VERDE, 38),
            Caja(1460, 640, 400, 200, "**4**\nMatrices y entrenar", NARANJA, 38),
        ),
    ),
    # ------------------------------------------------------- derivada numérica
    Seccion("Parte 1 · Por qué no basta la derivada numérica"),
    Escena(
        "Una consola con 2410 perillas",
        Paso(
            "La red de los dígitos de la semana seis tiene dos mil cuatrocientos diez parámetros. Para entrenarla "
            "necesitamos, en cada paso, la derivada de la pérdida respecto de cada uno.",
            Caja(130, 250, 760, 150, "red $64 \\to 32 \\to 10$", AZUL, 44, subtitulo="2410 parámetros"),
        ),
        Paso(
            "Con la derivada numérica, es como una consola de sonido con dos mil cuatrocientas diez perillas. "
            "Giras una un poquito a la derecha, escuchas; un poquito a la izquierda, escuchas; y la devuelves. "
            "Una perilla a la vez.",
            *[Circulo(1040 + 90 * (i % 9), 290 + 90 * (i // 9), 32, "", MORADO if i == 4 else GRIS, relleno=None)
              for i in range(18)],
            Texto(1400, 470, "una perilla a la vez: **2 pasadas por parámetro**", 32, alinear="centro"),
        ),
        Paso(
            "En el notebook, eso son cuatro mil ochocientas veinte pasadas, unos dos segundos, para un solo paso "
            "de entrenamiento. Y entrenar son cientos de pasos.",
            Terminal(130, 600, "parámetros:                     2410\n"
                               "una pasada hacia adelante:      0.42 ms\n"
                               "pasadas para el gradiente:      4820\n"
                               "tiempo del gradiente numérico:  2.23 s, para UN paso de descenso de gradiente",
                     30, ancho=1660),
        ),
        Paso(
            "Backpropagation da los dos mil cuatrocientos diez números con una ida y una vuelta, sin importar "
            "cuántos parámetros haya. Es la regla de la cadena de la semana cuatro, aplicada con orden.",
            Caja(130, 860, 1660, 100, "**Backpropagation:** todos los gradientes con **una ida y una vuelta**", VERDE, 36),
        ),
    ),
    # ------------------------------------------------------------------ grafo
    Seccion("Parte 2 · El grafo computacional"),
    Escena(
        "Una compra como grafo",
        Paso(
            "Empecemos lejos de las redes, con una compra. Pides cuatro camisetas: cada una cuesta tres dólares, "
            "y el envío suma dos dólares por camiseta.",
            *compra(),
        ),
        Paso(
            "El subtotal es tres más dos, cinco. El total es cinco por cuatro, veinte. "
            "Esto es un grafo computacional: cada operación es un nodo, y las flechas llevan los números hacia adelante.",
            Texto(1290, 800, "**grafo computacional**\n(computational graph)", 34, alinear="centro"),
        ),
        Paso(
            "La pregunta de backpropagation es: si cada entrada sube un poquito, ¿cuánto sube el total?",
            Caja(1400, 230, 440, 150, "¿Cuánto sube el **total** si sube cada entrada?", NARANJA, 32),
        ),
    ),
    Escena(
        "De derecha a izquierda",
        Paso(
            "Cada nodo conoce solo su derivada local. La multiplicación: el total cambia lo que valga la otra entrada. "
            "La suma: cada entrada pasa tal cual, con derivada uno.",
            *compra(),
            Caja(1380, 180, 470, 230, "**derivadas locales**\n× : la otra entrada\n+ : 1", MORADO, 32),
        ),
        Paso(
            "Ahora recorremos el grafo al revés. En el total empezamos con un uno: el total cambia exactamente "
            "lo que cambia el total.",
            grad(TOTAL[0], 700, "1"),
        ),
        Paso(
            "Una sola regla en cada nodo: el gradiente de una entrada es el gradiente que llega por la derivada local. "
            "Al subtotal le toca uno por cuatro, cuatro. A la cantidad, uno por cinco, cinco.",
            Formula(960, 900, r"\text{gradiente de una entrada} = \text{gradiente que llega} \times \text{derivada local}",
                    40, alinear="centro"),
            grad(980, 560, "4"),
            grad(CANTIDAD[0], 860, "5"),
        ),
        Paso(
            "Y el subtotal reparte su cuatro: precio y envío reciben cuatro por uno, cuatro cada uno.",
            grad(PRECIO[0], 398, "4"),
            grad(ENVIO[0], 628, "4"),
        ),
        Paso(
            "Se lee así: si el precio sube un centavo, el total sube cuatro centavos, porque compras cuatro camisetas. "
            "Y una camiseta más sube el total en cinco dólares.",
            Marco(200, 255, 200, 190, ROJO),
        ),
    ),
    # --------------------------------------------------------------- patrones
    Seccion("Parte 3 · Tres patrones y una regla para las ramas"),
    Escena(
        "Suma, multiplicación y máximo",
        Paso(
            "En una red casi todos los nodos son de tres tipos. La suma es un repartidor: copia el gradiente "
            "que llega a sus dos entradas. Si llega un cinco, las dos reciben cinco.",
            Texto(330, 210, "**Suma: copia**", 38, AZUL, alinear="centro"),
            *enlace((150, 340, 48), (380, 440, 52)),
            *enlace((150, 540, 48), (380, 440, 52)),
            *enlace((380, 440, 52), (560, 440, 40)),
            Circulo(150, 340, 48, "$2$", AZUL, 36),
            Circulo(150, 540, 48, "$3$", AZUL, 36),
            Circulo(380, 440, 52, "$+$", MORADO, 50),
            grad(540, 360, "5"),
            grad(150, 395, "5"),
            grad(150, 595, "5"),
        ),
        Paso(
            "La multiplicación es un intercambio: cada entrada recibe el gradiente por la otra entrada. "
            "Al dos le toca cinco por tres, quince. Al tres, cinco por dos, diez.",
            Texto(960, 210, "**Multiplicación: intercambia**", 38, AZUL, alinear="centro"),
            *enlace((780, 340, 48), (1010, 440, 52)),
            *enlace((780, 540, 48), (1010, 440, 52)),
            *enlace((1010, 440, 52), (1190, 440, 40)),
            Circulo(780, 340, 48, "$2$", AZUL, 36),
            Circulo(780, 540, 48, "$3$", AZUL, 36),
            Circulo(1010, 440, 52, r"$\times$", MORADO, 50),
            grad(1170, 360, "5"),
            grad(780, 395, "15"),
            grad(780, 595, "10"),
        ),
        Paso(
            "El máximo es una válvula: el gradiente va entero al ganador, y al otro le toca cero. "
            "Solo pasa por donde pasó el valor.",
            Texto(1590, 210, "**Máximo: al ganador**", 38, AZUL, alinear="centro"),
            *enlace((1410, 340, 48), (1640, 440, 52)),
            *enlace((1410, 540, 48), (1640, 440, 52)),
            *enlace((1640, 440, 52), (1820, 440, 40)),
            Circulo(1410, 340, 48, "$2$", AZUL, 36),
            Circulo(1410, 540, 48, "$3$", AZUL, 36),
            Circulo(1640, 440, 52, "max", MORADO, 30),
            grad(1800, 360, "5"),
            grad(1410, 395, "0"),
            grad(1410, 595, "5"),
        ),
        Paso(
            "Y ReLU es un máximo contra cero. Si su entrada era positiva, el gradiente pasa entero. Si no, no pasa nada.",
            Caja(330, 740, 1260, 120, "$\\text{ReLU}(z) = \\max(z, 0)$: si $z > 0$ el gradiente **pasa**; si no, **se corta**",
                 NARANJA, 36),
        ),
    ),
    Escena(
        "Las ramas: los gradientes se suman",
        Paso(
            "Una regla más. Si un valor se usa en dos lugares, recibe gradiente por los dos caminos, y se suman. "
            "Como un empleado que trabaja en dos proyectos: su aporte total es la suma de los dos.",
            Flecha(355, 375, 703, 470, GRIS, 4),
            Flecha(330, 452, 708, 530, GRIS, 4),
            Circulo(300, 400, 60, "$x$", AZUL, 44),
            Circulo(760, 500, 60, r"$\times$", MORADO, 56),
            Texto(530, 330, "primera $x$", 30, GRIS, alinear="centro"),
            Texto(500, 560, "segunda $x$", 30, GRIS, alinear="centro"),
            Formula(1000, 420, r"f = x \cdot x", 52),
        ),
        Paso(
            "Por ejemplo, x por x. La multiplicación le da a la primera x el valor de la segunda, y viceversa. "
            "Sumados, x más x, dos x. Es justo la derivada de x al cuadrado.",
            Formula(1000, 540, r"\frac{\partial f}{\partial x} = \textcolor{rojo}{x} + \textcolor{rojo}{x} = 2x", 52),
            Caja(330, 780, 1260, 110, "Un valor usado dos veces: **los gradientes se suman**", VERDE, 38),
        ),
    ),
    Escena(
        "Tu turno 1: dos grafos en papel",
        Paso(
            "Tu turno. Con las letras a, be y ce valiendo uno, dos y tres, dibuja estos dos grafos, haz la ida "
            "y después la vuelta, con lápiz. En el primero, ojo: la letra be aparece dos veces. Pausa el video.",
            Formula(960, 260, r"f = (a + b) \cdot (b + c) \qquad\qquad g = \max(a, c) \cdot b \qquad\qquad a = 1,\ b = 2,\ c = 3",
                    44, alinear="centro"),
            Codigo(130, 430, "df_da, df_db, df_dc = None, None, None  # TODO\n"
                             "dg_da, dg_db, dg_dc = None, None, None  # TODO", 36),
            pausa=1.2,
        ),
    ),
    # ---------------------------------------------------------------- neurona
    Seccion("Parte 4 · Una neurona, de atrás hacia adelante"),
    Escena(
        "La neurona de la semana 4",
        Paso(
            "Volvamos a la neurona de la semana cuatro. x vale dos, w cero coma cinco y b menos uno. "
            "Hacia adelante: z da cero, la sigmoide da cero coma cinco, y como la respuesta era uno, "
            "el error al cuadrado da cero coma veinticinco.",
            *neurona(),
        ),
        Paso(
            "Ahora la vuelta. Empezamos con un uno en la pérdida. La derivada de a menos y al cuadrado es "
            "dos por a menos y: menos uno.",
            grad(N_L[0], 530, "1"),
            grad(1150, 470, "−1"),
        ),
        Paso(
            "La sigmoide tiene derivada a por uno menos a, que da cero coma veinticinco. Multiplicamos: menos uno por "
            "cero coma veinticinco da menos cero coma veinticinco.",
            grad(860, 470, "−0.25", id="dz"),
        ),
        Paso(
            "Y desde ahí, los patrones. La suma copia: a b le toca menos cero coma veinticinco. "
            "La multiplicación intercambia: a w le toca ese número por x, menos cero coma cinco; "
            "y a x, ese número por w, menos cero coma ciento veinticinco.",
            grad(590, 470, "−0.25"),
            grad(N_B[0], 750, "−0.25"),
            grad(N_W[0], 398, "−0.5"),
            grad(N_X[0], 628, "−0.125"),
        ),
        Paso(
            "El menos cero coma cinco es el número de la semana cuatro. Lo nuevo es esto: el menos cero coma veinticinco "
            "se calculó una sola vez y sirvió para los tres. Por eso backpropagation es barato.",
            Marco(790, 462, 140, 60, ROJO),
            Caja(130, 830, 1660, 100, "Cada gradiente se calcula **una vez** y se **reutiliza** para todo lo que está antes",
                 VERDE, 34),
        ),
    ),
    Escena(
        "Sigmoide con entropía cruzada",
        Paso(
            "Para clasificar no usamos el error al cuadrado sino la entropía cruzada, con p igual a la sigmoide de z.",
            Formula(130, 230, r"L = -\left[\, y \ln p + (1 - y) \ln (1 - p) \,\right] \qquad p = \sigma(z)", 50),
        ),
        Paso(
            "Dos eslabones. El de la pérdida tiene p por uno menos p abajo. El de la sigmoide es justo p por uno menos p. "
            "Al multiplicarlos, se cancelan.",
            Formula(130, 380, r"\frac{\partial L}{\partial z} = \frac{p - y}{\textcolor{rojo}{p\,(1 - p)}} \times "
                              r"\textcolor{rojo}{p\,(1 - p)}", 52),
        ),
        Paso(
            "Queda p menos y: cuánto te pasaste. Si dijiste cero coma ochenta y dos y era uno, el gradiente es "
            "menos cero coma dieciocho. Es el resultado más usado de esta clase.",
            Caja(130, 590, 760, 160, "$\\dfrac{\\partial L}{\\partial z} = p - y$", VERDE, 56),
            Texto(980, 615, "**cuánto te pasaste**\n$0.82 - 1 = -0.18$", 40, alinear="izq"),
        ),
    ),
    # -------------------------------------------------------------- red papel
    Seccion("Parte 5 · Una red en papel"),
    Escena(
        "La vuelta por la red de la semana 6",
        Paso(
            "Ahora la red de la semana seis, con la ida ya hecha para la entrada dos, menos uno. La primera neurona "
            "oculta dio uno; la segunda estaba apagada. La salida es cero coma ochenta y dos, y la respuesta correcta es uno.",
            *red_ida(),
            Tabla(1030, 230, [["Hacia atrás", "Valor"]], [470, 330], tam=32, alto_fila=70),
        ),
        Paso(
            "Primera fila: p menos y, cero coma ochenta y dos menos uno: menos cero coma dieciocho.",
            Formula(1050, 330, r"\partial L / \partial z = 0.818 - 1", 36),
            Formula(1530, 330, r"\textcolor{rojo}{-0.182}", 36),
            grad(880, 610, "−0.182"),
        ),
        Paso(
            "Los pesos de salida: multiplicación, así que reciben ese número por lo que entró, h. "
            "Para la primera neurona, menos cero coma dieciocho por uno. Para la segunda, por cero: cero.",
            Formula(1050, 420, r"\partial L / \partial W_2 = -0.182 \times h", 36),
            Formula(1530, 420, r"\textcolor{rojo}{(-0.182,\ 0)}", 36),
        ),
        Paso(
            "Hacia h, la multiplicación intercambia: menos cero coma dieciocho por los pesos, dos y menos uno. "
            "Da menos cero coma treinta y seis y cero coma dieciocho.",
            Formula(1050, 510, r"\partial L / \partial h = -0.182 \times (2, -1)", 36),
            Formula(1530, 510, r"\textcolor{rojo}{(-0.365,\ 0.182)}", 36),
            grad(510, 445, "−0.365"),
        ),
        Paso(
            "Ahora la válvula ReLU. La primera neurona estaba abierta y deja pasar. La segunda estaba cerrada, "
            "con menos tres: su gradiente se corta y queda cero.",
            Formula(1050, 600, r"\text{ReLU: pasa, se corta}", 36),
            Formula(1530, 600, r"\textcolor{rojo}{(-0.365,\ 0)}", 36),
            grad(510, 812, "0 (cortado)", 30),
            Marco(370, 614, 280, 246, ROJO),
        ),
        Paso(
            "Los pesos de la primera capa: el gradiente de cada neurona por cada entrada. La segunda fila es toda cero: "
            "la neurona apagada no influyó en la salida, así que no recibe culpa, y sus pesos no se mueven con este ejemplo.",
            Formula(1050, 700, r"\partial L / \partial W_1", 36),
            Formula(1430, 680, r"\begin{pmatrix} \textcolor{rojo}{-0.730} & \textcolor{rojo}{0.365} \\ 0 & 0 \end{pmatrix}", 38),
            Caja(1030, 850, 800, 90, "neurona apagada: **cero culpa**", GRIS, 32),
        ),
        Paso(
            "Un detalle de signo: los gradientes de la primera neurona son negativos, y el descenso de gradiente resta. "
            "Así que sus pesos suben: esa neurona empujaba hacia la respuesta correcta. "
            "En el notebook, la derivada numérica da los mismos números.",
            Texto(130, 900, "gradiente negativo → el paso **sube** el peso", 32, alinear="izq"),
        ),
    ),
    Escena(
        "Tu turno 2: la vuelta de tu red de papel",
        Paso(
            "Tu turno. En la semana seis hiciste la ida de esta otra red, con la entrada uno, dos. La red decía clase cero, "
            "pero la respuesta correcta es uno. Te damos p: cero coma trescientos setenta y ocho. Sigue la tabla fila por fila "
            "con lápiz, y responde: ¿qué fila de la primera capa es cero esta vez, y por qué? Pausa el video.",
            Formula(960, 240, r"h_1 = \text{ReLU}(1 \cdot x_1 - 2 \cdot x_2 + 1) \qquad h_2 = \text{ReLU}(-1 \cdot x_1 + 3 \cdot x_2 + 0)",
                    42, alinear="centro"),
            Formula(960, 340, r"z = 1 \cdot h_1 - 0.5 \cdot h_2 + 2 \qquad x = (1, 2) \qquad y = 1 \qquad p = 0.378",
                    42, alinear="centro"),
            Codigo(130, 470, "paper_dz = None   # TODO\npaper_dW2 = None  # TODO\npaper_dh = None   # TODO\n"
                             "paper_dW1 = None  # TODO", 36),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- matrices
    Seccion("Parte 6 · La red en matrices"),
    Escena(
        "Tres fórmulas para cualquier capa",
        Paso(
            "Con treinta y dos neuronas y mil trescientos ejemplos, la tabla no cabe en una hoja. Igual que la ida, "
            "la vuelta se escribe con matrices. Una capa calcula Z igual a X por W transpuesta más b, y desde arriba "
            "le llega G, el gradiente respecto de Z.",
            Formula(130, 220, r"Z = X W^\top + b \qquad G = \frac{\partial L}{\partial Z}", 48),
        ),
        Paso(
            "Tres fórmulas. El gradiente de W es G transpuesta por X. El de b, la suma de las filas de G. "
            "Y el que sigue hacia atrás, para la capa anterior, es G por W.",
            Formula(130, 380, r"\frac{\partial L}{\partial W} = G^\top X \qquad\quad \frac{\partial L}{\partial b} = "
                              r"\sum_{\text{filas}} G \qquad\quad \frac{\partial L}{\partial X} = G\, W", 48),
        ),
        Paso(
            "No hace falta memorizarlas. El gradiente de algo tiene la forma de ese algo, y solo hay una manera "
            "de combinar las piezas para obtener esa forma.",
            Tabla(130, 560, [["Quieres", "Forma", "Única combinación"],
                             ["$\\partial L / \\partial W$", "`(salidas, entradas)`", "`G.T @ X`"],
                             ["$\\partial L / \\partial b$", "`(salidas,)`", "`G.sum(axis=0)`"],
                             ["$\\partial L / \\partial X$", "`(n, entradas)`", "`G @ W`"]],
                  [400, 560, 500], tam=34, alto_fila=78),
        ),
    ),
    Escena(
        "El truco de las formas",
        Paso(
            "Míralo con formas. Con cuatro ejemplos, tres entradas y dos salidas, G tiene cuatro filas y dos columnas, "
            "y X cuatro filas y tres columnas.",
            Matriz(200, 300, 4, 2, 60, NARANJA, etiqueta="`G` (4, 2)"),
            Matriz(480, 300, 4, 3, 60, AZUL, etiqueta="`X` (4, 3)"),
        ),
        Paso(
            "Queremos algo de dos por tres, la forma de W. La única manera: transponer G, que queda de dos por cuatro, "
            "y multiplicarla por X. Cuatro con cuatro se cancela, y sale dos por tres.",
            Matriz(900, 360, 2, 4, 60, NARANJA, etiqueta="`G.T` (2, 4)"),
            Texto(1170, 380, "@", 56, alinear="centro"),
            Matriz(1230, 300, 4, 3, 60, AZUL, etiqueta="`X` (4, 3)"),
            Texto(1460, 380, "→", 56, alinear="centro"),
            Matriz(1530, 360, 2, 3, 60, VERDE, etiqueta="`dW` (2, 3)"),
        ),
        Paso(
            "Cada peso une una entrada con una salida, así que recibe el gradiente de su salida por su entrada, "
            "sumado sobre los ejemplos. Es la misma regla de la multiplicación, para todos los pesos a la vez.",
            Caja(130, 700, 1660, 120, "peso $W_{ij}$: gradiente de la salida $i$ × entrada $j$, **sumado sobre los ejemplos**",
                 VERDE, 36),
        ),
    ),
    Escena(
        "Las piezas sin pesos",
        Paso(
            "Faltan dos piezas sin pesos. ReLU, la válvula, con una matriz: G por Z mayor que cero. "
            "Pasa solo donde Z era positivo.",
            Caja(130, 230, 760, 130, "ReLU: `G * (Z > 0)`", NARANJA, 44),
        ),
        Paso(
            "Y la softmax con entropía cruzada, lo que prometimos en la semana cuatro. La pérdida de un ejemplo "
            "de la clase c se puede escribir así: menos z c, más el logaritmo de la suma de exponenciales.",
            Formula(130, 430, r"L = -\ln p_c = -z_c + \ln \sum_j e^{z_j}", 50),
        ),
        Paso(
            "Al derivar respecto de z k, el primer término da menos uno si k es la clase correcta, y el segundo da "
            "exactamente p k. Otra vez, p menos y.",
            Formula(130, 590, r"\frac{\partial L}{\partial z_k} = -[k = c] + \frac{e^{z_k}}{\sum_j e^{z_j}} = "
                              r"\textcolor{verde}{p_k - [k = c]}", 50),
        ),
        Paso(
            "Con una fila por ejemplo, queda P menos Y, donde Y tiene un uno en la clase correcta y ceros en las demás. "
            "Y como la pérdida es el promedio, se divide entre n.",
            Caja(130, 790, 760, 140, "$\\dfrac{\\partial L}{\\partial Z} = \\dfrac{P - Y}{n}$", VERDE, 50),
        ),
    ),
    Escena(
        "La ida y la vuelta en código",
        Paso(
            "Con eso, la red de los dígitos entera. La ida guarda lo que la vuelta va a necesitar. "
            "La vuelta empieza con P menos Y sobre n, y aplica las tres fórmulas dos veces, con la válvula ReLU en el medio.",
            Codigo(130, 200, "def backward(scores, cache, y, W2):\n"
                             "    X, Z1, H = cache\n"
                             "    n = len(y)\n"
                             "    G2 = softmax(scores)\n"
                             "    G2[np.arange(n), y] -= 1\n"
                             "    G2 /= n                       # (P − Y) / n\n"
                             "    dW2 = G2.T @ H\n"
                             "    db2 = G2.sum(axis=0)\n"
                             "    dH = G2 @ W2\n"
                             "    G1 = dH * (Z1 > 0)            # ReLU: pasa solo donde estaba abierta\n"
                             "    dW1 = G1.T @ X\n"
                             "    db1 = G1.sum(axis=0)\n"
                             "    return dW1, db1, dW2, db2", 30),
        ),
    ),
    Escena(
        "¿Más rápido? ¿Y da lo mismo?",
        Paso(
            "¿Cuánto más rápido? Dos segundos contra un milisegundo: unas dos mil veces. "
            "Y la ventaja crece con la red.",
            Terminal(130, 220, "gradiente numérico (parte 1):   2.23 s\n"
                               "backpropagation (ida y vuelta): 1.06 ms\n"
                               "backpropagation es unas 2,106 veces más rápida", 32, ancho=1660),
        ),
        Paso(
            "¿Y da lo mismo? Se compara con el error relativo. Menos de diez a la menos siete es que coinciden. "
            "Comprobar así un gradiente es lo que hacen los profesionales siempre que escriben una vuelta a mano.",
            Formula(130, 470, r"\text{error relativo} = \frac{\lVert a - b \rVert}{\lVert a \rVert + \lVert b \rVert}", 46),
            Terminal(980, 450, "W1: forma (32, 64)  error relativo 6.2e-10\n"
                               "b1: forma (32,)     error relativo 4.1e-10\n"
                               "W2: forma (10, 32)  error relativo 4.0e-10\n"
                               "b2: forma (10,)     error relativo 2.5e-10", 28, ancho=810),
            Caja(130, 740, 1660, 110, "menos de $10^{-7}$: **coinciden** · más de $10^{-4}$: **hay un error en tu vuelta**",
                 VERDE, 36),
        ),
    ),
    Escena(
        "Tu turno 3: la vuelta para cualquier profundidad",
        Paso(
            "Tu turno. En la semana seis escribiste la ida para cualquier cantidad de capas. Ahora la vuelta: "
            "ida guardando la entrada y la Z de cada capa, P menos Y sobre n, y de la última capa a la primera, "
            "las tres fórmulas con la válvula ReLU. Si te trabas, vuelve a la tabla de las formas. Pausa el video.",
            *vinetas(130, 230, [
                "**Ida:** guarda la entrada y la `Z` de cada capa.",
                "**Salida:** `G = (P - Y) / n`.",
                "**Vuelta:** de la última capa a la primera, `dW`, `db` y el `G` de la anterior.",
            ], tam=38, sep=90, ancho=1650, numeradas=True),
            Codigo(130, 580, "def mlp_backward(X, y, weights, biases):\n    # TODO\n    raise NotImplementedError", 36),
            pausa=1.2,
        ),
    ),
    # ---------------------------------------------------------------- entrenar
    Seccion("Parte 7 · Entrenar con nuestro propio gradiente"),
    Escena(
        "Ida, vuelta, un paso",
        Paso(
            "Ya tenemos todo. Un paso de entrenamiento son tres cosas, en este orden: la ida, la vuelta, "
            "y un paso cuesta abajo, como en la semana cuatro.",
            Codigo(130, 220, "for _ in range(steps):\n"
                             "    scores, cache = forward(X, *params)               # 1. ida\n"
                             "    grads = backward(scores, cache, y, params[2])      # 2. vuelta\n"
                             "    for p, g in zip(params, grads):\n"
                             "        p -= lr * g                                    # 3. un paso cuesta abajo", 32),
        ),
        Paso(
            "Los pesos empiezan al azar, con una escala que veremos en la semana nueve. Si empezaran todos iguales, "
            "todas las neuronas recibirían el mismo gradiente y harían siempre lo mismo.",
            Caja(130, 560, 1660, 120, "pesos iniciales **al azar**: si empiezan iguales, todas las neuronas aprenden lo mismo",
                 NARANJA, 36),
        ),
    ),
    Escena(
        "Primero XOR",
        Paso(
            "Primero XOR con ruido, el de la semana seis, con ocho neuronas ocultas. La pérdida arranca cerca de "
            "logaritmo de dos, lo de adivinar, y cae en pocos cientos de pasos.",
            Dispersion(200, 260, 620, 460, [((), ())], rx=(0, 2000), ry=(0, 0.9), marcas_x=(0, 1000, 2000),
                       marcas_y=(0, 0.69), etiqueta_x="paso", etiqueta_y="entropía cruzada",
                       curvas=[(lambda t: 0.693, GRIS, 3), (curva(XOR_PASOS, XOR_PERDIDA), AZUL, 6)]),
            Texto(560, 250, "adivinar: $\\ln 2 = 0.69$", 28, GRIS, alinear="centro"),
        ),
        Paso(
            "Y las regiones que aprendió: las cuatro esquinas de XOR. Nadie le dijo a la red qué tenía que hacer "
            "cada neurona oculta: lo encontró el gradiente. Acierta el noventa y siete por ciento de la prueba, "
            "contra el cincuenta y uno de la línea base.",
            Regiones(1080, 260, 560, 460, red_xor, (-0.8, 1.8), (-0.8, 1.8), DOS, XOR_SERIES,
                     marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x="$x_1$", etiqueta_y="$x_2$", radio=7),
            Texto(1720, 420, "**clase 0**", 32, AZUL, AZUL),
            Texto(1720, 470, "**clase 1**", 32, NARANJA, NARANJA),
            Terminal(200, 860, "línea base: 0.510   ·   nuestra red: 0.970   (accuracy de prueba)", 28, ancho=1500),
        ),
    ),
    Escena(
        "Después, los dígitos",
        Paso(
            "Después, los dígitos, con la red de la parte uno. La pérdida baja desde logaritmo de diez, "
            "lo de adivinar entre diez clases.",
            Dispersion(200, 260, 560, 440, [((), ())], rx=(0, 500), ry=(0, 2.5), marcas_x=(0, 250, 500),
                       marcas_y=(0, 1, 2), etiqueta_x="paso", etiqueta_y="entropía cruzada",
                       curvas=[(lambda t: 2.303, GRIS, 3), (curva(DIG_PASOS, DIG_PERDIDA), AZUL, 6)]),
            Texto(500, 250, "adivinar: $\\ln 10 = 2.30$", 28, GRIS, alinear="centro"),
        ),
        Paso(
            "Contra los demás: la línea base, cero coma diez. Nuestra red, cero coma noventa y siete, empatada con la "
            "regresión logística y casi igual a la red ya hecha de la librería. Todo con unas treinta líneas de NumPy, escritas hoy.",
            Barras(1000, 330, 820, 380, ["línea base", "logística", "nuestra red", "sklearn MLP"],
                   [0.102, 0.969, 0.969, 0.973], [GRIS, AZUL, VERDE, MORADO], vmax=1.0, formato="{:.3f}", tam=28),
        ),
        Paso(
            "Y una cuenta más: quinientos pasos tardaron menos de un segundo. Con la derivada numérica habrían sido "
            "unos diecinueve minutos.",
            Caja(200, 850, 1620, 90, "500 pasos: **0.6 s** con backpropagation · unos **19 minutos** con la derivada numérica",
                 VERDE, 34),
        ),
    ),
    Escena(
        "Tu turno 4: cazar un error",
        Paso(
            "El último tu turno. Un error en la vuelta no hace fallar el código: la red entrena igual, solo que peor. "
            "En el notebook hay una vuelta con un error: compara cada gradiente con el numérico, di cuáles están mal "
            "y corrige la línea. Pausa el video.",
            Caja(130, 240, 1660, 130, "`buggy_backward` tiene **un** error", ROJO, 44),
            Codigo(130, 450, "wrong_params = None  # TODO\n\n"
                             "def fixed_backward(scores, cache, y, W2):\n    # TODO\n    raise NotImplementedError", 36),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------------ cierre
    Seccion("Cierre"),
    Escena(
        "Lo que vimos hoy",
        *[
            Paso(decir, *grupo)
            for decir, grupo in zip(
                [
                    "Repasemos. La derivada numérica cuesta dos pasadas por parámetro. Backpropagation da todo con una ida y una vuelta.",
                    "Una sola regla: gradiente que llega por derivada local. La suma copia, la multiplicación intercambia, "
                    "el máximo manda todo al ganador, y las ramas se suman.",
                    "Con entropía cruzada, en la salida, p menos y. Para una capa, tres fórmulas que salen de las formas. "
                    "Y una vuelta escrita a mano se comprueba siempre con la derivada numérica.",
                ],
                (lambda v: [v[0:2], v[2:4], v[4:6]])(vinetas(130, 240, [
                    "**Derivada numérica:** 2 pasadas por parámetro. **Backprop:** una ida y una vuelta.",
                    "La ida **guarda** sus valores: la vuelta los necesita.",
                    "**Gradiente que llega × derivada local**, de atrás hacia adelante.",
                    "Suma **copia**, multiplicación **intercambia**, máximo **al ganador**, ramas **se suman**.",
                    "Salida con entropía cruzada: $p - y$. Capa: $G^\\top X$, $\\sum G$, $G W$.",
                    "Una vuelta a mano se **comprueba** con la derivada numérica.",
                ], tam=38, sep=112, ancho=1650)),
            )
        ],
    ),
    Escena(
        "Un poco de historia",
        Paso(
            "La idea de recorrer la regla de la cadena hacia atrás se publicó en mil novecientos setenta. "
            "Se popularizó para redes en mil novecientos ochenta y seis, con un artículo de Rumelhart, Hinton y Williams. "
            "Era la pieza que faltaba desde el libro de mil novecientos sesenta y nueve: cómo entrenar las capas ocultas.",
            Caja(130, 300, 480, 200, "**1969**", GRIS, 50, subtitulo="Perceptrons: el muro de XOR"),
            Flecha(630, 400, 720, 400, GRIS, 5),
            Caja(740, 300, 480, 200, "**1970**", AZUL, 50, subtitulo="Linnainmaa: la cadena hacia atrás"),
            Flecha(1240, 400, 1330, 400, GRIS, 5),
            Caja(1350, 300, 480, 200, "**1986**", VERDE, 50, subtitulo="Rumelhart, Hinton y Williams"),
        ),
    ),
    Escena(
        "Para la semana 8",
        Paso(
            "Para la semana ocho: entrega el P uno, un motor de diferenciación automática desde cero. "
            "Hoy escribimos la vuelta a mano para una red fija. En P uno la vas a automatizar: cada operación recuerda "
            "sus entradas y su derivada local, y una sola llamada recorre el grafo hacia atrás.",
            Vineta(130, 250, "**P1:** motor de diferenciación automática desde cero, al estilo de micrograd. "
                   "Entrena un MLP sobre \"two moons\". En `P1/`, al inicio de la semana 8.", 38, NARANJA, numero=1, ancho=1650),
        ),
        Paso(
            "Y la cuota semanal. Para verlo con otras imágenes, los videos de 3Blue1Brown y de Karpathy están enlazados en el notebook.",
            Vineta(130, 450, "**Cuota semanal** en ioai.artix.tech.", 38, NARANJA, numero=2, ancho=1650),
            Vineta(130, 560, "**Para ver:** 3Blue1Brown y Karpathy, enlaces en el notebook.", 38, NARANJA, numero=3, ancho=1650),
        ),
        Paso(
            "La semana que viene llega PyTorch. Con una sola línea, el método backward, hace exactamente lo de hoy, "
            "para cualquier grafo. Ahora ya sabes qué hay dentro de esa línea. ¡Nos vemos en la semana ocho!",
            Caja(130, 680, 1660, 130, "**Semana 8: PyTorch**", AZUL, 44, subtitulo="`loss.backward()` hace lo de hoy, para cualquier grafo"),
            Texto(960, 860, "**¡Nos vemos en la semana 8!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
