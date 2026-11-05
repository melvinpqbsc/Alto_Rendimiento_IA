"""Video de la semana 6: Nivel_Intermedio/clases/sem06_perceptron_mlp.ipynb."""

from motor import (AZUL, GRIS, MORADO, NARANJA, ROJO, TINTA, VERDE, Barras, Caja, Circulo, Codigo, Dispersion, Escena,
                   Flecha, Formula, Grafica, Linea, Marco, Matriz, Paso, Regiones, Seccion, Tabla, Terminal, Texto, Vineta,
                   vinetas)
import numpy as np

META = {
    "pie": "Semana 6 · Del perceptrón a la red multicapa · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}

# Datos del notebook, copiados aquí porque el entorno de los videos no tiene pandas ni sklearn.
# Pingüinos Adelie y Gentoo: profundidad del pico en décimas de mm, aleta en mm, 1 si es Gentoo.
# XOR con ruido: la parte de prueba (x1, x2 en centésimas) y su clase.
# Redes MLPClassifier del notebook, (W1, b1, W2, b2): ocho neuronas ocultas, solo cambia la activación.
PICO_PROF_DECIMAS = [
    187, 174, 180, 193, 206, 178, 196, 181, 202, 171, 173, 176, 212, 211, 178, 190, 207, 184, 215, 183, 187,
    192, 181, 172, 189, 186, 179, 186, 189, 167, 181, 178, 189, 170, 211, 200, 185, 193, 191, 180, 184, 185,
    197, 169, 188, 190, 189, 179, 212, 177, 189, 179, 195, 181, 186, 175, 188, 166, 191, 169, 211, 170, 182,
    171, 180, 162, 191, 166, 194, 190, 184, 172, 189, 175, 185, 168, 194, 161, 191, 172, 176, 188, 194, 178,
    203, 195, 186, 192, 188, 180, 181, 171, 181, 173, 189, 186, 185, 161, 185, 179, 200, 160, 200, 186, 189,
    172, 200, 170, 190, 165, 203, 177, 195, 207, 183, 170, 205, 170, 186, 172, 198, 170, 185, 159, 190, 176,
    183, 171, 180, 179, 192, 185, 185, 176, 175, 175, 201, 165, 179, 171, 172, 155, 170, 168, 187, 186, 184,
    178, 181, 171, 185, 132, 163, 141, 152, 145, 135, 146, 153, 134, 154, 137, 161, 137, 146, 146, 157, 135,
    152, 145, 151, 143, 145, 145, 158, 131, 151, 143, 150, 143, 153, 153, 142, 145, 170, 148, 163, 137, 173,
    136, 157, 137, 160, 137, 150, 159, 139, 139, 159, 133, 158, 142, 141, 144, 150, 144, 154, 139, 150, 145,
    153, 138, 149, 139, 157, 142, 168, 144, 162, 142, 150, 150, 156, 156, 148, 150, 160, 142, 163, 138, 164,
    145, 156, 146, 159, 138, 173, 144, 142, 140, 170, 150, 171, 145, 161, 147, 157, 158, 146, 144, 165, 150,
    170, 155, 150, 138, 161, 147, 158, 140, 151, 152, 159, 152, 163, 141, 160, 157, 162, 137, 143, 157, 148, 161,
]
ALETA_MM = [
    181, 186, 195, 193, 190, 181, 195, 193, 190, 186, 180, 182, 191, 198, 185, 195, 197, 184, 194, 174, 180,
    189, 185, 180, 187, 183, 187, 172, 180, 178, 178, 188, 184, 195, 196, 190, 180, 181, 184, 182, 195, 186,
    196, 185, 190, 182, 179, 190, 191, 186, 188, 190, 200, 187, 191, 186, 193, 181, 194, 185, 195, 185, 192,
    184, 192, 195, 188, 190, 198, 190, 190, 196, 197, 190, 195, 191, 184, 187, 195, 189, 196, 187, 193, 191,
    194, 190, 189, 189, 190, 202, 205, 185, 186, 187, 208, 190, 196, 178, 192, 192, 203, 183, 190, 193, 184,
    199, 190, 181, 197, 198, 191, 193, 197, 191, 196, 188, 199, 189, 189, 187, 198, 176, 202, 186, 199, 191,
    195, 191, 210, 190, 197, 193, 199, 187, 190, 191, 200, 185, 193, 193, 187, 188, 190, 192, 185, 190, 184,
    195, 193, 187, 201, 211, 230, 210, 218, 215, 210, 211, 219, 209, 215, 214, 216, 214, 213, 210, 217, 210,
    221, 209, 222, 218, 215, 213, 215, 215, 215, 216, 215, 210, 220, 222, 209, 207, 230, 220, 220, 213, 219,
    208, 208, 208, 225, 210, 216, 222, 217, 210, 225, 213, 215, 210, 220, 210, 225, 217, 220, 208, 220, 208,
    224, 208, 221, 214, 231, 219, 230, 214, 229, 220, 223, 216, 221, 221, 217, 216, 230, 209, 220, 215, 223,
    212, 221, 212, 224, 212, 228, 218, 218, 212, 230, 218, 228, 212, 224, 214, 226, 216, 222, 203, 225, 219,
    228, 215, 228, 216, 215, 210, 219, 208, 209, 216, 229, 213, 230, 217, 230, 217, 222, 214, 215, 222, 212, 213,
]
GENTOO = "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111"
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
MLP_IDENTITY = (
    [[0.2726, 0.2804, 0.3736, -0.1801, -0.1256, 0.279, 0.0532, 0.4975], [0.643, -0.173, 0.3469, 0.1509, 0.1017, 0.6291, -0.7373, -0.6153]],
    [-0.6929, 0.4868, 0.4614, 0.5202, 0.7337, 0.466, -0.0373, 0.3846],
    [[-0.5153], [0.0547], [-0.627], [0.7098], [0.0477], [-0.1511], [-0.4775], [0.1158]],
    [-0.135],
)
MLP_TANH = (
    [[3.9914, 4.0011, -29.6701, 6.1528, -1.55, -9.1385, -30.7156, -16.8396], [5.3657, -7.3358, 17.1202, -31.9073, 0.2203, 13.1918, -4.4698, 19.5052]],
    [-6.8669, 4.9154, 16.6242, 0.8026, -2.139, -4.1965, -2.2892, 1.0809],
    [[-41.5026], [-35.1844], [-28.7671], [-19.8265], [0.9747], [13.1879], [-21.437], [-17.9033]],
    [-6.9821],
)
MLP_RELU = (
    [[24.587, -12.1919, -0.697, -5.5394, -4.1746, -4.4652, -7.8922, 28.0863], [8.7058, -1.1625, -12.6849, 17.0204, 2.2078, 0.9044, -5.8423, -1.8838]],
    [-14.4152, -0.7535, 5.9435, -8.2402, 3.9123, 5.1299, 11.5523, -12.2498],
    [[-17.4612], [-10.4454], [5.5667], [9.4066], [6.378], [7.5336], [-9.1587], [17.1022]],
    [3.7347],
)

pico = np.array(PICO_PROF_DECIMAS) / 10
aleta = np.array(ALETA_MM, float)
es_gentoo = np.array(list(GENTOO)) == "1"
# StandardScaler y Perceptron del notebook (ajustados con el entrenamiento).
MEDIA, DESV = np.array([16.8941, 202.3317]), np.array([2.0831, 14.7885])
P_W, P_B = np.array([-1.58697, 1.71293]), -1.0
pico_z, aleta_z = (pico - MEDIA[0]) / DESV[0], (aleta - MEDIA[1]) / DESV[1]

xor_x = np.array(XOR_X_CENT).reshape(-1, 2) / 100
xor_y = np.array(list(XOR_Y)) == "1"
XOR_SERIES = [(xor_x[~xor_y, 0], xor_x[~xor_y, 1]), (xor_x[xor_y, 0], xor_x[xor_y, 1])]

GATE_X = [0, 0, 1, 1]
GATE_Y = [0, 1, 0, 1]
DOS = (AZUL, NARANJA)
DA = ["da 0", "da 1"]


def recta(m, b, color, hasta=None):
    """La recta y = m·x + b como curva, que a diferencia de una recta se recorta al rango del panel."""
    return (lambda x: m * x + b if hasta is None or x < hasta else 1e9), color, 5


def puntos(salidas):
    """Las cuatro esquinas repartidas en dos series según la salida de la compuerta."""
    cero = [(a, b) for a, b, s in zip(GATE_X, GATE_Y, salidas) if s == 0]
    uno = [(a, b) for a, b, s in zip(GATE_X, GATE_Y, salidas) if s == 1]
    return [(tuple(p[0] for p in cero), tuple(p[1] for p in cero)), (tuple(p[0] for p in uno), tuple(p[1] for p in uno))]


def compuerta(x, y, w, h, salidas, rectas=(), id=None, etiqueta_x="$x_1$", etiqueta_y="$x_2$"):
    return Dispersion(x, y, w, h, puntos(salidas), colores=DOS, nombres=DA, rx=(-0.5, 1.5), ry=(-0.5, 1.9),
                      marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x=etiqueta_x, etiqueta_y=etiqueta_y, radio=15,
                      curvas=[recta(*r) for r in rectas], id=id)


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


def red_mlp(params, activacion):
    """predecir(A, B) de una red entrenada del notebook: (W1, b1, W2, b2) de MLPClassifier."""
    W1, b1, W2, b2 = (np.array(p) for p in params)

    def predecir(A, B):
        X = np.c_[A.ravel(), B.ravel()]
        return ((activacion(X @ W1 + b1) @ W2 + b2).ravel() > 0).astype(int).reshape(A.shape)

    return predecir


def relu(z):
    return np.maximum(z, 0)


def sigmoide(z):
    return 1 / (1 + np.exp(-z))


def plegable(n):
    """La aproximación de sin(x) del notebook: n neuronas ReLU con pesos calculados a mano."""
    nudos = np.linspace(-np.pi, np.pi, n + 1)
    pendientes = np.diff(np.sin(nudos)) / np.diff(nudos)
    c = np.diff(pendientes, prepend=0.0)
    return lambda x: float(np.sin(nudos[0]) + relu(x - nudos[:-1]) @ c)


def panel_plegable(x, n, error):
    return [
        Dispersion(x, 270, 300, 250, [((), ())], rx=(-np.pi, np.pi), ry=(-1.2, 1.2), marcas_x=(-3, 0, 3), marcas_y=(-1, 0, 1),
                   curvas=[(np.sin, GRIS, 10), (plegable(n), AZUL, 5)]),
        Texto(x + 119, 625, f"**{n} neuronas**\nerror medio {error}", 30, alinear="centro"),
    ]


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la sexta clase del nivel intermedio. Hoy construimos una red neuronal, "
            "desde una sola neurona hasta varias capas.",
            Texto(960, 250, "SEMANA 6 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**Del perceptrón a la red multicapa**", 80, alinear="centro"),
            Texto(960, 450, "una neurona, un muro y cómo saltarlo", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Cuatro ideas: qué calcula una neurona, un problema que no puede resolver, cómo dos capas "
            "lo resuelven, y por qué hacen falta las activaciones.",
            Caja(110, 640, 400, 200, "**1**\nUna neurona", AZUL, 38),
            Caja(560, 640, 400, 200, "**2**\nEl muro: XOR", ROJO, 38),
            Caja(1010, 640, 400, 200, "**3**\nDos capas", VERDE, 38),
            Caja(1460, 640, 400, 200, "**4**\nActivaciones", MORADO, 38),
        ),
    ),
    # --------------------------------------------------------------- perceptrón
    Seccion("Parte 1 · Una neurona: el perceptrón"),
    Escena(
        "¿Vas a la playa?",
        Paso(
            "Imagina que decides si vas a la playa. Miras tres cosas, y no todas pesan igual: que haya sol suma tres, "
            "que vayan tus amigos suma dos, y que mañana haya examen resta cuatro.",
            Tabla(130, 250, [["Evidencia", "¿Hoy? ($x$)", "Peso ($w$)"], ["Hace sol", "1", "**+3**"],
                             ["Van tus amigos", "1", "**+2**"], ["Mañana hay examen", "1", "**−4**"]],
                  [460, 260, 260], tam=36, alto_fila=84),
        ),
        Paso(
            "Sumas todo, más tu ánimo de partida: menos cero coma cinco, porque eres un poco perezoso. "
            "Da cero coma cinco. Como es mayor que cero, vas.",
            Formula(130, 640, r"3 + 2 - 4 - 0.5 = \textcolor{verde}{0.5} > 0", 56),
            Caja(1230, 610, 500, 130, "**Vas a la playa**", VERDE, 44),
        ),
        Paso(
            "Si tus amigos no van, esa evidencia vale cero, y da menos uno coma cinco. Te quedas en casa.",
            Formula(130, 800, r"3 + 0 - 4 - 0.5 = \textcolor{rojo}{-1.5} \le 0", 56),
            Caja(1230, 770, 500, 130, "**Te quedas**", ROJO, 44),
        ),
    ),
    Escena(
        "Eso es un perceptrón",
        Paso(
            "Esa cuenta es un perceptrón, propuesto por Frank Rosenblatt en mil novecientos cincuenta y ocho. "
            "Las entradas se multiplican por sus pesos, se suman, y un escalón decide: sí o no.",
            *enlace((250, 300, 75), (880, 480, 95), etiqueta="$w_1 = +3$", t=0.45, dx=30, dy=-52),
            *enlace((250, 480, 75), (880, 480, 95), etiqueta="$w_2 = +2$", t=0.45, dy=-50),
            *enlace((250, 660, 75), (880, 480, 95), etiqueta="$w_3 = -4$", t=0.45, dx=30, dy=8),
            Circulo(250, 300, 75, "sol", AZUL, 32),
            Circulo(250, 480, 75, "amigos", AZUL, 28),
            Circulo(250, 660, 75, "examen", AZUL, 28),
            Circulo(880, 480, 95, "$+$", MORADO, 64),
            Texto(880, 600, "$b = -0.5$", 34, MORADO, alinear="centro"),
            Flecha(975, 480, 1130, 480, GRIS, 4),
            Caja(1130, 410, 260, 140, "**escalón**", NARANJA, 38),
            Flecha(1390, 480, 1530, 480, GRIS, 4),
            Circulo(1610, 480, 80, "sí / no", VERDE, 30),
        ),
        Paso(
            "En fórmula: si la suma con pesos más be es mayor que cero, la salida es uno. Los números doble ve son los pesos, "
            "y be es el sesgo, que solo comparte el nombre con el de la semana cinco.",
            Formula(960, 720, r"\hat{y} = 1 \ \text{ si } \ \textcolor{azul}{w} \cdot x + \textcolor{morado}{b} > 0, "
                    r"\quad \text{si no, } \hat{y} = 0", 50, alinear="centro"),
            Texto(960, 840, "$w$: **pesos** (weights)   ·   $b$: **sesgo** (bias), no es el de sesgo-varianza", 36, GRIS,
                  alinear="centro"),
        ),
    ),
    Escena(
        "Una neurona es una recta",
        Paso(
            "Con buenos pesos, un perceptrón calcula funciones lógicas. La compuerta AND solo da uno si las dos "
            "entradas son uno.",
            Texto(469, 190, "**AND**", 36, alinear="centro"),
            compuerta(190, 280, 620, 380, (0, 0, 0, 1), rectas=[(-1, 1.5, GRIS)]),
            Texto(469, 765, "$w = (1, 1)$,  $b = -1.5$", 34, alinear="centro"),
        ),
        Paso(
            "La compuerta OR da uno si alguna lo es: los mismos pesos, con otro sesgo.",
            Texto(1409, 190, "**OR**", 36, alinear="centro"),
            compuerta(1130, 280, 620, 380, (0, 1, 1, 1), rectas=[(-1, 0.5, GRIS)]),
            Texto(1409, 765, "$w = (1, 1)$,  $b = -0.5$", 34, alinear="centro"),
        ),
        Paso(
            "La línea es donde la suma vale exactamente cero: de un lado dice uno, del otro dice cero. "
            "Con dos entradas, una neurona es una recta.",
            Caja(330, 835, 1260, 100, "Con dos entradas, un perceptrón es **una recta** que separa el plano", AZUL, 38),
        ),
    ),
    Escena(
        "Tu turno 1: elige los pesos",
        Paso(
            "Tu turno. Busca pesos para dos compuertas nuevas. La compuerta NOR da uno solo si las dos entradas "
            "son cero. Y la mayoría de tres da uno si al menos dos de las tres son uno. Prueba con lápiz. Pausa el video.",
            Codigo(130, 280, "nor_w, nor_b = None, None  # TODO\nmaj_w, maj_b = None, None  # TODO", 36),
            Caja(130, 500, 1660, 140, "Hay muchas respuestas correctas: la prueba revisa todos los casos", VERDE, 38),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- la regla
    Seccion("Parte 2 · Aprender de los errores"),
    Escena(
        "Corregir la receta",
        Paso(
            "Elegir pesos a mano no escala. Por eso, la neurona aprende de sus errores, como cuando corriges una receta: "
            "si salió sin sal, le pones más. Si salió bien, no tocas nada.",
            Caja(130, 260, 800, 160, "**Acertó**", VERDE, 42, subtitulo="no se toca nada"),
            Caja(990, 260, 800, 160, "**Se equivocó**", ROJO, 42, subtitulo="se mueven los pesos de las entradas encendidas"),
        ),
        Paso(
            "La regla mira un ejemplo a la vez. Al peso se le suma la tasa de aprendizaje, por el error, por la entrada.",
            Formula(960, 490, r"w \leftarrow w + \eta\,(y - \hat{y})\,x \qquad\qquad b \leftarrow b + \eta\,(y - \hat{y})", 54,
                    alinear="centro"),
        ),
        Paso(
            "Si dijo cero y era uno, los pesos suben. Si dijo uno y era cero, bajan. "
            "Una pasada completa por los datos se llama época.",
            Caja(130, 660, 800, 170, "Dijo 0 y era 1: $y - \\hat{y} = +1$", AZUL, 38, subtitulo="los pesos **suben**"),
            Caja(990, 660, 800, 170, "Dijo 1 y era 0: $y - \\hat{y} = -1$", NARANJA, 38, subtitulo="los pesos **bajan**"),
            Texto(960, 860, "Una pasada completa por los datos: una **época** (epoch)", 34, GRIS, alinear="centro"),
        ),
    ),
    Escena(
        "La regla, paso a paso",
        Paso(
            "Sigamos la regla en AND, con todo en cero. Los tres primeros puntos aciertan. "
            "El punto uno, uno debía dar uno y dio cero: los pesos suben. La recta todavía queda fuera del gráfico.",
            Tabla(110, 240, [["$x$", "$y$", "dijo", "$w$ después", "$b$ después"], ["(1, 1)", "1", "0", "(1, 1)", "1"]],
                  [150, 80, 110, 230, 240], tam=30, alto_fila=62),
            compuerta(1090, 270, 650, 520, (0, 0, 0, 1), rectas=[(-1, -1, GRIS)], id="r1"),
        ),
        Paso(
            "Ahora se pasó: el punto cero, cero da uno, y debía dar cero. Baja el sesgo. "
            "Cada error empuja la recta un poco.",
            Tabla(110, 240, [["$x$", "$y$", "dijo", "$w$ después", "$b$ después"], ["(1, 1)", "1", "0", "(1, 1)", "1"],
                             ["(0, 0)", "0", "1", "(1, 1)", "0"], ["(0, 1)", "0", "1", "(1, 0)", "−1"]],
                  [150, 80, 110, 230, 240], tam=30, alto_fila=62),
        ),
        Paso(
            "Tras diez correcciones, en la sexta época ya no hay errores. La recta deja al punto uno, uno solo de su lado. "
            "El punto uno, cero queda justo encima, y cuenta como cero.",
            Terminal(110, 520, "... 10 correcciones en 5 épocas\népoca 6: ningún error, se detiene\n"
                     "w = [2. 1.]   b = -2.0\nresultado: [0 0 0 1]   AND: [0 0 0 1]", 26, ancho=810),
            compuerta(1090, 270, 650, 520, (0, 0, 0, 1), rectas=[(-2, 2, TINTA)], id="r2"),
            ocultar=["r1"],
        ),
        Paso(
            "Y hay una garantía, el teorema de convergencia del perceptrón: si existe una recta que separa las clases, "
            "la regla encuentra una en un número finito de pasos.",
            Caja(110, 760, 810, 100, "**Si existe la recta, la regla encuentra una**", VERDE, 34),
        ),
    ),
    Escena(
        "Tu turno 2: el perceptrón completo",
        Paso(
            "Tu turno. Escribe el mismo bucle para cualquier cantidad de columnas. Guarda cuántos errores hubo en "
            "cada época, y termina antes si una época no tiene ninguno. La prueba lo usa con pingüinos. Pausa el video.",
            Codigo(130, 280, "def perceptron_fit(X, y, epochs=20, lr=1.0):\n    # TODO\n    raise NotImplementedError", 36),
            Caja(130, 560, 1660, 140, "Devuelve `w, b, errors` · $w$ empieza con `np.zeros(X.shape[1])`", VERDE, 38),
            pausa=1.2,
        ),
    ),
    Escena(
        "Con pingüinos de verdad",
        Paso(
            "Con datos reales: pingüinos Adelie y Gentoo, según el pico y la aleta. "
            "En scikit-learn, esta misma regla se usa con fit y predict.",
            Dispersion(190, 250, 760, 560, [(pico_z[~es_gentoo], aleta_z[~es_gentoo]), (pico_z[es_gentoo], aleta_z[es_gentoo])],
                       colores=DOS, nombres=["Adelie", "Gentoo"], rx=(-2.2, 2.6), ry=(-2.5, 3.2), marcas_x=(-2, -1, 0, 1, 2),
                       marcas_y=(-2, -1, 0, 1, 2), etiqueta_x="profundidad del pico (estandarizada)",
                       etiqueta_y="largo de la aleta (estandarizado)", radio=6,
                       curvas=[recta(-P_W[0] / P_W[1], -P_B / P_W[1], GRIS, hasta=1.5)]),
            Codigo(1040, 270, "clf = Perceptron(random_state=SEED)\nclf.fit(X_train, y_train)", 28, ancho=780),
        ),
        Paso(
            "La línea base acierta el cincuenta y cinco por ciento. El perceptrón, todos los de prueba: "
            "estas dos especies se separan con una recta.",
            Terminal(1040, 490, "línea base: accuracy 0.551\nperceptrón: accuracy 1.000", 30, ancho=780),
            pausa=0.8,
        ),
    ),
    # --------------------------------------------------------------- XOR
    Seccion("Parte 3 · El muro: XOR"),
    Escena(
        "Los interruptores de la escalera",
        Paso(
            "Piensa en una escalera, con un interruptor abajo y otro arriba. Si están iguales, la luz está "
            "apagada. Si están distintos, encendida. Esa compuerta se llama XOR.",
            Tabla(130, 260, [["abajo: $x_1$", "arriba: $x_2$", "luz: XOR"], ["0", "0", "0"], ["0", "1", "**1**"], ["1", "0", "**1**"],
                             ["1", "1", "0"]], [290, 290, 240], tam=36, alto_fila=88),
        ),
        Paso(
            "Cada color queda en esquinas opuestas del cuadrado. Intenta trazar una recta: "
            "siempre se te cuela un punto del otro color.",
            compuerta(1090, 270, 650, 520, (0, 1, 1, 0), rectas=[(-1, 0.5, ROJO), (-1, 1.5, ROJO), (1, -0.4, ROJO)]),
        ),
    ),
    Escena(
        "Ninguna recta alcanza",
        Paso(
            "Probemos doscientas mil rectas al azar. Ninguna llega a cuatro de cuatro, y la regla del perceptrón "
            "nunca termina. No es mala suerte.",
            Terminal(130, 250, "de 200 000 rectas al azar:\n  1 de 4: 25415 rectas\n  2 de 4: 148834 rectas\n  3 de 4: 25751 rectas\n  4 de 4: ninguna", 32, ancho=800),
        ),
        Paso(
            "Las cuatro filas de la tabla piden estas cuatro cosas a la vez. Suma las dos del medio: sale mayor que cero. "
            "Suma la primera y la última: la misma cantidad sale menor o igual que cero. No existe ninguna recta.",
            Formula(1020, 260, r"b \le 0", 42),
            Formula(1020, 330, r"w_2 + b > 0", 42),
            Formula(1020, 400, r"w_1 + b > 0", 42),
            Formula(1020, 470, r"w_1 + w_2 + b \le 0", 42),
            Formula(130, 620, r"\textcolor{azul}{w_1 + w_2 + 2b > 0} \qquad \text{y a la vez} \qquad \textcolor{rojo}{w_1 + w_2 + 2b \le 0}", 46),
            Caja(1400, 610, 400, 110, "**Imposible**", ROJO, 42),
        ),
        Paso(
            "En mil novecientos sesenta y nueve, Minsky y Papert publicaron un libro con estos límites, "
            "y eso contribuyó a que el entusiasmo por las redes neuronales cayera durante años.",
            Texto(130, 790, "1969: Perceptrons, de Minsky y Papert. La salida: poner una neurona después de otra.", 36, GRIS, ancho=1660),
        ),
    ),
    # --------------------------------------------------------------- dos capas
    Seccion("Parte 4 · Dos capas"),
    Escena(
        "Doblar la hoja",
        Paso(
            "Toma una hoja con los cuatro puntos dibujados. Con un solo corte recto de tijera, no puedes separar los colores.",
            Caja(160, 300, 460, 460, "", GRIS, relleno="#FFFFFF"),
            Circulo(160, 300, 30, "", NARANJA),
            Circulo(620, 760, 30, "", NARANJA),
            Circulo(160, 760, 30, "", AZUL),
            Circulo(620, 300, 30, "", AZUL),
            Texto(390, 800, "un corte recto no alcanza", 34, GRIS, alinear="centro"),
        ),
        Paso(
            "Pero si doblas la hoja por la diagonal, las dos esquinas naranjas quedan una encima de la otra.",
            Flecha(680, 530, 800, 530, GRIS, 6, etiqueta="doblar", tam=30),
            Linea(860, 300, 860, 760, GRIS, 5),
            Linea(860, 300, 1320, 300, GRIS, 5),
            Linea(860, 760, 1320, 300, GRIS, 5),
            Circulo(860, 760, 30, "", AZUL),
            Circulo(1320, 300, 30, "", AZUL),
            Circulo(860, 300, 34, "2", NARANJA, 30),
        ),
        Paso(
            "Ahora un solo corte basta. Una capa oculta hace eso: cambia la forma de los datos para que una recta alcance.",
            Linea(830, 530, 1090, 270, ROJO, 7),
            Texto(1105, 218, "un solo corte", 32, ROJO),
            Caja(1400, 400, 400, 280, "**La capa oculta**\ndobla el espacio", VERDE, 38),
        ),
    ),
    Escena(
        "XOR con compuertas que ya tenemos",
        Paso(
            "Con compuertas: XOR es alguna encendida, y no las dos. Las dos neuronas del medio son OR y NAND, "
            "lo contrario de AND. Y la salida es una compuerta AND.",
            *enlace((150, 380, 58), (510, 380, 72)),
            *enlace((150, 380, 58), (510, 680, 72)),
            *enlace((150, 680, 58), (510, 380, 72)),
            *enlace((150, 680, 58), (510, 680, 72)),
            *enlace((510, 380, 72), (880, 530, 72)),
            *enlace((510, 680, 72), (880, 530, 72)),
            Circulo(150, 380, 58, "$x_1$", AZUL, 40),
            Circulo(150, 680, 58, "$x_2$", AZUL, 40),
            Circulo(510, 380, 72, "OR", VERDE, 34),
            Circulo(510, 680, 72, "NAND", VERDE, 28),
            Circulo(880, 530, 72, "AND", NARANJA, 32),
            Texto(150, 250, "entrada", 32, GRIS, alinear="centro"),
            Texto(510, 250, "capa oculta", 32, VERDE, alinear="centro"),
            Texto(880, 250, "salida", 32, NARANJA, alinear="centro"),
        ),
        Paso(
            "Sigue la tabla: el resultado es XOR. Y mira las columnas del medio: las entradas cero, uno y uno, cero, "
            "que estaban en esquinas opuestas, caen en el mismo punto. Esa es la hoja doblada.",
            Tabla(1030, 290, [["$x_1$", "$x_2$", "$h_1$", "$h_2$", "salida"], ["0", "0", "0", "1", "0"], ["0", "1", "1", "1", "**1**"],
                              ["1", "0", "1", "1", "**1**"], ["1", "1", "1", "0", "0"]], [130, 130, 130, 130, 180], tam=36, alto_fila=84),
            Marco(1285, 462, 262, 164, ROJO),
        ),
    ),
    Escena(
        "La capa oculta cambia el espacio",
        Paso(
            "Antes, ninguna recta servía. Después, en la capa oculta, los dos naranjas son uno solo, y una recta los separa.",
            Texto(469, 190, "**Antes: la entrada**", 36, alinear="centro"),
            compuerta(190, 280, 620, 420, (0, 1, 1, 0)),
            Texto(1409, 190, "**Después: la capa oculta**", 36, alinear="centro"),
            Dispersion(1130, 280, 620, 420, [((0, 1), (1, 0)), ((1,), (1,))], colores=DOS, nombres=DA, rx=(-0.5, 1.5), ry=(-0.5, 1.9),
                       marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x="$h_1$ (OR)", etiqueta_y="$h_2$ (NAND)", radio=15,
                       curvas=[recta(-1, 1.5, GRIS)]),
        ),
        Paso(
            "Acabamos de armar un perceptrón multicapa, o MLP. La capa del medio se llama oculta porque "
            "no aparece ni en la entrada ni en la respuesta.",
            Texto(960, 840, "**Perceptrón multicapa** (multilayer perceptron, MLP) · **capa oculta** (hidden layer)", 36, alinear="centro"),
        ),
    ),
    Escena(
        "La propagación hacia adelante, en papel",
        Paso(
            "Calcular la salida capa por capa se llama propagación hacia adelante. Hagámosla a mano. "
            "Las neuronas ocultas usan ReLU, que convierte lo negativo en cero.",
            *enlace((130, 370, 58), (510, 370, 62), etiqueta="$1$", t=0.5),
            *enlace((130, 370, 58), (510, 690, 62), etiqueta="$-1$", t=0.25, dx=46, dy=-30),
            *enlace((130, 690, 58), (510, 370, 62), etiqueta="$2$", t=0.25, dx=40, dy=0),
            *enlace((130, 690, 58), (510, 690, 62), etiqueta="$1$", t=0.5),
            *enlace((510, 370, 62), (880, 530, 62), etiqueta="$2$", t=0.5, dx=14, dy=-48),
            *enlace((510, 690, 62), (880, 530, 62), etiqueta="$-1$", t=0.5, dx=-18, dy=-50),
            Circulo(130, 370, 58, "$2$", AZUL, 44),
            Circulo(130, 690, 58, "$-1$", AZUL, 44),
            Texto(130, 250, "$x = (2, -1)$", 34, AZUL, alinear="centro"),
            Circulo(510, 370, 62, "$h_1$", GRIS, 40, id="h1"),
            Circulo(510, 690, 62, "$h_2$", GRIS, 40, id="h2"),
            Circulo(880, 530, 62, "$z$", GRIS, 40, id="z"),
            Texto(510, 265, "$b = 1$", 30, TINTA, alinear="centro"),
            Texto(510, 760, "$b = 0$", 30, TINTA, alinear="centro"),
            Texto(880, 600, "$b = -0.5$", 30, TINTA, alinear="centro"),
            Caja(1030, 240, 790, 110, "$\\text{ReLU}(z) = \\max(z, 0)$", NARANJA, 40),
        ),
        Paso(
            "La primera neurona: uno por dos, más dos por menos uno, más uno. Da uno, y pasa tal cual.",
            Formula(1030, 390, r"h_1:\ 1 \cdot 2 + 2 \cdot (-1) + 1 = 1 \ \rightarrow\ \textcolor{verde}{1}", 42),
            Circulo(510, 370, 62, "$1$", VERDE, 48),
            ocultar=["h1"],
        ),
        Paso(
            "La segunda da menos tres. Es negativo: ReLU lo convierte en cero.",
            Formula(1030, 480, r"h_2:\ -1 \cdot 2 + 1 \cdot (-1) + 0 = -3 \ \rightarrow\ \textcolor{rojo}{0}", 42),
            Circulo(510, 690, 62, "$0$", ROJO, 48),
            ocultar=["h2"],
        ),
        Paso(
            "La salida: dos por uno, más menos uno por cero, menos cero coma cinco. Da uno coma cinco.",
            Formula(1030, 570, r"z:\ 2 \cdot 1 + (-1) \cdot 0 - 0.5 = \textcolor{morado}{1.5}", 42),
            Circulo(880, 530, 62, "$1.5$", MORADO, 40),
            ocultar=["z"],
        ),
        Paso(
            "Y la sigmoide lo vuelve una probabilidad: cero coma ochenta y dos. "
            "Esta es la cuenta que una red hace millones de veces.",
            Formula(1030, 670, r"\sigma(1.5) = \dfrac{1}{1 + e^{-1.5}} = \textcolor{verde}{0.818}", 46),
            Caja(1030, 810, 790, 100, "La cuenta que una red hace **millones de veces**", VERDE, 34),
            pausa=0.8,
        ),
    ),
    Escena(
        "Tu turno 3: una red en papel",
        Paso(
            "Tu turno. Con papel y lápiz, calcula esta otra red con la entrada uno, dos, y escribe los tres números "
            "en el notebook. ¿Qué clase dice? ¿Qué le pasó a la primera neurona? Pausa el video.",
            Formula(960, 260, r"h_1 = \text{ReLU}(1 \cdot x_1 - 2 \cdot x_2 + 1) \qquad h_2 = \text{ReLU}(-1 \cdot x_1 + 3 \cdot x_2 + 0)", 42, alinear="centro"),
            Formula(960, 360, r"z = 1 \cdot h_1 - 0.5 \cdot h_2 + 2 \qquad\quad x = (1, 2)", 42, alinear="centro"),
            Codigo(130, 480, "paper_h1 = None  # TODO\npaper_h2 = None  # TODO\npaper_z = None   # TODO", 36),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- matrices
    Seccion("Parte 5 · La red en matrices"),
    Escena(
        "De neuronas a matrices",
        Paso(
            "Para redes grandes, usamos lo de la semana dos: matriz por vector es una capa. "
            "Cada fila de la matriz es una neurona.",
            Formula(130, 260, r"\textcolor{azul}{W_1} = \begin{pmatrix} 1 & 2 \\ -1 & 1 \end{pmatrix} \quad "
                    r"b_1 = \begin{pmatrix} 1 \\ 0 \end{pmatrix} \qquad\qquad \textcolor{verde}{W_2} = \begin{pmatrix} 2 & -1 \end{pmatrix} \quad b_2 = -0.5", 46),
        ),
        Paso(
            "Cada capa es una suma con pesos, y una función aplicada a cada número.",
            Formula(130, 410, r"h = \text{ReLU}(\textcolor{azul}{W_1} x + b_1) \qquad\qquad z = \textcolor{verde}{W_2} h + b_2", 50),
        ),
        Paso(
            "Con un lote de cuatro entradas, en código son dos líneas. La primera fila es nuestra cuenta en papel.",
            Codigo(130, 530, "H = relu(X @ W1.T + b1)\nZ = H @ W2.T + b2", 34, ancho=720),
            Terminal(130, 710, "X (4, 2) → H (4, 2) → Z (4, 1)\nZ = [1.5, 1.5, 7.5, -1.5]", 30, ancho=720),
            Matriz(980, 570, 4, 2, 58, AZUL, etiqueta="`X`"),
            Flecha(1110, 690, 1190, 690, GRIS, 5),
            Matriz(1210, 570, 4, 2, 58, VERDE, etiqueta="`H`"),
            Flecha(1340, 690, 1420, 690, GRIS, 5),
            Matriz(1440, 570, 4, 1, 58, NARANJA, etiqueta="`Z`"),
            pausa=0.8,
        ),
    ),
    Escena(
        "Las piezas de la semana 1",
        Paso(
            "Las clases de la semana uno son exactamente esto. Con ellas armamos una red para los dígitos: "
            "sesenta y cuatro píxeles, treinta y dos ocultas, diez salidas.",
            Codigo(130, 260, "net = Sequential(\n    Linear(64, 32, rng), ReLU(),\n    Linear(32, 10, rng),\n)\nscores = net(X_dig_test)", 34, ancho=800),
            Terminal(1000, 260, "entrada: (450, 64)\npuntajes: (450, 10)", 32, ancho=790),
        ),
        Paso(
            "Una capa de a entradas y ce salidas tiene a por ce pesos y ce sesgos. De tres a cuatro: dieciséis.",
            Formula(1000, 460, r"a \cdot c + c \qquad\quad 3 \cdot 4 + 4 = 16", 46),
        ),
        Paso(
            "Con pesos al azar acierta el doce por ciento, casi como adivinar. Es un piano sin afinar: "
            "la forma correcta y los números equivocados. Encontrar esos números es entrenar.",
            Terminal(130, 590, "línea base (la clase más común): accuracy 0.102\nred con pesos al azar:           accuracy 0.122", 30, ancho=1100),
            Caja(1300, 590, 490, 150, "Forma correcta,\n**números equivocados**", ROJO, 36),
        ),
    ),
    Escena(
        "Tu turno 4: la red sin clases",
        Paso(
            "Tu turno. Escribe la propagación con una lista de matrices y otra de sesgos, con ReLU después de cada capa menos "
            "la última. Y cuenta los parámetros de una red a partir de sus tamaños. Pausa el video.",
            Codigo(130, 280, "def mlp_forward(X, weights, biases):\n    # TODO\n    raise NotImplementedError\n\n"
                   "def count_mlp_params(sizes):\n    # TODO: por ejemplo, sizes = [64, 32, 10]\n    raise NotImplementedError", 32),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- activaciones
    Seccion("Parte 6 · Funciones de activación"),
    Escena(
        "El escalón no da pistas",
        Paso(
            "La función que va después de la suma se llama función de activación. El escalón es un interruptor de pared: apagado o encendido.",
            Grafica(130, 280, 760, 520, lambda z: 1.0 if z > 0 else 0.0, -5, 5, -0.5, 1.5, AZUL, etiqueta="escalón"),
        ),
        Paso(
            "Para entrenar necesitamos el gradiente, y el escalón es plano. Es como buscar la bajada con los ojos vendados "
            "en una meseta: mover un peso un poquito no cambia nada. Necesitamos funciones con pendiente.",
            Caja(1000, 300, 790, 170, "Su derivada es **0**", ROJO, 44, subtitulo="y en el salto, no existe"),
            Caja(1000, 530, 790, 170, "Una meseta plana:\nel gradiente **no da pistas**", NARANJA, 38),
        ),
    ),
    Escena(
        "Tres funciones con pendiente",
        Paso(
            "La sigmoide es un regulador de luz en vez de un interruptor: va de cero a uno, de forma suave.",
            Grafica(100, 285, 540, 350, sigmoide, -5, 5, -1.2, 1.2, AZUL),
            Texto(370, 195, "**Sigmoide**", 36, alinear="centro"),
            Formula(370, 665, r"\sigma(z) = \dfrac{1}{1 + e^{-z}}", 38, alinear="centro"),
            Caja(120, 790, 500, 110, "un regulador de luz", AZUL, 34),
        ),
        Paso(
            "La tangente hiperbólica, o tanh, tiene la misma forma, pero va de menos uno a uno y está centrada en cero.",
            Grafica(690, 285, 540, 350, np.tanh, -5, 5, -1.2, 1.2, VERDE),
            Texto(960, 195, "**Tanh**", 36, alinear="centro"),
            Formula(960, 665, r"\tanh(z) = \dfrac{e^{z} - e^{-z}}{e^{z} + e^{-z}}", 38, alinear="centro"),
            Caja(710, 790, 500, 110, "un regulador de −1 a 1", VERDE, 34),
        ),
        Paso(
            "Y ReLU es una válvula de una sola vía: deja pasar lo positivo tal cual, y bloquea lo negativo.",
            Grafica(1280, 285, 540, 350, relu, -5, 5, -1.2, 5, NARANJA),
            Texto(1550, 195, "**ReLU**", 36, alinear="centro"),
            Formula(1550, 680, r"\text{ReLU}(z) = \max(z, 0)", 38, alinear="centro"),
            Caja(1300, 790, 500, 110, "una válvula de una vía", NARANJA, 34),
        ),
    ),
    Escena(
        "Las pendientes",
        Paso(
            "Sus derivadas. La sigmoide llega como mucho a cero coma veinticinco, y lejos del centro, ella y tanh casi "
            "desaparecen: la neurona está saturada. ReLU vale uno en todo el lado positivo.",
            Texto(370, 195, "**Sigmoide:** máximo 0.25", 34, alinear="centro"),
            Grafica(100, 285, 540, 285, lambda z: sigmoide(z) * (1 - sigmoide(z)), -5, 5, -0.2, 1.2, AZUL),
            Texto(960, 195, "**Tanh:** máximo 1", 34, alinear="centro"),
            Grafica(690, 285, 540, 285, lambda z: 1 - np.tanh(z) ** 2, -5, 5, -0.2, 1.2, VERDE),
            Texto(1550, 195, "**ReLU:** 0 o 1", 34, alinear="centro"),
            Grafica(1280, 285, 540, 285, lambda z: 1.0 if z > 0 else 0.0, -5, 5, -0.2, 1.2, NARANJA),
        ),
        Paso(
            "Por la regla de la cadena, el gradiente es un producto de derivadas, una por capa. "
            "Es como el juego del teléfono: si cada persona repite el mensaje más bajito, al final no se oye nada.",
            Tabla(100, 620, [["capas", "sigmoide, en su mejor caso", "ReLU, lado positivo"], ["1", "0.25", "1"], ["5", "0.00098", "1"],
                             ["10", "**0.00000095**", "1"]], [160, 480, 380], tam=30, alto_fila=66),
        ),
        Paso(
            "Con diez sigmoides, el gradiente se multiplica por menos de una millonésima: se desvanece. "
            "ReLU no es gratis: una neurona siempre negativa deja de aprender.",
            Caja(1180, 620, 640, 130, "**Gradiente que se desvanece**", ROJO, 34, subtitulo="vanishing gradient"),
            Caja(1180, 770, 640, 130, "**Neurona muerta**", NARANJA, 34, subtitulo="el precio de ReLU: siempre negativa, siempre 0"),
        ),
    ),
    Escena(
        "Dónde va cada una",
        Paso(
            "En las capas ocultas, ReLU por defecto. En la última capa depende del problema: para un número, nada. "
            "Para dos clases, sigmoide. Y para varias, softmax.",
            Caja(130, 260, 1660, 130, "**Capas ocultas:** ReLU, por defecto", NARANJA, 42),
            Tabla(130, 450, [["Problema", "Última capa", "Se lee como"], ["Regresión", "nada", "un número"],
                             ["Dos clases", "sigmoide", "una probabilidad"], ["Varias clases", "softmax", "una probabilidad por clase"]],
                  [500, 460, 700], tam=38, alto_fila=88),
        ),
    ),
    Escena(
        "Tu turno 5: tanh desde cero",
        Paso(
            "Tu turno. Escribe tanh y su derivada usando solo la exponencial de NumPy. Un dato útil: la derivada es uno "
            "menos tanh al cuadrado. Pausa el video.",
            Codigo(130, 280, "def my_tanh(z):\n    # TODO\n    raise NotImplementedError\n\n"
                   "def my_tanh_derivative(z):\n    # TODO\n    raise NotImplementedError", 32),
            Formula(1120, 370, r"\tanh'(z) = 1 - \tanh^2(z)", 48),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- no linealidad
    Seccion("Parte 7 · Por qué importa la no linealidad"),
    Escena(
        "Tres zooms son un solo zoom",
        Paso(
            "¿Hace falta la activación? Piensa en una foto: tres zooms seguidos son un solo zoom. "
            "Con las capas lineales pasa igual.",
            Texto(130, 280, "$x$", 44),
            Flecha(180, 305, 260, 305),
            Caja(270, 240, 180, 130, "$W_1$", AZUL, 44),
            Flecha(450, 305, 530, 305),
            Caja(540, 240, 180, 130, "$W_2$", AZUL, 44),
            Flecha(720, 305, 800, 305),
            Caja(810, 240, 180, 130, "$W_3$", AZUL, 44),
            Flecha(990, 305, 1070, 305),
            Texto(1090, 280, "$y$", 44),
        ),
        Paso(
            "Multiplicar por una matriz tras otra es multiplicar por una sola. Cien capas lineales equivalen a una.",
            Formula(960, 430, r"W_3 \left( W_2 \left( W_1 x \right) \right) = \left( W_3 W_2 W_1 \right) x = \textcolor{rojo}{W} x", 56, alinear="centro"),
        ),
        Paso(
            "En el notebook, tres capas dan exactamente lo mismo que una matriz. Con ReLU de por medio, ya no.",
            Terminal(130, 590, "W_single: (1, 2)\n¿tres capas = una sola? True\n¿con ReLU también?      False", 34, ancho=900),
            Caja(1130, 590, 660, 200, "Sin activación, una red profunda\nes **una sola capa**", ROJO, 38),
        ),
    ),
    Escena(
        "XOR con ruido",
        Paso(
            "Llevemos XOR a datos: nubes de puntos en las cuatro esquinas. Tres redes de ocho neuronas ocultas, "
            "y solo cambia la activación. El método fit las entrena; cómo lo hace es el tema de la próxima clase.",
            Texto(339, 195, "**Sin activación**", 36, alinear="centro"),
            Regiones(130, 285, 480, 410, red_mlp(MLP_IDENTITY, lambda z: z), (-0.9, 1.9), (-0.9, 1.9), DOS, XOR_SERIES, radio=5,
                     marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x="$x_1$", etiqueta_y="$x_2$"),
        ),
        Paso(
            "Sin activación, dibuja una recta, y se queda en el cincuenta por ciento, como la línea base.",
            Texto(339, 800, "prueba: **0.500**\nlínea base: 0.510", 32, ROJO, alinear="centro"),
        ),
        Paso(
            "Con tanh, se dobla alrededor de las nubes: noventa y ocho por ciento.",
            Texto(949, 195, "**Con tanh**", 36, alinear="centro"),
            Regiones(740, 285, 480, 410, red_mlp(MLP_TANH, np.tanh), (-0.9, 1.9), (-0.9, 1.9), DOS, XOR_SERIES, radio=5,
                     marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x="$x_1$", etiqueta_y="$x_2$"),
            Texto(949, 800, "prueba: **0.980**", 32, VERDE, alinear="centro"),
        ),
        Paso(
            "Con ReLU, también, con fronteras de tramos rectos. La activación es lo único que cambió.",
            Texto(1559, 195, "**Con ReLU**", 36, alinear="centro"),
            Regiones(1350, 285, 480, 410, red_mlp(MLP_RELU, relu), (-0.9, 1.9), (-0.9, 1.9), DOS, XOR_SERIES, radio=5,
                     marcas_x=(0, 1), marcas_y=(0, 1), etiqueta_x="$x_1$", etiqueta_y="$x_2$"),
            Texto(1559, 800, "prueba: **0.980**", 32, VERDE, alinear="centro"),
            pausa=0.8,
        ),
    ),
    Escena(
        "Una regla plegable",
        Paso(
            "¿Cómo arma curvas una red con ReLU? Cada neurona es una recta con un doblez, como una regla de carpintero. "
            "Sumando reglas dobladas se puede seguir una curva. Con dos neuronas, todavía es una línea plana.",
            Formula(1320, 80, r"\hat{f}(x) = f(t_0) + \sum_i c_i \, \text{ReLU}(x - t_i)", 42, alinear="centro"),
            *panel_plegable(188, 2, "0.635"),
        ),
        Paso(
            "Sigamos al seno, con pesos calculados a mano. Con cuatro neuronas ya se parece, y con dieciséis no se distingue. "
            "Más dobleces, menos error.",
            *panel_plegable(622, 4, "0.136"),
            *panel_plegable(1056, 8, "0.033"),
            *panel_plegable(1490, 16, "0.008"),
        ),
        Paso(
            "Es el teorema de aproximación universal: con suficientes neuronas, una capa oculta se acerca tanto como "
            "quieras a cualquier función continua en un intervalo. Pero no dice cómo encontrar los pesos, "
            "y más neuronas también pueden memorizar el ruido.",
            Caja(100, 745, 840, 160, "**Aproximación universal**", VERDE, 36, subtitulo="los pesos existen..."),
            Caja(970, 745, 846, 160, "**...pero no dice cómo encontrarlos**", NARANJA, 36, subtitulo="y el sobreajuste de la semana 5 sigue valiendo"),
        ),
    ),
    Escena(
        "Todo junto: los dígitos",
        Paso(
            "Volvamos a los dígitos. Con pesos al azar, la red acertaba el doce por ciento. Entrenada, el noventa y siete.",
            Barras(130, 260, 1660, 480, ["línea base", "red al azar", "regresión logística", "red entrenada"], [0.102, 0.122, 0.969, 0.973],
                   [GRIS, ROJO, AZUL, VERDE], vmax=1.0, formato="{:.3f}", tam=34),
        ),
        Paso(
            "Pero la regresión logística, con una sola capa, casi empata: en estos dígitos limpios no hace falta una red. "
            "El modelo simple siempre va primero.",
            Caja(330, 830, 1260, 90, "**El modelo simple siempre va primero**: a veces no hace falta una red", AZUL, 34),
        ),
    ),
    # --------------------------------------------------------------- cierre
    Seccion("Cierre"),
    Escena(
        "Lo que vimos hoy",
        *[
            Paso(decir, *grupo)
            for decir, grupo in zip(
                [
                    "Repasemos. Un perceptrón es una recta, y su regla la encuentra si existe.",
                    "XOR no se separa con ninguna recta. Una capa oculta dobla la hoja: eso es un MLP.",
                    "En las capas ocultas, ReLU por defecto. Y sin no linealidad, muchas capas equivalen a una sola.",
                ],
                (lambda v: [v[0:2], v[2:4], v[4:6]])(vinetas(130, 250, [
                    "**Perceptrón:** suma con pesos más un escalón. Una recta.",
                    "**Su regla:** corrige solo cuando se equivoca; encuentra la recta si existe.",
                    "**XOR:** ninguna recta alcanza. Una capa oculta dobla la hoja: un **MLP**.",
                    "**Propagación hacia adelante:** $h = \\text{ReLU}(W_1 x + b_1)$, $z = W_2 h + b_2$.",
                    "**Activaciones:** el escalón no da pendiente; ReLU por defecto en las ocultas.",
                    "**Sin no linealidad,** muchas capas equivalen a una sola.",
                ], tam=38, sep=118, ancho=1650)),
            )
        ],
    ),
    Escena(
        "Para la semana 7",
        Paso(
            "Para la semana siete: entrega el B tres, regresión lineal y logística desde cero en NumPy. "
            "Va en la carpeta B tres de tu repositorio.",
            Vineta(130, 260, "**B3:** regresión lineal y logística desde cero en NumPy, comparadas con sklearn. "
                   "En `B3/`, al inicio de la semana 7.", 40, NARANJA, numero=1, ancho=1600),
        ),
        Paso(
            "Y la cuota semanal. La próxima clase abrimos la caja de fit: backpropagation, la regla de la cadena aplicada a una red entera. "
            "Es lo que faltaba en mil novecientos sesenta y nueve. ¡Nos vemos en la semana siete!",
            Vineta(130, 440, "**Cuota semanal** en ioai.artix.tech.", 40, NARANJA, numero=2, ancho=1600),
            Caja(130, 590, 1660, 150, "**Semana 7: backpropagation**", AZUL, 44, subtitulo="cómo se encuentran los pesos"),
            Texto(960, 830, "**¡Nos vemos en la semana 7!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
