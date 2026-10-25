"""Video de la semana 4: Nivel_Intermedio/clases/sem04_calculo.ipynb."""

from motor import (AZUL, GRIS, GRIS_CLARO, MORADO, NARANJA, ROJO, TINTA, VERDE, Barras, Caja, Circulo, Codigo,
                   Contornos, Dispersion, Escena, Flecha, Formula, Grafica, Marco, Matriz, Paso, Seccion, Tabla,
                   Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 4 · Cálculo y probabilidad para deep learning · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}

# Pingüinos de Palmer (los 342 con aleta, masa, pico y especie), copiados aquí porque el
# entorno de los videos no tiene pandas: aleta en mm, masa en unidades de 25 g, pico en décimas de mm.
ALETAS = [
    181, 186, 195, 193, 190, 181, 195, 193, 190, 186, 180, 182, 191, 198, 185, 195, 197, 184, 194, 174, 180,
    189, 185, 180, 187, 183, 187, 172, 180, 178, 178, 188, 184, 195, 196, 190, 180, 181, 184, 182, 195, 186,
    196, 185, 190, 182, 179, 190, 191, 186, 188, 190, 200, 187, 191, 186, 193, 181, 194, 185, 195, 185, 192,
    184, 192, 195, 188, 190, 198, 190, 190, 196, 197, 190, 195, 191, 184, 187, 195, 189, 196, 187, 193, 191,
    194, 190, 189, 189, 190, 202, 205, 185, 186, 187, 208, 190, 196, 178, 192, 192, 203, 183, 190, 193, 184,
    199, 190, 181, 197, 198, 191, 193, 197, 191, 196, 188, 199, 189, 189, 187, 198, 176, 202, 186, 199, 191,
    195, 191, 210, 190, 197, 193, 199, 187, 190, 191, 200, 185, 193, 193, 187, 188, 190, 192, 185, 190, 184,
    195, 193, 187, 201, 192, 196, 193, 188, 197, 198, 178, 197, 195, 198, 193, 194, 185, 201, 190, 201, 197,
    181, 190, 195, 181, 191, 187, 193, 195, 197, 200, 200, 191, 205, 187, 201, 187, 203, 195, 199, 195, 210,
    192, 205, 210, 187, 196, 196, 196, 201, 190, 212, 187, 198, 199, 201, 193, 203, 187, 197, 191, 203, 202,
    194, 206, 189, 195, 207, 202, 193, 210, 198, 211, 230, 210, 218, 215, 210, 211, 219, 209, 215, 214, 216,
    214, 213, 210, 217, 210, 221, 209, 222, 218, 215, 213, 215, 215, 215, 216, 215, 210, 220, 222, 209, 207,
    230, 220, 220, 213, 219, 208, 208, 208, 225, 210, 216, 222, 217, 210, 225, 213, 215, 210, 220, 210, 225,
    217, 220, 208, 220, 208, 224, 208, 221, 214, 231, 219, 230, 214, 229, 220, 223, 216, 221, 221, 217, 216,
    230, 209, 220, 215, 223, 212, 221, 212, 224, 212, 228, 218, 218, 212, 230, 218, 228, 212, 224, 214, 226,
    216, 222, 203, 225, 219, 228, 215, 228, 216, 215, 210, 219, 208, 209, 216, 229, 213, 230, 217, 230, 217,
    222, 214, 215, 222, 212, 213,
]
MASAS_25G = [
    150, 152, 130, 138, 146, 145, 187, 139, 170, 132, 148, 128, 152, 176, 148, 138, 180, 133, 168, 136, 144,
    152, 158, 152, 152, 142, 128, 126, 158, 130, 156, 132, 156, 133, 166, 158, 142, 132, 186, 126, 156, 124,
    176, 120, 184, 137, 119, 138, 166, 140, 172, 138, 162, 116, 148, 142, 152, 114, 150, 126, 176, 144, 162,
    114, 158, 134, 164, 122, 178, 144, 156, 142, 166, 148, 170, 148, 156, 142, 160, 128, 188, 152, 168, 134,
    142, 152, 140, 158, 144, 142, 172, 136, 178, 132, 172, 148, 174, 116, 164, 149, 189, 123, 170, 117, 142,
    150, 156, 127, 191, 153, 184, 128, 171, 156, 163, 116, 151, 134, 133, 126, 140, 138, 155, 122, 160, 131,
    172, 122, 160, 133, 140, 140, 179, 137, 156, 127, 159, 136, 170, 136, 139, 122, 149, 120, 146, 170, 139,
    138, 150, 148, 160, 140, 156, 146, 141, 149, 158, 130, 150, 166, 148, 152, 151, 148, 162, 143, 162, 132,
    148, 138, 176, 144, 136, 116, 152, 132, 166, 136, 152, 148, 182, 128, 172, 134, 164, 144, 156, 154, 192,
    108, 180, 158, 146, 142, 140, 147, 178, 136, 172, 130, 147, 133, 158, 144, 162, 134, 138, 130, 162, 152,
    141, 158, 146, 146, 160, 136, 151, 164, 151, 180, 228, 178, 228, 216, 182, 192, 208, 176, 206, 186, 222,
    186, 234, 168, 234, 166, 252, 192, 214, 228, 200, 176, 202, 200, 204, 164, 226, 184, 222, 210, 188, 202,
    242, 206, 216, 198, 210, 174, 214, 158, 228, 172, 190, 222, 196, 168, 216, 204, 212, 194, 212, 176, 200,
    196, 202, 172, 200, 178, 222, 168, 212, 176, 226, 188, 228, 186, 232, 188, 222, 190, 200, 204, 208, 188,
    232, 184, 240, 190, 238, 185, 218, 189, 214, 190, 224, 184, 212, 195, 222, 198, 216, 190, 226, 194, 208,
    197, 195, 185, 210, 194, 224, 199, 220, 189, 220, 188, 220, 183, 220, 200, 238, 186, 220, 175, 234, 195,
    240, 197, 194, 230, 208, 216,
]
PICOS_DECIMAS = [
    391, 395, 403, 367, 393, 389, 392, 341, 420, 378, 378, 411, 386, 346, 366, 387, 425, 344, 460, 378, 377,
    359, 382, 388, 353, 406, 405, 379, 405, 395, 372, 395, 409, 364, 392, 388, 422, 376, 398, 365, 408, 360,
    441, 370, 396, 411, 375, 360, 423, 396, 401, 350, 420, 345, 414, 390, 406, 365, 376, 357, 413, 376, 411,
    364, 416, 355, 411, 359, 418, 335, 397, 396, 458, 355, 428, 409, 372, 362, 421, 346, 429, 367, 351, 373,
    413, 363, 369, 383, 389, 357, 411, 340, 396, 362, 408, 381, 403, 331, 432, 350, 410, 377, 378, 379, 397,
    386, 382, 381, 432, 381, 456, 397, 422, 396, 427, 386, 373, 357, 411, 362, 377, 402, 414, 352, 406, 388,
    415, 390, 441, 385, 431, 368, 375, 381, 411, 356, 402, 370, 397, 402, 406, 321, 407, 373, 390, 392, 366,
    360, 378, 360, 415, 465, 500, 513, 454, 527, 452, 461, 513, 460, 513, 466, 517, 470, 520, 459, 505, 503,
    580, 464, 492, 424, 485, 432, 506, 467, 520, 505, 495, 464, 528, 409, 542, 425, 510, 497, 475, 476, 520,
    469, 535, 490, 462, 509, 455, 509, 508, 501, 490, 515, 498, 481, 514, 457, 507, 425, 522, 452, 493, 502,
    456, 519, 468, 457, 558, 435, 496, 508, 502, 461, 500, 487, 500, 476, 465, 454, 467, 433, 468, 409, 490,
    455, 484, 458, 493, 420, 492, 462, 487, 502, 451, 465, 463, 429, 461, 445, 478, 482, 500, 473, 428, 451,
    596, 491, 484, 426, 444, 440, 487, 427, 496, 453, 496, 505, 436, 455, 505, 449, 452, 466, 485, 451, 501,
    465, 450, 438, 455, 432, 504, 453, 462, 457, 543, 458, 498, 462, 495, 435, 507, 477, 464, 482, 465, 464,
    486, 475, 511, 452, 452, 491, 525, 474, 500, 449, 508, 434, 513, 475, 521, 475, 522, 455, 495, 445, 508,
    494, 469, 484, 511, 485, 559, 472, 491, 473, 468, 417, 534, 433, 481, 505, 498, 435, 515, 462, 551, 445,
    488, 472, 468, 504, 452, 499,
]
ESPECIES = (
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"
    "CCCCCCCCCCCCCCCCCCCGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
    "GGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG"
)

aleta = np.array(ALETAS, float)
x_aleta = (aleta - aleta.mean()) / aleta.std()  # aleta estandarizada, como en el notebook
masa = np.array(MASAS_25G) * 25 / 1000          # masa en kg
especie = np.array(list(ESPECIES))
pico = np.array(PICOS_DECIMAS) / 10
x_pico = (pico - pico.mean()) / pico.std()


def mse(w, b):
    """El MSE de la recta w·x + b sobre los pingüinos; acepta arreglos de w y b (para el paisaje)."""
    w, b = np.asarray(w, float), np.asarray(b, float)
    return np.mean((w[..., None] * x_aleta + b[..., None] - masa) ** 2, axis=-1)


W_MEJOR, B_MEJOR = np.polyfit(x_aleta, masa, 1)


def descenso(w, b, lr, pasos):
    camino = [(w, b)]
    for _ in range(pasos):
        r = w * x_aleta + b - masa
        w, b = w - lr * np.mean(2 * r * x_aleta), b - lr * np.mean(2 * r)
        camino.append((w, b))
    return camino


CAMINO = descenso(-0.5, 2.5, 0.1, 60)


def camino_1d(lr, pasos, w0=0.0):
    ws = [w0]
    for _ in range(pasos):
        ws.append(ws[-1] - lr * 2 * (ws[-1] - 3))
    return ws


def parabola(w):
    return (w - 3) ** 2


def nubes_de_pinguinos(x, y, w, h, **kw):
    return Dispersion(x, y, w, h, [(x_aleta, masa)], colores=(GRIS,), rx=(-2.3, 2.4), ry=(2.5, 6.5),
                      marcas_x=(-2, -1, 0, 1, 2), marcas_y=(3, 4, 5, 6), etiqueta_x="aleta (estandarizada)",
                      etiqueta_y="masa (kg)", radio=5, **kw)


def paisaje(x, y, w, h, caminos=(), marcas=(), id=None):
    return Contornos(x, y, w, h, mse, (-1.0, 2.4), (2.0, 6.0), [0.2, 0.3, 0.5, 0.8, 1.2, 1.8, 2.6, 3.6],
                     caminos=caminos, marcas=marcas, marcas_x=(-1, 0, 1, 2), marcas_y=(2, 3, 4, 5, 6),
                     etiqueta_x="w (pendiente)", etiqueta_y="b (sesgo)", id=id)


def panel_tasa(i, lr, titulo, color):
    x = 130 + i * 430
    return [
        Grafica(x, 330, 380, 300, parabola, -1.5, 7.5, -1, 22, camino=camino_1d(lr, 15)),
        Texto(x + 190, 670, f"η = {lr}", 36, color, alinear="centro"),
        Texto(x + 190, 725, titulo, 30, GRIS, alinear="centro"),
    ]


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la cuarta clase del nivel intermedio: cálculo y probabilidad para deep learning. "
            "Es la matemática con la que se entrena toda red neuronal.",
            Texto(960, 250, "SEMANA 4 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**Cálculo y probabilidad**", 92, alinear="centro"),
            Texto(960, 460, "derivadas, gradiente, descenso de gradiente y entropía cruzada", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Vamos a ver cuatro ideas: la derivada y el gradiente, la regla de la cadena, el descenso de gradiente, "
            "que es cómo aprende un modelo, y la softmax con la entropía cruzada.",
            Caja(110, 640, 400, 200, "**1**\nDerivada y gradiente", AZUL, 38),
            Caja(560, 640, 400, 200, "**2**\nRegla de la cadena", VERDE, 38),
            Caja(1010, 640, 400, 200, "**3**\nDescenso de gradiente", MORADO, 38),
            Caja(1460, 640, 400, 200, "**4**\nSoftmax y entropía cruzada", NARANJA, 36),
        ),
    ),
    # --------------------------------------------------------------- derivada
    Seccion("Parte 1 · La derivada"),
    Escena(
        "La derivada: un velocímetro",
        Paso(
            "Imagina que vas en auto. El velocímetro no dice dónde estás, sino qué tan rápido cambia tu posición "
            "en este instante. La derivada es eso: cuánto cambia una función cuando su entrada se mueve un poquito.",
            Caja(130, 260, 760, 200, "**Velocímetro**", AZUL, 40, subtitulo="no dice dónde estás:\nqué tan rápido cambia tu posición"),
            Caja(1030, 260, 760, 200, "**Derivada**", VERDE, 40, subtitulo="cuánto cambia f\ncuando x se mueve un poquito"),
        ),
        Paso(
            "Un ejemplo con números: equis al cuadrado, en equis igual tres. Movemos equis un poquito, una distancia h, "
            "y medimos cuánto sube la función, dividido por h.",
            Formula(130, 530, r"f'(x) \approx \frac{f(x + h) - f(x)}{h}", 50),
            Tabla(1030, 510, [["h", "pendiente"], ["1", "7"], ["0.1", "6.1"]], [300, 460], tam=32, id="t"),
        ),
        Paso(
            "Con h igual a uno da siete; con un décimo, seis coma uno; con un milésimo, seis coma cero cero uno. "
            "Se acerca a seis, que es lo que dice la regla de la potencia: la derivada de equis al cuadrado es dos equis.",
            Tabla(1030, 510, [["h", "pendiente"], ["1", "7"], ["0.1", "6.1"], ["0.01", "6.01"], ["0.001", "6.001"]],
                  [300, 460], tam=32),
            Texto(130, 880, r"$f(x) = x^2 \;\Rightarrow\; f'(x) = 2x, \quad f'(3) = \textcolor{verde}{6}$", 48),
            ocultar=["t"],
        ),
    ),
    Escena(
        "La tangente",
        Paso(
            "En un gráfico, la derivada es la pendiente de la recta tangente, la que imita la curva de cerca. "
            "En equis igual tres, la pendiente es seis; en menos uno, es menos dos.",
            Grafica(150, 280, 900, 600, lambda x: x**2, -3, 4.5, -2, 20, rectas=[(3, NARANJA), (-1, VERDE)], id="g"),
            Texto(1180, 330, "en `x = 3`: pendiente **6**", 40, NARANJA, acento=NARANJA),
            Texto(1180, 410, "en `x = -1`: pendiente **−2**", 40, VERDE, acento=VERDE),
        ),
        Paso(
            "Y en el fondo del valle, en equis igual cero, la tangente es horizontal: la derivada vale cero. "
            "Recuérdalo: buscar el mínimo de una función es buscar dónde la pendiente es cero.",
            Grafica(150, 280, 900, 600, lambda x: x**2, -3, 4.5, -2, 20,
                    rectas=[(3, NARANJA), (-1, VERDE), (0, MORADO)]),
            Texto(1180, 490, "en `x = 0`: pendiente **0**", 40, MORADO, acento=MORADO),
            Caja(1180, 620, 620, 170, "En el mínimo, $f'(x) = 0$", MORADO, 40),
            ocultar=["g"],
        ),
    ),
    Escena(
        "La diferencia centrada",
        Paso(
            "Para calcular derivadas con la computadora hay una fórmula mejor: mirar a los dos lados de equis, "
            "un poquito a la izquierda y un poquito a la derecha. Se llama diferencia centrada.",
            Formula(130, 270, r"f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}", 54),
        ),
        Paso(
            "Con h igual a diez a la menos cinco, la fórmula hacia adelante se equivoca en el quinto decimal, "
            "y la centrada recién en el undécimo. Pero con un h demasiado chico las dos empeoran: "
            "la computadora guarda solo unos dieciséis dígitos.",
            Terminal(130, 470, "       h    error hacia adelante    error centrada\n"
                     "   1e-01                1.41e-01          4.53e-03\n"
                     "   1e-03                1.36e-03          4.53e-07\n"
                     "   1e-05                1.36e-05          5.86e-11\n"
                     "   1e-08                6.60e-09          6.60e-09\n"
                     "   1e-12                4.32e-04          2.10e-04", 30, ancho=1660),
            Marco(122, 613, 1676, 46, VERDE),
            Marco(122, 700, 1676, 46, ROJO),
        ),
    ),
    Escena(
        "Tu turno 1: la derivada numérica",
        Paso(
            "Tu turno. Escribe la función de derivada numérica con la diferencia centrada. La verificación la compara "
            "con derivadas que ya conoces, como la del seno, que es el coseno. Pausa el video y resuélvelo en el notebook.",
            Codigo(130, 280, "def numerical_derivative(f, x, h=1e-5):\n    # TODO: la diferencia centrada\n"
                   "    raise NotImplementedError", 34),
            Caja(130, 600, 1660, 140, "Debe funcionar con un número y con un arreglo de NumPy.", VERDE, 38),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- gradiente
    Seccion("Parte 2 · Derivadas parciales y gradiente"),
    Escena(
        "Derivadas parciales: una perilla a la vez",
        Paso(
            "Un modelo tiene muchos parámetros, así que su error depende de muchas variables. Piensa en un ecualizador "
            "con varias perillas: para saber qué hace una, la mueves sola y dejas las demás quietas. "
            "Eso es una derivada parcial.",
            Caja(130, 260, 1660, 170, "**Derivada parcial:** mover una sola variable, con las demás quietas", AZUL, 40),
        ),
        Paso(
            "Ejemplo: equis al cuadrado más tres ye al cuadrado. Si ye queda quieta, tres ye al cuadrado es una constante, "
            "y la derivada parcial respecto de equis es dos equis. Si equis queda quieta, la derivada respecto de ye "
            "es seis ye.",
            Formula(960, 500, r"f(x, y) = x^2 + 3y^2", 54, alinear="centro"),
            Formula(500, 640, r"\frac{\partial f}{\partial x} = \textcolor{azul}{2x}", 54, alinear="centro"),
            Formula(1420, 640, r"\frac{\partial f}{\partial y} = \textcolor{verde}{6y}", 54, alinear="centro"),
        ),
        Paso(
            "En el punto uno, dos, valen dos y doce. Moverse en ye cambia la función seis veces más que moverse en equis.",
            Texto(960, 840, r"en $(1, 2)$:  $\textcolor{azul}{2}$  y  $\textcolor{verde}{12}$", 46, alinear="centro"),
        ),
    ),
    Escena(
        "El gradiente: una montaña con niebla",
        Paso(
            "El gradiente junta todas las derivadas parciales en un vector. Para nuestra función es dos equis, seis ye. "
            "En el punto uno, dos, es dos, doce.",
            Formula(130, 260, r"\nabla f(x, y) = (2x,\; 6y)", 50),
            Formula(130, 380, r"\nabla f(1, 2) = (2,\; 12)", 50),
        ),
        Paso(
            "Imagina que estás en una montaña con niebla. No ves el valle, pero con los pies sientes hacia dónde "
            "se inclina el suelo. El gradiente es esa información local: apunta hacia donde la función sube más rápido.",
            Contornos(960, 250, 840, 0, lambda a, b: a**2 + 3 * b**2, (-3, 3), (-2, 2), [0.5, 2, 4, 7, 11, 16],
                      flechas=[(a, b, 0.05 * 2 * a, 0.05 * 6 * b) for a in np.linspace(-2.5, 2.5, 6)
                               for b in np.linspace(-1.3, 1.3, 5)],
                      marcas_x=(-3, -2, -1, 0, 1, 2, 3), marcas_y=(-2, -1, 0, 1, 2), etiqueta_x="x", etiqueta_y="y",
                      igual=True),
        ),
        Paso(
            "En este mapa de curvas de nivel, cada línea une los puntos donde la función vale lo mismo, "
            "como en un mapa topográfico. Las flechas rojas son el gradiente: cruzan las curvas en ángulo recto, "
            "cuesta arriba. Para bajar, se camina en sentido contrario.",
            Caja(130, 560, 680, 200, "$-\\nabla f$: **cuesta abajo**", ROJO, 42, subtitulo="hacia donde f baja más rápido"),
        ),
    ),
    Escena(
        "Tu turno 2: el gradiente",
        Paso(
            "Tu turno, con la función ge, igual a equis al cuadrado por ye, más tres ye. Primero calcula su gradiente a mano. "
            "Después escribe un gradiente numérico que sirva para cualquier función: mueve una coordenada a la vez, "
            "con la diferencia centrada. Pausa el video.",
            Formula(130, 260, r"g(x, y) = x^2 y + 3y", 50),
            Codigo(130, 400, "def grad_g(p):\n    # TODO: a mano, np.array([∂g/∂x, ∂g/∂y])\n\n"
                   "def numerical_gradient(f, p, h=1e-5):\n    # TODO: una coordenada a la vez; no modifiques p", 32),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- cadena
    Seccion("Parte 3 · La regla de la cadena"),
    Escena(
        "Engranajes",
        Paso(
            "Una red neuronal es una función dentro de otra, dentro de otra. Para derivarla usamos la regla de la cadena, "
            "y la mejor imagen son los engranajes.",
            Circulo(300, 420, 110, "A", AZUL, 60),
            Circulo(760, 420, 80, "B", VERDE, 54),
            Circulo(1140, 420, 60, "C", NARANJA, 48),
        ),
        Paso(
            "Si cada vuelta del engranaje A hace girar al B tres veces, y cada vuelta del B hace girar al C dos veces, "
            "entonces cada vuelta del A hace girar al C tres por dos: seis veces. Las tasas de cambio se multiplican.",
            Flecha(420, 420, 670, 420, GRIS, 6, etiqueta="**× 3**", tam=36),
            Flecha(850, 420, 1070, 420, GRIS, 6, etiqueta="**× 2**", tam=36),
            Texto(1290, 390, "A → C:  **× 6**", 52, NARANJA, acento=NARANJA),
        ),
        Paso(
            "En fórmula: la derivada de y respecto de x es la derivada de y respecto de u, por la derivada de u "
            "respecto de x.",
            Formula(960, 600, r"\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}", 56, alinear="centro"),
        ),
        Paso(
            "Ejemplo: tres equis más uno, al cuadrado. Llamamos u a tres equis más uno. La derivada de u al cuadrado "
            "es dos u, y la de u es tres. En equis igual uno, u vale cuatro: dos por cuatro, por tres, da veinticuatro.",
            Formula(130, 800, r"h(x) = (3x+1)^2: \quad \frac{dh}{dx} = \textcolor{verde}{2u} \cdot \textcolor{azul}{3}"
                    r" = 2 \cdot 4 \cdot 3 = \mathbf{24}", 44),
            Terminal(1300, 800, "regla de la cadena: 24\nnumérica:           24.0", 28, ancho=500),
        ),
    ),
    Escena(
        "Una neurona, eslabón por eslabón",
        Paso(
            "Ahora una neurona. Multiplica la entrada equis por un peso doble u, y le suma un sesgo be: eso es zeta. "
            "Después aplica la sigmoide, que aplasta cualquier número entre cero y uno: eso es la salida, a. "
            "Y su error, comparado con la respuesta ye, es a menos ye, al cuadrado.",
            Caja(130, 300, 260, 150, "$w$", AZUL, 48),
            Flecha(400, 375, 520, 375),
            Caja(530, 300, 360, 150, "$z = wx + b$", AZUL, 40),
            Flecha(900, 375, 1020, 375),
            Caja(1030, 300, 330, 150, r"$a = \sigma(z)$", VERDE, 40),
            Flecha(1370, 375, 1490, 375),
            Caja(1500, 300, 300, 150, "$L = (a - y)^2$", NARANJA, 36),
        ),
        Paso(
            "Con números pequeños: equis igual dos, ye igual uno, doble u igual cero coma cinco, y be igual menos uno. "
            "Entonces zeta vale cero, la salida vale cero coma cinco, y el error es cero coma veinticinco.",
            Texto(260, 480, "0.5", 40, AZUL, alinear="centro"),
            Texto(710, 480, "0", 40, AZUL, alinear="centro"),
            Texto(1195, 480, "0.5", 40, VERDE, alinear="centro"),
            Texto(1650, 480, "0.25", 40, NARANJA, alinear="centro"),
            Texto(130, 550, "con x = 2,  y = 1,  b = −1", 34, GRIS),
        ),
        Paso(
            "Cada eslabón tiene su derivada local. La del error respecto de la salida es dos por a menos ye: menos uno. "
            "La de la sigmoide da cero coma veinticinco. Y la de zeta respecto del peso es equis, o sea dos.",
            Texto(460, 670, r"$\dfrac{\partial z}{\partial w} = x = \textcolor{azul}{2}$", 40, alinear="centro"),
            Texto(960, 670, r"$\dfrac{\partial a}{\partial z} = \sigma(1-\sigma) = \textcolor{verde}{0.25}$", 40,
                  alinear="centro"),
            Texto(1440, 670, r"$\dfrac{\partial L}{\partial a} = 2(a - y) = \textcolor{naranja}{-1}$", 40,
                  alinear="centro"),
        ),
        Paso(
            "Se multiplican: menos uno, por cero coma veinticinco, por dos, da menos cero coma cinco. Si el peso sube un poquito, "
            "el error baja la mitad de eso. Recorrer la cadena de atrás hacia adelante, multiplicando derivadas, "
            "es backpropagation: el tema de la semana siete.",
            Formula(960, 820, r"\frac{\partial L}{\partial w} = \textcolor{naranja}{(-1)} \cdot \textcolor{verde}{0.25}"
                    r" \cdot \textcolor{azul}{2} = \mathbf{-0.5}", 50, alinear="centro"),
        ),
    ),
    Escena(
        "Tu turno 3: las dos derivadas de la neurona",
        Paso(
            "Tu turno. Escribe las dos derivadas de la neurona: respecto del peso y respecto del sesgo. Para la del sesgo "
            "solo cambia el último eslabón: ¿cuánto cambia zeta cuando be sube uno? Pausa el video.",
            Codigo(130, 280, "def neuron_grads(w, b, x, y):\n    # TODO: devuelve (dL_dw, dL_db)\n"
                   "    raise NotImplementedError", 34),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- pérdida
    Seccion("Parte 4 · La pérdida como función de los parámetros"),
    Escena(
        "Una recta para los pingüinos",
        Paso(
            "Volvemos a los pingüinos de la semana tres: ¿se puede predecir la masa de un pingüino con el largo "
            "de su aleta? Cada punto es un pingüino.",
            nubes_de_pinguinos(230, 280, 860, 560, id="d"),
        ),
        Paso(
            "El modelo más simple es una recta: doble u por la aleta, más be. Cada par de valores es una recta distinta. "
            "La gris es la línea base: siempre predice la masa promedio, cuatro coma dos kilos, sin mirar la aleta.",
            nubes_de_pinguinos(230, 280, 860, 560, rectas=[(0, masa.mean(), "#9CA3AF")], id="d"),
            Formula(1240, 290, r"\hat{y} = w\,x + b", 54),
            Texto(1240, 420, "línea base: **MSE 0.64**", 38, GRIS, acento=GRIS),
            ocultar=["d"],
        ),
        Paso(
            "Para medir qué tan mala es una recta, promediamos los errores al cuadrado: el error cuadrático medio. "
            "La línea base tiene cero coma sesenta y cuatro. La roja, mal elegida, cero coma cuarenta y siete. "
            "Y la verde, cero coma quince.",
            nubes_de_pinguinos(230, 280, 860, 560, rectas=[(0, masa.mean(), "#9CA3AF"), (0.3, 3.8, ROJO),
                                                          (0.7, 4.2, VERDE)]),
            Texto(1240, 490, "roja: **MSE 0.47**", 38, ROJO, acento=ROJO),
            Texto(1240, 560, "verde: **MSE 0.15**", 38, VERDE, acento=VERDE),
            Formula(1240, 680, r"L(w, b) = \frac{1}{n}\sum_{i=1}^{n} (w\,x_i + b - y_i)^2", 40),
            ocultar=["d"],
        ),
    ),
    Escena(
        "El paisaje de la pérdida",
        Paso(
            "Ahora cambia el punto de vista. Los datos están fijos: lo que se mueve son doble u y be. Si calculamos el error "
            "para cada par, obtenemos un paisaje. Cada punto del mapa es una recta, y la altura es su error.",
            paisaje(230, 270, 760, 620, id="p"),
            Caja(1200, 280, 600, 190, "datos **fijos**\nparámetros **variables**", AZUL, 40),
        ),
        Paso(
            "El fondo del valle es la mejor recta: doble u igual cero coma setenta, be igual cuatro coma veinte, "
            "con un error de cero coma quince. El punto gris es la línea base. Entrenar es bajar desde cualquier punto "
            "hasta el fondo.",
            paisaje(230, 270, 760, 620, marcas=[(W_MEJOR, B_MEJOR, NARANJA, ""), (0, masa.mean(), GRIS, "")]),
            Texto(1200, 540, "**mínimo:** `w = 0.70`, `b = 4.20`", 38, NARANJA, acento=NARANJA),
            Texto(1200, 600, "MSE 0.155", 38, NARANJA),
            Texto(1200, 690, "**línea base:** `w = 0`, `b = 4.20`", 38, GRIS, acento=GRIS),
            Texto(1200, 750, "MSE 0.64", 38, GRIS),
            ocultar=["p"],
        ),
    ),
    Escena(
        "Tu turno 4: el gradiente de la pérdida",
        Paso(
            "Tu turno. Escribe el gradiente del error cuadrático medio, sin bucles. Con la regla de la cadena: "
            "llama erre al error de cada pingüino. La derivada de erre al cuadrado es dos veces erre, por la derivada de erre. "
            "Pausa el video.",
            Formula(130, 260, r"r_i = w\,x_i + b - y_i, \qquad \frac{\partial r_i}{\partial w} = x_i, \qquad"
                    r" \frac{\partial r_i}{\partial b} = 1", 44),
            Codigo(130, 420, "def mse_grad(w, b, x, y):\n    # TODO: devuelve (dL_dw, dL_db), sin bucles\n"
                   "    raise NotImplementedError", 34),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- descenso
    Seccion("Parte 5 · Descenso de gradiente"),
    Escena(
        "Bajar la montaña con niebla",
        Paso(
            "¿Cómo se baja una montaña con niebla? Sientes hacia dónde baja el suelo, das un paso, y repites. "
            "Eso es el descenso de gradiente: los parámetros se mueven un poquito en contra del gradiente.",
            Formula(130, 270, r"\theta \;\leftarrow\; \theta - \textcolor{morado}{\eta}\, \nabla L(\theta)", 60),
        ),
        Paso(
            "La letra eta es el largo del paso: la tasa de aprendizaje, o learning rate. No la aprende el modelo: "
            "la elegimos nosotros.",
            Caja(1000, 250, 800, 170, r"$\eta$ = **tasa de aprendizaje**", MORADO, 40, subtitulo="el largo del paso; la elegimos nosotros"),
        ),
        Paso(
            "Un ejemplo en una dimensión: la función doble u menos tres, al cuadrado, con el mínimo en tres. Desde cero, con eta igual "
            "cero coma uno, el primer paso va a cero coma seis, el segundo a uno coma cero ocho, y cada paso "
            "recorre el veinte por ciento de lo que falta.",
            Grafica(150, 520, 760, 380, parabola, -1, 7, -1, 12, camino=camino_1d(0.1, 12)),
            Terminal(1060, 520, "paso  1: w = 0.6000   L = 5.760000\n"
                     "paso  2: w = 1.0800   L = 3.686400\n"
                     "paso  3: w = 1.4640   L = 2.359296\n"
                     "paso 10: w = 2.6779   L = 0.103763\n"
                     "paso 30: w = 2.9963   L = 0.000014", 28, ancho=740),
        ),
    ),
    Escena(
        "La tasa de aprendizaje",
        Paso(
            "La elección de eta lo es todo. Con eta muy chica, cero coma cero dos, avanza tan lento que en quince pasos "
            "no llega.",
            *panel_tasa(0, 0.02, "muy chica: no llega", AZUL),
        ),
        Paso("Con cero coma tres, llega en pocos pasos.", *panel_tasa(1, 0.3, "buena", VERDE)),
        Paso(
            "Con cero coma nueve, se pasa al otro lado del valle y rebota, aunque termina llegando.",
            *panel_tasa(2, 0.9, "grande: rebota", NARANJA),
        ),
        Paso(
            "Y con uno coma cero cinco, cada rebote es más alto que el anterior: el algoritmo diverge.",
            *panel_tasa(3, 1.05, "demasiado grande: diverge", ROJO),
        ),
        Paso(
            "¿Por qué? En esta función, cada paso multiplica la distancia al mínimo por uno, menos dos veces la tasa de aprendizaje. "
            "Si ese número está entre menos uno y uno, la distancia se achica; si no, crece. "
            "En el trabajo B2 vas a buscar este límite en otras funciones.",
            Formula(960, 820, r"w_{t+1} - 3 = (1 - 2\eta)\,(w_t - 3) \qquad \text{converge si } 0 < \eta < 1", 44,
                    alinear="centro"),
        ),
    ),
    Escena(
        "En dos dimensiones: los pingüinos",
        Paso(
            "Con dos parámetros es igual: los dos se mueven a la vez, cada uno en contra de su derivada parcial. "
            "Empezamos lejos del valle.",
            paisaje(230, 270, 760, 620, caminos=[(CAMINO[:1], NARANJA)], marcas=[(W_MEJOR, B_MEJOR, ROJO, "")], id="p"),
        ),
        Paso(
            "Sesenta pasos con eta igual cero coma uno. El camino baja perpendicular a las curvas de nivel "
            "y llega al fondo del valle.",
            paisaje(230, 270, 760, 620, caminos=[(CAMINO, NARANJA)], marcas=[(W_MEJOR, B_MEJOR, ROJO, "")]),
            ocultar=["p"],
        ),
        Paso(
            "Y llega al mismo punto que la fórmula exacta de mínimos cuadrados, sin usar ninguna fórmula especial: "
            "solo el gradiente. El error final es cero coma quince, contra cero coma sesenta y cuatro de la línea base.",
            Terminal(1120, 300, "descenso de gradiente:\n  w = 0.698, b = 4.202\n  MSE = 0.155\n\n"
                     "np.polyfit:\n  w = 0.698, b = 4.202\n  MSE = 0.155\n\nlínea base:\n  MSE = 0.641", 30, ancho=680),
        ),
    ),
    # --------------------------------------------------------------- probabilidad
    Seccion("Parte 6 · Probabilidad, softmax y entropía cruzada"),
    Escena(
        "Probabilidades: números que suman uno",
        Paso(
            "Para clasificar, el modelo da una probabilidad para cada clase: "
            "números entre cero y uno que suman uno. La más simple es contar: el cuarenta y cuatro "
            "por ciento de los pingüinos son Adelie.",
            Barras(230, 330, 900, 450, ["Adelie", "Gentoo", "Chinstrap"], [0.442, 0.360, 0.199], [AZUL, VERDE, NARANJA],
                   vmax=0.5, formato="{:.3f}", tam=34),
        ),
        Paso(
            "Esa es también la línea base: decir siempre la especie más común acierta el cuarenta y cuatro por ciento.",
            Caja(1260, 380, 540, 220, "**Línea base**", GRIS, 40, subtitulo="siempre Adelie:\naccuracy 0.44"),
        ),
    ),
    Escena(
        "Softmax: de puntajes a porcentajes",
        Paso(
            "Una red termina con un puntaje por clase, que puede ser cualquier número. La softmax los convierte "
            "en probabilidades, en dos pasos.",
            Matriz(130, 300, 1, 3, 150, AZUL, valores=[[2.0, 1.0, 0.1]], tam=38, etiqueta="puntajes `z`"),
        ),
        Paso(
            "Primero, la exponencial vuelve positivo cada puntaje: siete coma tres nueve, dos coma siete dos, "
            "y uno coma uno uno.",
            Flecha(590, 375, 720, 375, GRIS, 6, etiqueta="$e^z$", tam=36),
            Matriz(740, 300, 1, 3, 150, MORADO, valores=[[7.39, 2.72, 1.11]], tam=38, etiqueta="suman 11.21"),
        ),
        Paso(
            "Después se divide por la suma, once coma dos uno, para que todo sume uno: cero coma sesenta y seis, "
            "cero coma veinticuatro y cero coma diez. Es como convertir votos en porcentajes, "
            "pero la exponencial exagera las diferencias.",
            Flecha(1200, 375, 1330, 375, GRIS, 6, etiqueta="÷ suma", tam=32),
            Barras(1350, 260, 450, 260, ["0", "1", "2"], [0.659, 0.242, 0.099], VERDE, vmax=0.7, tam=30),
            Formula(130, 620, r"\text{softmax}(z)_j = \frac{e^{z_j}}{\sum_k e^{z_k}}", 52),
        ),
        Paso(
            "Sumar lo mismo a todos los puntajes no cambia nada. Multiplicarlos por diez, en cambio, "
            "deja al modelo casi seguro de la primera clase.",
            Terminal(880, 620, "softmax(z):        [0.659 0.242 0.099]\n"
                     "softmax(z + 100):  [0.659 0.242 0.099]  ← igual\n"
                     "softmax(10 · z):   [1.    0.    0.   ]  ← más seguro", 26, ancho=920),
        ),
    ),
    Escena(
        "El logaritmo: medir la sorpresa",
        Paso(
            "Para juzgar un clasificador miramos la probabilidad que le dio a la respuesta correcta de cada ejemplo. "
            "Multiplicar dos mil probabilidades de cero coma cinco da un número tan chico que la computadora "
            "lo redondea a cero. El logaritmo convierte ese producto en una suma.",
            Terminal(130, 270, "producto de 2000 probabilidades de 0.5: 0.0  ← se redondeó a 0\n"
                     "suma de sus logaritmos:                 -1386.29", 30, ancho=1660),
            Formula(130, 430, r"\ln(a \cdot b) = \ln a + \ln b", 48),
        ),
        Paso(
            "Además, menos el logaritmo de la probabilidad mide la sorpresa. Si el modelo le daba probabilidad uno a la respuesta "
            "correcta, la sorpresa es cero. Si le daba solo un uno por ciento, la sorpresa es cuatro coma seis. "
            "Equivocarse con mucha confianza cuesta caro.",
            Tabla(130, 560, [["p (clase correcta)", "sorpresa"], ["1.0", "0.000"], ["0.5", "0.693"],
                             ["0.1", "2.303"], ["0.01", "**4.605**"]], [480, 420], tam=30),
            Caja(1150, 600, 650, 220, "**sorpresa** $= -\\ln p$", NARANJA, 40, subtitulo="seguro y equivocado:\ncuesta caro"),
        ),
    ),
    Escena(
        "La entropía cruzada",
        Paso(
            "La entropía cruzada es la sorpresa promedio sobre todos los ejemplos. Solo importa la probabilidad "
            "que el modelo le dio a la clase correcta.",
            Formula(130, 270, r"L = -\frac{1}{n} \sum_{i=1}^{n} \ln P_{i,\,y_i}", 56),
        ),
        Paso(
            "Por ejemplo, si un pingüino es Adelie y el modelo le dio cero coma siete a Adelie, su sorpresa es "
            "cero coma treinta y seis. Las otras dos probabilidades no entran en la cuenta.",
            Matriz(130, 500, 1, 3, 150, AZUL, valores=[[0.7, 0.2, 0.1]], tam=38, etiqueta="`P[i]`, la clase correcta es 0"),
            Marco(125, 495, 160, 160, NARANJA),
            Texto(700, 530, r"$-\ln 0.7 = 0.36$", 48),
        ),
        Paso(
            "Y la línea base de la pérdida: un modelo que reparte por igual entre tres clases tiene pérdida "
            "logaritmo de tres, uno coma uno. Si tu clasificador no baja de ahí, no aprendió nada.",
            Caja(130, 760, 1660, 150, r"**Línea base:** probabilidades iguales $\Rightarrow L = \ln 3 \approx 1.099$", GRIS, 40),
        ),
    ),
    Escena(
        "Tu turno 5 y el gradiente de la pérdida",
        Paso(
            "Tu turno. Escribe la entropía cruzada sin bucles. La pista es la indexación con arreglos, "
            "para elegir en cada fila la probabilidad de la clase correcta. Pausa el video.",
            Codigo(130, 260, "def cross_entropy(P, y):\n    # TODO: sin bucles; pista: P[np.arange(n), y]\n"
                   "    raise NotImplementedError", 34),
            pausa=1.2,
        ),
        Paso(
            "Y un regalo de la regla de la cadena: la derivada de la entropía cruzada respecto de los puntajes es "
            "pe menos ye, dividido por ene, donde ye tiene un uno en la clase correcta. A la clase correcta le sube "
            "el puntaje en lo que le faltó para llegar a uno; a las demás se lo baja.",
            Matriz(130, 600, 1, 3, 120, AZUL, valores=[[0.7, 0.2, 0.1]], tam=34, etiqueta="`P`"),
            Texto(550, 630, "−", 60, alinear="centro"),
            Matriz(610, 600, 1, 3, 120, GRIS, valores=[[1, 0, 0]], tam=34, etiqueta="`Y` (one-hot)"),
            Texto(1030, 630, "=", 60, alinear="centro"),
            Matriz(1090, 600, 1, 3, 120, VERDE, valores=[[-0.3, 0.2, 0.1]], tam=34, etiqueta="$\\partial L / \\partial z$ (× n)"),
        ),
    ),
    # --------------------------------------------------------------- todo junto
    Seccion("Parte 7 · Todo junto"),
    Escena(
        "Un clasificador de pingüinos",
        Paso(
            "Todo junto. Con el largo del pico y el de la aleta clasificamos las tres especies: "
            "una capa lineal, una softmax, y la entropía cruzada como pérdida.",
            Dispersion(230, 270, 640, 470, [(x_pico[especie == e], x_aleta[especie == e]) for e in "ACG"],
                       colores=(AZUL, NARANJA, VERDE), nombres=["Adelie", "Chinstrap", "Gentoo"], rx=(-2.4, 3.0),
                       ry=(-2.4, 4.4), marcas_x=(-2, -1, 0, 1, 2, 3), marcas_y=(-2, -1, 0, 1, 2),
                       etiqueta_x="pico (estandarizado)", etiqueta_y="aleta (estandarizada)", radio=6),
            Caja(1030, 270, 220, 100, "`X`", AZUL, 34),
            Flecha(1260, 320, 1330, 320),
            Caja(1340, 270, 460, 100, "`X @ W + b`", MORADO, 34),
            Flecha(1570, 380, 1570, 410),
            Caja(1340, 420, 460, 100, "softmax → `P`", VERDE, 34),
            Flecha(1570, 530, 1570, 560),
            Caja(1340, 570, 460, 100, "entropía cruzada", NARANJA, 32),
        ),
        Paso(
            "Trescientos pasos de descenso de gradiente, con pe menos ye. La pérdida empieza justo en logaritmo de tres, "
            "porque al principio todos los puntajes son iguales, y después baja.",
            Codigo(1030, 700, "G = (P - Y_tr) / len(y_tr)\nW -= lr * X_tr.T @ G\nb -= lr * G.sum(axis=0)", 28, ancho=770),
        ),
        Paso(
            "En los pingüinos de prueba acierta el noventa y cinco por ciento, contra el cuarenta y dos de la línea base. "
            "Y desde la semana ocho, PyTorch calcula los gradientes "
            "por nosotros.",
            Terminal(130, 900, "línea base (siempre Adelie): accuracy 0.424   ·   softmax con gradiente: accuracy 0.946",
                     26, ancho=1670),
            pausa=0.8,
        ),
    ),
    # --------------------------------------------------------------- cierre
    Seccion("Cierre"),
    Escena(
        "Lo que vimos hoy",
        *[
            Paso(decir, vineta)
            for decir, vineta in zip(
                [
                    "Repasemos. La derivada es una pendiente, y la diferencia centrada sirve para verificar tus cuentas.",
                    "Una derivada parcial mueve una variable a la vez. El gradiente las junta y apunta cuesta arriba.",
                    "En la regla de la cadena, como en los engranajes, las derivadas de los eslabones se multiplican.",
                    "La pérdida es una función de los parámetros, con los datos fijos. Entrenar es buscar su mínimo.",
                    "El descenso de gradiente da pasos en contra del gradiente. Con una tasa muy chica no llega; "
                    "con una demasiado grande, diverge.",
                    "Y la softmax convierte puntajes en probabilidades; la entropía cruzada es la sorpresa promedio, "
                    "con línea base logaritmo de k.",
                ],
                vinetas(130, 250, [
                    "**Derivada:** una pendiente. La diferencia centrada la verifica.",
                    "**Gradiente:** las derivadas parciales juntas; apunta cuesta arriba.",
                    "**Regla de la cadena:** las derivadas de los eslabones se multiplican.",
                    "**Pérdida:** función de los parámetros; los datos quedan fijos.",
                    r"**Descenso de gradiente:** $\theta \leftarrow \theta - \eta \nabla L$; cuidado con $\eta$.",
                    r"**Softmax + entropía cruzada:** $-\ln$ de la probabilidad correcta; base $\ln k$.",
                ], tam=38, sep=118, ancho=1650),
            )
        ],
    ),
    Escena(
        "Para la semana 5",
        Paso(
            "Para la próxima clase, entrega el trabajo B2: descenso de gradiente a mano, con gráficos de contornos, "
            "y rompiéndolo a propósito con una mala tasa de aprendizaje. Se guarda en la carpeta B2 de tu repositorio.",
            Vineta(130, 260, "**Entrega al inicio de la clase:** B2, Descenso de gradiente a mano, en "
                   "`B2/B2_descenso_gradiente.ipynb`.", 40, NARANJA, numero=1, ancho=1600),
        ),
        Paso(
            "Haz la lectura previa de MÓLÓ, en inglés: aprendizaje supervisado, entrenamiento y evaluación, "
            "y generalización y sobreajuste.",
            Vineta(130, 440, "**Lectura previa** (MÓLÓ): Supervised Learning; Inference, Training, Test and Evaluation; "
                   "Generalisation, Overfitting and Bias.", 40, NARANJA, numero=2, ancho=1600),
        ),
        Paso(
            "Y no olvides la cuota semanal en la plataforma. La semana cinco es machine learning clásico en síntesis, "
            "y al final se publica el Sprint cero.",
            Vineta(130, 620, "**Cuota semanal** en ioai.artix.tech.", 40, NARANJA, numero=3, ancho=1600),
            Texto(130, 760, "Los enlaces están al final de `sem04_calculo.ipynb`.", 36, GRIS),
        ),
        Paso(
            "Eso es todo por hoy. ¡Nos vemos en la semana cinco!",
            Texto(960, 880, "**¡Nos vemos en la semana 5!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
