"""Video de la semana 5: Nivel_Intermedio/clases/sem05_ml_clasico.ipynb."""

from motor import (AZUL, GRIS, MORADO, NARANJA, ROJO, TINTA, VERDE, Caja, Circulo, Codigo, Dispersion,
                   Escena, Flecha, Formula, Linea, Marco, Paso, Regiones, Seccion, Tabla, Terminal, Texto, Vineta,
                   tinte, vinetas)
import numpy as np

META = {
    "pie": "Semana 5 · ML clásico en síntesis · Nivel Intermedio",
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

# La partición del notebook: train_test_split(test_size=0.25, random_state=0, stratify=especie).
TEST_IDX = [
    9, 14, 19, 21, 25, 29, 31, 32, 36, 39, 47, 49, 57, 58, 65, 67, 70, 72, 77, 79, 81, 82, 87, 88, 99, 103, 115,
    117, 118, 122, 130, 131, 136, 138, 140, 141, 143, 146, 154, 155, 162, 164, 165, 167, 170, 173, 175, 177, 180,
    183, 196, 203, 209, 210, 212, 222, 228, 229, 239, 242, 246, 248, 250, 254, 258, 260, 264, 265, 269, 270, 272,
    280, 286, 290, 292, 302, 304, 308, 309, 324, 328, 329, 335, 337, 338, 341,
]

aleta = np.array(ALETAS, float)
pico = np.array(PICOS_DECIMAS) / 10
masa_g = np.array(MASAS_25G) * 25
clase = np.searchsorted(np.array(list("ACG")), np.array(list(ESPECIES)))  # 0 Adelie, 1 Chinstrap, 2 Gentoo
COLORES_ESP = (AZUL, NARANJA, VERDE)
NOMBRES_ESP = ["Adelie", "Chinstrap", "Gentoo"]
prueba = np.zeros(len(aleta), bool)
prueba[TEST_IDX] = True
X_TR, C_TR = np.c_[pico, aleta][~prueba], clase[~prueba]
MEDIA, DESV = X_TR.mean(axis=0), X_TR.std(axis=0)  # el StandardScaler del notebook

# Regresión logística del notebook (coeficientes sobre las columnas estandarizadas).
COEF = np.array([[-2.36408, -1.07625], [2.40407, -1.87157], [-0.03999, 2.94782]])
INTER = np.array([0.43796, -0.27094, -0.16702])

# Árboles del notebook (DecisionTreeClassifier, random_state=0): característica (0 pico, 1 aleta),
# umbral, hijo izquierdo (≤ umbral), hijo derecho y la clase de cada nodo; -1 marca una hoja.
ARBOL_2 = dict(
    feat=[1, 0, -2, -2, 0, -2, -2], umbral=[206.5, 43.35, 0, 0, 41.25, 0, 0],
    izq=[1, 2, -1, -1, 5, -1, -1], der=[4, 3, -1, -1, 6, -1, -1], cls=[0, 0, 0, 1, 2, 0, 2],
)
ARBOL_TODO = dict(
    feat=[1, 0, 0, 0, -2, 0, 1, -2, -2, -2, 1, -2, -2, 0, 1, 0, -2, -2, 0, 0, -2, -2, -2, 1, -2, 0, -2, -2, 0,
          -2, 1, 0, 0, 0, -2, -2, -2, 1, -2, 0, -2, -2, -2],
    umbral=[206.5, 43.35, 42.35, 40.85, 0, 40.95, 189.0, 0, 0, 0, 189.5, 0, 0, 46.05, 194.5, 45.4, 0, 0, 44.65,
            43.8, 0, 0, 0, 202.0, 0, 48.85, 0, 0, 41.25, 0, 212.5, 48.85, 44.5, 44.05, 0, 0, 0, 211.0, 0, 49.05, 0,
            0, 0],
    izq=[1, 2, 3, 4, -1, 6, 7, -1, -1, -1, 11, -1, -1, 14, 15, 16, -1, -1, 19, 20, -1, -1, -1, 24, -1, 26, -1,
         -1, 29, -1, 31, 32, 33, 34, -1, -1, -1, 38, -1, 40, -1, -1, -1],
    der=[28, 13, 10, 5, -1, 9, 8, -1, -1, -1, 12, -1, -1, 23, 18, 17, -1, -1, 22, 21, -1, -1, -1, 25, -1, 27, -1,
         -1, 30, -1, 42, 37, 36, 35, -1, -1, -1, 39, -1, 41, -1, -1, -1],
    cls=[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 2, 1, 2, 0, 2, 2, 2, 2, 2,
         0, 2, 1, 1, 1, 1, 2, 2],
)


def logistica(A, B):
    Z = (np.stack([A, B], axis=-1) - MEDIA) / DESV
    return (Z @ COEF.T + INTER).argmax(axis=-1)


def knn(k):
    z_tr = (X_TR - MEDIA) / DESV

    def predecir(A, B):
        P = (np.c_[A.ravel(), B.ravel()] - MEDIA) / DESV
        salida = np.empty(len(P), int)
        for i in range(0, len(P), 2000):
            D = ((P[i:i + 2000, None, :] - z_tr[None]) ** 2).sum(axis=-1)
            vecinos = C_TR[np.argsort(D, axis=1, kind="stable")[:, :k]]
            votos = np.stack([(vecinos == c).sum(axis=1) for c in range(3)], axis=1)
            salida[i:i + 2000] = votos.argmax(axis=1)
        return salida.reshape(A.shape)

    return predecir


def arbol(t):
    feat, umbral, izq, der, cls = (np.array(t[k]) for k in ("feat", "umbral", "izq", "der", "cls"))

    def predecir(A, B):
        V = np.stack([A.ravel(), B.ravel()], axis=1)
        nodo = np.zeros(len(V), int)
        for _ in range(len(feat)):
            hoja = izq[nodo] == -1
            if hoja.all():
                break
            a_la_izq = V[np.arange(len(V)), np.maximum(feat[nodo], 0)] <= umbral[nodo]
            nodo = np.where(hoja, nodo, np.where(a_la_izq, izq[nodo], der[nodo]))
        return cls[nodo].reshape(A.shape)

    return predecir


def regiones(x, y, w, h, predecir, **kw):
    return Regiones(x, y, w, h, predecir, (30, 62), (168, 235), COLORES_ESP,
                    series=[(X_TR[C_TR == c, 0], X_TR[C_TR == c, 1]) for c in range(3)], radio=4,
                    marcas_x=(35, 45, 55), marcas_y=(180, 200, 220), etiqueta_x="pico (mm)", **kw)


# Los datos inventados de la parte 3, con la misma semilla y en el mismo orden que el notebook.
rng = np.random.default_rng(0)
x_poli = np.sort(rng.uniform(0, 1, 20))
t_poli = np.sin(2 * np.pi * x_poli) + rng.normal(0, 0.25, 20)
x_val = rng.uniform(0, 1, 200)
t_val = np.sin(2 * np.pi * x_val) + rng.normal(0, 0.25, 200)
GRADOS = np.arange(0, 11)
MSE_TR, MSE_VAL = [], []
for g in GRADOS:
    c = np.polyfit(x_poli, t_poli, g)
    MSE_TR.append(np.mean((np.polyval(c, x_poli) - t_poli) ** 2))
    MSE_VAL.append(np.mean((np.polyval(c, x_val) - t_val) ** 2))
MEDICIONES = {g: [np.polyfit(x_poli, np.sin(2 * np.pi * x_poli) + rng.normal(0, 0.25, 20), g) for _ in range(30)]
              for g in (1, 3, 9)}


def verdad(x):
    return np.sin(2 * np.pi * x)


def panel_polinomio(x, grado, color, id=None):
    coefs = np.polyfit(x_poli, t_poli, grado)
    return Dispersion(x, 300, 420, 400, [(x_poli, t_poli)], colores=(AZUL,), rx=(0, 1), ry=(-1.8, 1.8),
                      marcas_x=(0, 0.5, 1), marcas_y=(-1, 0, 1), radio=7,
                      curvas=[(verdad, "#B8BCC4", 4), (lambda v: np.polyval(coefs, v), color, 6)], id=id)


def panel_mediciones(x, grado):
    curvas = [(lambda v, c=c: np.polyval(c, v), tinte(NARANJA, 0.55), 2) for c in MEDICIONES[grado]]
    return Dispersion(x, 300, 420, 360, [], rx=(x_poli.min(), x_poli.max()), ry=(-1.8, 1.8), marcas_y=(-1, 0, 1),
                      curvas=curvas + [(verdad, TINTA, 5)])


def interpolar(xs, ys):
    return lambda v: np.interp(v, xs, ys)


# Datos de cáncer de mama (calculados con el notebook).
SPLITS = [0.8772, 0.9561, 0.9123, 0.9386, 0.9298, 0.9561, 0.9298, 0.9123, 0.9474, 0.9298, 0.9123, 0.9035, 0.9123,
          0.9474, 0.9561, 0.9211, 0.8947, 0.9211, 0.9474, 0.9386]
PROFUNDIDADES = np.arange(1, 13)
CV_MEDIA = [0.8989, 0.9209, 0.9319, 0.9319, 0.9341, 0.9297, 0.9275, 0.9275, 0.9275, 0.9275, 0.9275, 0.9275]
CV_DESV = [0.0189, 0.0254, 0.0254, 0.0176, 0.0155, 0.0149, 0.0164, 0.0164, 0.0164, 0.0164, 0.0164, 0.0164]
CV_ENTREN = [0.9231, 0.9516, 0.9758, 0.9912, 0.9956, 0.9978, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def fila_de_pinguinos(y, orden, marco=None):
    """Una fila de 20 pingüinos como círculos de color; orden es una cadena de A, C y G."""
    colores = {"A": AZUL, "C": NARANJA, "G": VERDE}
    elementos = [Circulo(170 + i * 84, y, 28, "", colores[e]) for i, e in enumerate(orden)]
    if marco:
        elementos.append(Marco(170 + 16 * 84 - 40, y - 42, 4 * 84 + 2, 84, ROJO))
    return elementos


def pliegues(vuelta_max):
    """La validación cruzada de 5 pliegues, una fila por vuelta, hasta vuelta_max."""
    elementos = []
    for v in range(vuelta_max):
        y = 290 + v * 95
        elementos.append(Texto(130, y + 14, f"vuelta {v + 1}", 34, GRIS))
        for p in range(5):
            val = p == v
            elementos.append(Caja(360 + p * 190, y, 175, 72, "validación" if val else "entrena",
                                  NARANJA if val else AZUL, 26))
    return elementos


def diana(cx, cy, flechas, color):
    """Un blanco de tiro con flechas (puntos) en las posiciones dadas, relativas al centro."""
    return [
        Circulo(cx, cy, 190, "", GRIS, relleno="#FFFFFF"),
        Circulo(cx, cy, 125, "", GRIS, relleno=tinte(ROJO, 0.8)),
        Circulo(cx, cy, 60, "", ROJO, relleno=tinte(ROJO, 0.5)),
        *[Circulo(cx + dx, cy + dy, 13, "", color) for dx, dy in flechas],
    ]


def nuevo_pinguino():
    """Un pingüino nuevo (pico 46 mm, aleta 196 mm) y sus 5 vecinos más cercanos en las medidas estandarizadas."""
    nuevo = np.array([46.0, 196.0])
    d = (((X_TR - nuevo) / DESV) ** 2).sum(axis=1)
    return nuevo, np.argsort(d, kind="stable")[:5]


NUEVO, VECINOS = nuevo_pinguino()


def nube_vecinos(id=None):
    return Dispersion(230, 270, 820, 560, [(X_TR[C_TR == c, 0], X_TR[C_TR == c, 1]) for c in range(3)],
                      colores=COLORES_ESP, nombres=NOMBRES_ESP, rx=(32, 60), ry=(170, 252),
                      marcas_x=(35, 40, 45, 50, 55), marcas_y=(180, 200, 220), etiqueta_x="largo del pico (mm)",
                      etiqueta_y="largo de la aleta (mm)", radio=6, id=id)


def flechas_vecinos():
    base = nube_vecinos()
    px, py = base._a_pantalla(*NUEVO)
    salida = []
    for i in VECINOS:
        qx, qy = base._a_pantalla(*X_TR[i])
        salida.append(Circulo(qx, qy, 11, "", TINTA, relleno=COLORES_ESP[C_TR[i]]))
    salida.append(Circulo(px, py, 9, "", TINTA))
    return salida


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la quinta clase del nivel intermedio: machine learning clásico en síntesis. "
            "Son las reglas del juego que vas a usar con cualquier modelo, también con las redes neuronales.",
            Texto(960, 250, "SEMANA 5 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**ML clásico en síntesis**", 92, alinear="centro"),
            Texto(960, 460, "entrenar, validar, probar, y medir bien", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Vamos a ver cuatro ideas: separar los datos para que la nota sea honesta, el sobreajuste, "
            "la validación cruzada y las métricas para clasificar.",
            Caja(110, 640, 400, 200, "**1**\nEntrenar, validar, probar", AZUL, 36),
            Caja(560, 640, 400, 200, "**2**\nSobreajuste", VERDE, 38),
            Caja(1010, 640, 400, 200, "**3**\nValidación cruzada", MORADO, 38),
            Caja(1460, 640, 400, 200, "**4**\nMétricas", NARANJA, 38),
        ),
    ),
    # --------------------------------------------------------------- supervisado
    Seccion("Parte 1 · Aprendizaje supervisado"),
    Escena(
        "Aprender con ejemplos resueltos",
        Paso(
            "En el aprendizaje supervisado tenemos ejemplos con su respuesta. Es como aprender a manejar con un "
            "instructor al lado que te corrige: después tienes que manejar solo.",
            Caja(130, 260, 760, 170, "**Con instructor**", AZUL, 40, subtitulo="ejemplos con la respuesta correcta"),
            Caja(1030, 260, 760, 170, "**Solo**", VERDE, 40, subtitulo="datos nuevos, sin la respuesta"),
        ),
        Paso(
            "Las entradas forman la matriz equis: una fila por pingüino, una columna por medida. "
            "Las respuestas forman el vector ye.",
            Tabla(130, 500, [["pico (mm)", "aleta (mm)", "especie"], ["39.1", "181", "Adelie"], ["40.3", "195", "Adelie"],
                             ["51.3", "198", "Chinstrap"], ["52.5", "221", "Gentoo"]], [260, 260, 300], tam=32),
            Marco(122, 492, 536, 342, AZUL),
            Marco(642, 492, 316, 342, NARANJA),
            Texto(390, 860, "`X`: forma `(n, d)`", 34, AZUL, alinear="centro"),
            Texto(800, 860, "`y`: forma `(n,)`", 34, NARANJA, alinear="centro"),
        ),
        Paso(
            "Si la respuesta es un número, como la masa, el problema es de regresión. Si es una categoría, "
            "como la especie, es de clasificación.",
            Caja(1130, 500, 660, 150, "**Regresión**: un número", MORADO, 38, subtitulo="la masa del pingüino"),
            Caja(1130, 690, 660, 150, "**Clasificación**: una categoría", VERDE, 38, subtitulo="la especie del pingüino"),
        ),
    ),
    Escena(
        "fit y predict",
        Paso(
            "En scikit-learn todos los modelos se usan igual, con dos métodos. El método fit es estudiar: busca los "
            "parámetros que minimizan la pérdida. El método predict es responder el examen.",
            Caja(130, 260, 760, 150, "`fit(X, y)`: **estudiar**", AZUL, 40, subtitulo="ajusta los parámetros"),
            Caja(1030, 260, 760, 150, "`predict(X_new)`: **responder**", VERDE, 40, subtitulo="usa los parámetros aprendidos"),
        ),
        Paso(
            "La regresión lineal de la semana cuatro, en tres líneas. Ojo con el doble corchete: equis siempre "
            "es una tabla, aunque tenga una sola columna. Aprende que cada milímetro de aleta suma unos cincuenta "
            "gramos.",
            Codigo(130, 470, 'reg = LinearRegression()\nreg.fit(penguins[["flipper_length_mm"]],\n'
                   '        penguins["body_mass_g"])\nreg.predict(new)', 30, ancho=760),
            Dispersion(1120, 480, 640, 360, [(aleta, masa_g)], rx=(170, 235), ry=(2500, 6500),
                       marcas_x=(180, 200, 220), marcas_y=(3000, 4000, 5000, 6000), etiqueta_x="aleta (mm)",
                       etiqueta_y="masa (g)", radio=4, rectas=[(49.69, -5780.8, NARANJA)]),
            Terminal(130, 790, "w = 49.7 gramos por milímetro de aleta\n"
                     "aleta de 200 mm → masa predicha 4156 g", 26, ancho=760),
        ),
    ),
    Escena(
        "La regresión logística es la softmax de la semana 4",
        Paso(
            "Para clasificar, la regresión logística es exactamente el clasificador que entrenamos a mano al final de "
            "la semana cuatro: una capa lineal, una softmax y la entropía cruzada.",
            Caja(130, 280, 330, 110, "`X`", AZUL, 36),
            Flecha(470, 335, 560, 335),
            Caja(570, 280, 420, 110, "`X @ W + b`", MORADO, 36),
            Flecha(1000, 335, 1090, 335),
            Caja(1100, 280, 320, 110, "softmax", VERDE, 36),
            Flecha(1430, 335, 1520, 335),
            Caja(1530, 280, 260, 110, "`P`", NARANJA, 36),
        ),
        Paso(
            "Antes la ponemos en un pipeline con un estandarizador, que resta la media y divide por la desviación "
            "estándar de cada columna. Todo el pipeline se usa como un solo modelo.",
            Codigo(130, 480, "clf = make_pipeline(StandardScaler(), LogisticRegression())\nclf.fit(X_peng, y_peng)\n"
                   "clf.predict_proba(X_peng[:1])", 32),
            Terminal(130, 760, "probabilidades del primero: [[0.978 0.022 0.   ]] ← ['Adelie' 'Chinstrap' 'Gentoo']", 28,
                     ancho=1660),
        ),
    ),
    Escena(
        "kNN: dime con quién andas",
        Paso(
            "Otro modelo, kNN, los k vecinos más cercanos, no aprende parámetros: guarda todos los ejemplos. "
            "Llega un pingüino nuevo, el punto negro, con pico de cuarenta y seis milímetros y aleta de ciento noventa y seis.",
            nube_vecinos(id="n"),
            Circulo(*nube_vecinos()._a_pantalla(*NUEVO), 9, "", TINTA),
            Caja(1230, 290, 560, 160, "dime con quién andas…", MORADO, 38, subtitulo="…y te diré quién eres"),
        ),
        Paso(
            "Buscamos los cinco pingüinos más parecidos y votan. Cuatro son Chinstrap y uno es Adelie: "
            "el modelo dice Chinstrap.",
            *flechas_vecinos(),
            Texto(1230, 530, "5 vecinos: **4 Chinstrap**, 1 Adelie", 38, NARANJA, acento=NARANJA),
            Texto(1230, 610, "→ predice **Chinstrap**", 40, NARANJA, acento=NARANJA),
        ),
    ),
    Escena(
        "Tu turno 1: tu primer kNN",
        Paso(
            "Tu turno. Crea un kNN con cinco vecinos, dentro de un pipeline con el estandarizador. Entrénalo con los "
            "pingüinos de entrenamiento y mide su porcentaje de aciertos con los de prueba. Pausa el video.",
            Codigo(130, 280, "# TODO: crea knn, entrénalo y calcula acc_knn\nknn = None\nacc_knn = None", 34),
            Caja(130, 560, 1660, 140, "`KNeighborsClassifier(n_neighbors=5)` · `fit` · `score`", VERDE, 38),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- splits
    Seccion("Parte 2 · Entrenamiento, validación y prueba"),
    Escena(
        "Un examen con las preguntas de la clase",
        Paso(
            "Imagina un examen con las mismas preguntas que se resolvieron en clase: quien memorizó las respuestas "
            "saca diez sin entender nada. Con un modelo pasa lo mismo: si lo evalúas con sus datos de entrenamiento, "
            "no sabes si aprendió o si memorizó.",
            Caja(130, 280, 1660, 170, "Examen = las preguntas de la clase  ⇒  **memorizar basta**", ROJO, 42),
            Caja(130, 520, 1660, 170, "Evaluar con los datos de entrenamiento  ⇒  **no dice nada**", ROJO, 42),
        ),
    ),
    Escena(
        "El árbol de decisión: veinte preguntas",
        Paso(
            "Lo vemos con un árbol de decisión, que clasifica con preguntas de sí o no, como el juego de las veinte "
            "preguntas. Con los pingüinos, la primera pregunta es: ¿la aleta mide doscientos seis milímetros o menos?",
            Caja(330, 260, 520, 110, "¿aleta ≤ 206.5 mm?", MORADO, 34),
        ),
        Paso(
            "Cada respuesta lleva a otra pregunta, ahora sobre el pico, y al final a una hoja con una especie.",
            Flecha(470, 380, 280, 470, GRIS, 5, etiqueta="sí", tam=30, lado="izquierda"),
            Flecha(710, 380, 900, 470, GRIS, 5, etiqueta="no", tam=30, lado="derecha"),
            Caja(110, 480, 400, 100, "¿pico ≤ 43.35?", MORADO, 32),
            Caja(680, 480, 400, 100, "¿pico ≤ 41.25?", MORADO, 32),
            Caja(110, 660, 190, 90, "Adelie", AZUL, 28),
            Caja(320, 660, 190, 90, "Chinstrap", NARANJA, 26),
            Caja(680, 660, 190, 90, "Adelie", AZUL, 28),
            Caja(890, 660, 190, 90, "Gentoo", VERDE, 28),
            Linea(220, 580, 205, 660, GRIS, 4),
            Linea(400, 580, 415, 660, GRIS, 4),
            Linea(790, 580, 775, 660, GRIS, 4),
            Linea(970, 580, 985, 660, GRIS, 4),
        ),
        Paso(
            "En el plano, cada pregunta es un corte recto, paralelo a un eje: el árbol divide el plano en escalones.",
            regiones(1320, 280, 470, 470, arbol(ARBOL_2)),
        ),
    ),
    Escena(
        "Memorizar no es aprender",
        Paso(
            "Sin límite de preguntas, el árbol puede cortar hasta dejar un solo ejemplo en cada hoja. Con los datos "
            "de cáncer de mama del B0, acierta el cien por ciento de los tumores con los que entrenó.",
            Terminal(130, 280, "árbol sin límite: accuracy de entrenamiento 1.000", 32, ancho=1660),
        ),
        Paso(
            "Pero con tumores que nunca vio, acierta el noventa y cuatro. La cifra honesta es la segunda. "
            "Lo que importa es la generalización: qué tan bien funciona con datos nuevos.",
            Terminal(130, 280, "árbol sin límite: accuracy de entrenamiento 1.000, de prueba 0.944", 32, ancho=1660),
            Caja(130, 520, 1660, 160, "**Generalización:** qué tan bien funciona el modelo con datos **nuevos**", VERDE, 42),
        ),
    ),
    Escena(
        "Tres partes, tres papeles",
        Paso(
            "Por eso los datos se separan en tres partes. Con el entrenamiento, el modelo ajusta sus parámetros: "
            "son los ejercicios de la clase.",
            Caja(130, 270, 520, 360, "**Entrenamiento**", AZUL, 42, subtitulo="`fit`: ajustar parámetros\n\nlos ejercicios de la clase"),
        ),
        Paso(
            "Con la validación comparamos modelos y elegimos hiperparámetros: son los simulacros. Un hiperparámetro "
            "es una perilla que eliges tú antes de entrenar, como la cantidad de vecinos o la profundidad del árbol.",
            Caja(700, 270, 520, 360, "**Validación**", NARANJA, 42, subtitulo="elegir modelos e\nhiperparámetros\n\nlos simulacros"),
            Texto(130, 700, "**Hiperparámetro:** una perilla que eliges tú: $k$ del kNN, `max_depth`, la tasa de aprendizaje.", 36, ancho=1660),
        ),
        Paso(
            "Y la prueba es el examen final: se guarda en una caja fuerte y se usa una sola vez. Si eliges las perillas "
            "mirando la prueba, deja de ser un examen sorpresa y la nota sale inflada.",
            Caja(1270, 270, 520, 360, "**Prueba**", ROJO, 42, subtitulo="la nota final,\n**una sola vez**\n\nel examen, en la caja fuerte"),
        ),
    ),
    Escena(
        "Mezclar y estratificar",
        Paso(
            "Un detalle importante. Los pingüinos del archivo vienen ordenados por especie: primero los Adelie, "
            "después los Chinstrap y al final los Gentoo.",
            *fila_de_pinguinos(330, "AAAAAAAAACCCCGGGGGGG"),
            Texto(130, 400, "en el orden del archivo", 32, GRIS),
        ),
        Paso(
            "Si cortamos el último veinte por ciento sin mezclar, la prueba tiene solo Gentoo, y el modelo nunca "
            "sería evaluado con las otras especies.",
            Marco(170 + 16 * 84 - 40, 288, 4 * 84 + 2, 84, ROJO),
            Texto(1500, 400, "prueba: solo Gentoo", 32, ROJO),
        ),
        Paso(
            "Por eso primero se mezcla, y además se estratifica: que cada parte tenga la misma proporción de especies "
            "que el total. En scikit-learn, eso hace el argumento stratify.",
            *fila_de_pinguinos(600, "GACAGAGCAAGAGACAGCAG", marco=True),
            Texto(130, 670, "mezclado y estratificado", 32, GRIS),
            Codigo(130, 760, "train_test_split(X, y, test_size=0.2, stratify=y)", 32),
        ),
    ),
    Escena(
        "Tu turno 2: separar a mano",
        Paso(
            "Tu turno. Escribe una función que mezcle los índices con una semilla y los corte en entrenamiento, "
            "validación y prueba. La verificación revisa que no compartan ningún índice y que estén todos. "
            "Pausa el video.",
            Codigo(130, 280, "def split_indices(n, val_frac, test_frac, seed):\n"
                   "    # TODO: mezcla con np.random.default_rng(seed).permutation(n)\n"
                   "    #       y devuelve (train_idx, val_idx, test_idx)\n    raise NotImplementedError", 32),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- sobreajuste
    Seccion("Parte 3 · Sobreajuste, sesgo y varianza"),
    Escena(
        "Tres curvas para los mismos puntos",
        Paso(
            "Para ver el sobreajuste usamos datos inventados: veinte puntos de una onda, la curva gris, con ruido. "
            "Les ajustamos polinomios, y el grado es la perilla: cuanto más alto, más flexible la curva. "
            "Con grado uno, una recta no puede seguir la onda: es el subajuste.",
            panel_polinomio(220, 1, ROJO),
            Texto(430, 790, "**grado 1**: muy rígido", 36, ROJO, acento=ROJO, alinear="centro"),
        ),
        Paso(
            "Con grado tres, la curva sigue la forma de la verdad y deja pasar el ruido.",
            panel_polinomio(820, 3, VERDE),
            Texto(1030, 790, "**grado 3**: justo", 36, VERDE, acento=VERDE, alinear="centro"),
        ),
        Paso(
            "Con grado nueve, pasa casi por cada punto, ruido incluido, y se dispara al final. Es el sobreajuste: "
            "el estudiante que se memorizó las respuestas.",
            panel_polinomio(1420, 9, NARANJA),
            Texto(1630, 790, "**grado 9**: demasiado flexible", 36, NARANJA, acento=NARANJA, alinear="centro"),
        ),
    ),
    Escena(
        "Tu turno 3: la curva de error",
        Paso(
            "Tu turno. Para cada grado, ajusta el polinomio con los datos de entrenamiento y calcula el error "
            "cuadrático medio en entrenamiento y en validación. Pausa el video.",
            Codigo(130, 280, "def mse_by_degree(x_tr, t_tr, x_va, t_va, degrees):\n"
                   "    # TODO: np.polyfit y np.polyval, un MSE por grado\n"
                   "    #       devuelve (train_mse, val_mse)\n    raise NotImplementedError", 32),
            pausa=1.2,
        ),
    ),
    Escena(
        "El gráfico más importante de la clase",
        Paso(
            "Con esa función se dibuja este gráfico. En azul, el error de entrenamiento: siempre baja al dar más "
            "flexibilidad, porque la curva se acomoda cada vez mejor a los puntos que ve.",
            Dispersion(300, 280, 1000, 520, [(GRADOS, MSE_TR)], colores=(AZUL,), nombres=["entrenamiento"],
                       rx=(-0.3, 10.3), ry=(0, 0.8), marcas_x=tuple(range(11)), marcas_y=(0, 0.2, 0.4, 0.6),
                       etiqueta_x="grado del polinomio (flexibilidad)", etiqueta_y="MSE", radio=9,
                       curvas=[(interpolar(GRADOS, MSE_TR), AZUL, 4)], id="e"),
        ),
        Paso(
            "En naranja, la validación: baja, toca fondo en grado tres, y vuelve a subir. A la izquierda, subajuste; "
            "a la derecha, sobreajuste. Se elige el punto más bajo de la validación: lo vas a hacer todo el año.",
            Dispersion(300, 280, 1000, 520, [(GRADOS, MSE_TR), (GRADOS, MSE_VAL)], colores=(AZUL, NARANJA),
                       nombres=["entrenamiento", "validación"], rx=(-0.3, 10.3), ry=(0, 0.8),
                       marcas_x=tuple(range(11)), marcas_y=(0, 0.2, 0.4, 0.6),
                       etiqueta_x="grado del polinomio (flexibilidad)", etiqueta_y="MSE", radio=9,
                       curvas=[(interpolar(GRADOS, MSE_TR), AZUL, 4), (interpolar(GRADOS, MSE_VAL), NARANJA, 4)]),
            Texto(1420, 330, "← **subajuste**", 36, ROJO, acento=ROJO),
            Texto(1420, 410, "**sobreajuste** →", 36, NARANJA, acento=NARANJA),
            Texto(1420, 520, "mejor en validación:", 34, GRIS),
            Texto(1420, 570, "**grado 3**", 44, VERDE, acento=VERDE),
            ocultar=["e"],
        ),
    ),
    Escena(
        "Sesgo y varianza: dos arqueros",
        Paso(
            "¿Por qué falla cada extremo? Piensa en dos arqueros. El primero tiene la mira torcida: sus flechas "
            "quedan juntas, pero lejos del centro. Eso es el sesgo.",
            *diana(480, 520, [(95, -80), (110, -60), (80, -95), (120, -90), (100, -70), (88, -62)], AZUL),
            Texto(480, 780, "**sesgo** alto: la mira torcida", 36, AZUL, acento=AZUL, alinear="centro"),
        ),
        Paso(
            "El segundo apunta bien, pero le tiembla el pulso: en promedio da en el centro, pero cada flecha cae "
            "en otro lado. Eso es la varianza.",
            *diana(1440, 520, [(-120, 40), (90, 110), (30, -140), (-60, -70), (140, -20), (-20, 130)], NARANJA),
            Texto(1440, 780, "**varianza** alta: el pulso tembloroso", 36, NARANJA, acento=NARANJA, alinear="centro"),
        ),
    ),
    Escena(
        "Treinta mediciones de la misma onda",
        Paso(
            "Repetimos el experimento treinta veces: medimos de nuevo los mismos veinte puntos, con ruido nuevo, "
            "y ajustamos el mismo grado. Con grado uno, las rectas salen casi iguales, pero todas lejos de la verdad: "
            "sesgo alto.",
            panel_mediciones(220, 1),
            Texto(430, 740, "grado 1", 36, alinear="centro"),
        ),
        Paso(
            "Con grado tres, las curvas quedan juntas y cerca de la verdad. Y con grado nueve, cada medición da "
            "una curva distinta: varianza alta. Al subir la flexibilidad, el sesgo baja y la varianza sube. "
            "El mejor modelo está en el medio.",
            panel_mediciones(820, 3),
            Texto(1030, 740, "grado 3", 36, alinear="centro"),
            panel_mediciones(1420, 9),
            Texto(1630, 740, "grado 9", 36, alinear="centro"),
            Terminal(640, 790, "grado         1       3       9\nsesgo²    0.234   0.006   0.001\n"
                     "varianza  0.005   0.012   0.039", 26),
        ),
    ),
    Escena(
        "El árbol con los tumores",
        Paso(
            "Con el árbol, la perilla de flexibilidad es la profundidad máxima. En los tumores, el entrenamiento sube "
            "hasta uno. Pero mira la validación: salta de un lado a otro, sin una forma clara.",
            Tabla(130, 270, [["max_depth", "entrenamiento", "validación"], ["1", "0.921", "0.895"], ["2", "0.959", "0.895"],
                             ["3", "0.982", "0.877"], ["4", "0.991", "0.886"], ["5", "0.994", "0.921"],
                             ["8", "1.000", "0.912"], ["None", "1.000", "0.912"]], [260, 320, 300], tam=30),
            Marco(704, 330, 306, 448, NARANJA),
        ),
        Paso(
            "Con solo ciento catorce tumores de validación, un tumor más o menos bien clasificado mueve la nota "
            "casi un punto. ¿Podemos elegir con una sola partición?",
            Caja(1150, 380, 640, 260, "114 tumores de validación:\n1 tumor ≈ **0.9 puntos**", NARANJA, 38),
        ),
    ),
    # --------------------------------------------------------------- validación cruzada
    Seccion("Parte 4 · Validación cruzada"),
    Escena(
        "Un solo partido",
        Paso(
            "Midamos cuánto depende la nota de la suerte: el mismo árbol, con los mismos datos, y veinte particiones "
            "al azar. Cada punto es una nota de validación.",
            Dispersion(300, 360, 1300, 160, [(SPLITS, [0] * 20)], colores=(MORADO,), rx=(0.865, 0.965), ry=(-1, 1),
                       marcas_x=(0.88, 0.9, 0.92, 0.94, 0.96), etiqueta_x="accuracy de validación", radio=14),
        ),
        Paso(
            "Van de ochenta y ocho a noventa y seis por ciento: casi ocho puntos solo por la suerte de la partición. "
            "Juzgar a un futbolista por un solo partido es injusto: mejor mirar toda la temporada.",
            Texto(300, 690, "mínima **0.877** · máxima **0.956**: casi 8 puntos de suerte", 40, MORADO, acento=MORADO),
            Caja(300, 790, 1300, 120, "un solo partido  vs  **toda la temporada**", VERDE, 40),
        ),
    ),
    Escena(
        "Cinco pliegues",
        Paso(
            "La validación cruzada con cinco pliegues divide los datos en cinco partes. En la primera vuelta, "
            "la primera parte valida y las otras cuatro entrenan.",
            *pliegues(1),
        ),
        Paso(
            "En la segunda vuelta valida la segunda parte, y así hasta cinco. Cada ejemplo valida exactamente una vez.",
            *pliegues(5),
        ),
        Paso(
            "Se reporta el promedio de las cinco notas, y su desviación estándar, que dice cuánto varían. "
            "Cuesta cinco entrenamientos en vez de uno: con modelos clásicos es barato.",
            *[Texto(1330, 304 + v * 95, s, 34, MORADO) for v, s in enumerate(["0.967", "0.945", "0.890", "0.923", "0.934"])],
            Caja(1500, 520, 300, 150, "**0.932**", MORADO, 44, subtitulo="± 0.025"),
        ),
    ),
    Escena(
        "Tu turno 4: los pliegues a mano",
        Paso(
            "Tu turno. Mezcla los índices, córtalos en pedazos iguales, y arma un par por pedazo: ese pedazo valida "
            "y entrenan todos los demás. Pausa el video.",
            Codigo(130, 280, "def kfold_indices(n, k, seed):\n"
                   "    # TODO: permutation, np.array_split, np.concatenate\n"
                   "    #       devuelve [(train_idx, val_idx), ...]\n    raise NotImplementedError", 32),
            Codigo(130, 620, "cross_val_score(model, X_dev, y_dev, cv=StratifiedKFold(5, shuffle=True))", 30,
                   titulo="después, lo mismo con scikit-learn"),
            pausa=1.2,
        ),
    ),
    Escena(
        "Elegir la profundidad",
        Paso(
            "Ahora sí, una curva confiable. En azul el entrenamiento, que sube hasta uno. En naranja la validación "
            "cruzada, con dos líneas finas a una desviación estándar.",
            Dispersion(300, 280, 1000, 520, [(PROFUNDIDADES, CV_ENTREN), (PROFUNDIDADES, CV_MEDIA)],
                       colores=(AZUL, NARANJA), nombres=["entrenamiento", "validación cruzada"], rx=(0.5, 12.5),
                       ry=(0.87, 1.06), marcas_x=tuple(range(1, 13)), marcas_y=(0.88, 0.92, 0.96, 1.0),
                       etiqueta_x="max_depth (profundidad máxima)", radio=9,
                       curvas=[(interpolar(PROFUNDIDADES, np.add(CV_MEDIA, CV_DESV)), tinte(NARANJA, 0.5), 3),
                               (interpolar(PROFUNDIDADES, np.subtract(CV_MEDIA, CV_DESV)), tinte(NARANJA, 0.5), 3),
                               (interpolar(PROFUNDIDADES, CV_ENTREN), AZUL, 4),
                               (interpolar(PROFUNDIDADES, CV_MEDIA), NARANJA, 4)]),
            Texto(320, 280, "accuracy", 28, GRIS),
        ),
        Paso(
            "El promedio más alto está en cinco, pero de tres a seis están empatados dentro del ruido. "
            "Cuando dos opciones empatan, conviene la más simple, que tiene menos varianza: profundidad tres.",
            Caja(1420, 330, 400, 260, "empate de 3 a 6\n→ **la más simple: 3**", VERDE, 36),
        ),
    ),
    # --------------------------------------------------------------- métricas
    Seccion("Parte 5 · Métricas"),
    Escena(
        "La prueba que siempre dice «sano»",
        Paso(
            "Piensa en una prueba de detección en un colegio de mil estudiantes, donde diez tienen una enfermedad. "
            "Una prueba que dice siempre «sano» acierta con novecientos noventa.",
            Caja(130, 280, 760, 220, "**1000** estudiantes", AZUL, 44, subtitulo="10 enfermos · 990 sanos"),
            Caja(1030, 280, 760, 220, "prueba: siempre «sano»", GRIS, 40, subtitulo="aciertos: 990 de 1000"),
        ),
        Paso(
            "Noventa y nueve por ciento de aciertos, y no detecta a ningún enfermo. Cuando una clase es rara, "
            "el porcentaje de aciertos, la accuracy, engaña.",
            Texto(960, 620, "accuracy **99%** · enfermos detectados: **0 de 10**", 52, ROJO, acento=ROJO, alinear="centro"),
        ),
    ),
    Escena(
        "La matriz de confusión: una alarma de incendios",
        Paso(
            "Para ver mejor, contamos los cuatro casos posibles. Piensa en una alarma de incendios. Si suena por una "
            "tostada quemada, es un falso positivo. Si no suena con el incendio, es un falso negativo.",
            Caja(130, 270, 760, 170, "suena sin fuego", NARANJA, 40, subtitulo="**falso positivo**: falsa alarma"),
            Caja(1030, 270, 760, 170, "no suena con fuego", ROJO, 40, subtitulo="**falso negativo**: se le escapó"),
        ),
        Paso(
            "En una tabla: las filas son la verdad y las columnas, lo que predice el modelo. Los aciertos están "
            "en la diagonal: verdaderos positivos y verdaderos negativos.",
            Tabla(330, 520, [["", "predice positivo", "predice negativo"],
                             ["**es positivo**", "verdadero positivo (**VP**)", "falso negativo (**FN**)"],
                             ["**es negativo**", "falso positivo (**FP**)", "verdadero negativo (**VN**)"]],
                  [300, 480, 480], tam=32),
        ),
    ),
    Escena(
        "Precisión y recall",
        Paso(
            "La precisión responde: cuando la alarma suena, ¿qué fracción de las veces había fuego?",
            Formula(130, 270, r"\text{precisión} = \frac{VP}{VP + FP}", 50),
            Texto(1000, 290, "cuando suena, ¿había fuego?", 38, AZUL),
        ),
        Paso(
            "El recall responde: de todos los incendios, ¿qué fracción detectó?",
            Formula(130, 440, r"\text{recall} = \frac{VP}{VP + FN}", 50),
            Texto(1000, 460, "de los incendios, ¿cuántos detectó?", 38, VERDE),
        ),
        Paso(
            "Y efe uno junta las dos con la media armónica: solo es alta si las dos son altas.",
            Formula(130, 610, r"F_1 = \frac{2 \cdot \text{precisión} \cdot \text{recall}}{\text{precisión} + \text{recall}}", 46),
            Texto(1000, 640, "alta solo si **las dos** son altas", 38, MORADO, acento=MORADO),
        ),
        Paso(
            "Un ejemplo: una prueba marca quince positivos, de los cuales ocho están enfermos, entre diez enfermos "
            "en total. Precisión, ocho de quince: cero coma cincuenta y tres. Recall, ocho de diez: cero coma ochenta.",
            Texto(130, 820, r"VP = 8, FP = 7, FN = 2:  precisión $\tfrac{8}{15} = 0.53$,  recall $\tfrac{8}{10} = 0.80$,"
                  r"  $F_1 = 0.64$", 40),
        ),
    ),
    Escena(
        "Tu turno 5: las métricas a mano",
        Paso(
            "Tu turno. Calcula precisión, recall y efe uno sin bucles. Cuenta los verdaderos positivos, los falsos "
            "positivos y los falsos negativos con máscaras booleanas, como en la semana dos. Pausa el video.",
            Codigo(130, 280, "def precision_recall_f1(y_true, y_pred):\n"
                   "    # TODO: sin bucles; máscaras como (y_pred == 1) & (y_true == 1)\n    raise NotImplementedError", 32),
            Caja(130, 560, 1660, 140, "La verificación usa el ejemplo de 15 positivos y lo compara con scikit-learn.", VERDE, 36),
            pausa=1.2,
        ),
    ),
    Escena(
        "Los tumores",
        Paso(
            "De vuelta a los tumores, con maligno como clase positiva. La línea base dice siempre «benigno»: "
            "sesenta y tres por ciento de aciertos, y recall cero. No encuentra ni un tumor maligno.",
            Terminal(130, 270, "siempre benigno:\n  matriz de confusión [[VN FP] [FN VP]]: [[72, 0], [42, 0]]\n"
                     "  accuracy 0.632   precisión 0.000   recall 0.000   F1 0.000", 30, ancho=1660),
        ),
        Paso(
            "La regresión logística se equivoca en tres de ciento catorce. Dos de esos errores son tumores malignos "
            "que se le escaparon: falsos negativos.",
            Terminal(130, 520, "regresión logística:\n  matriz de confusión [[VN FP] [FN VP]]: [[71, 1], [2, 40]]\n"
                     "  accuracy 0.974   precisión 0.976   recall 0.952   F1 0.964", 30, ancho=1660),
            Texto(130, 760, "**FN = 2**: dos tumores malignos se le escaparon.", 38, ROJO, acento=ROJO),
        ),
    ),
    Escena(
        "El umbral",
        Paso(
            "El modelo dice «maligno» cuando la probabilidad pasa de cero coma cinco. Ese umbral es otra perilla. "
            "Si lo bajas, el modelo se vuelve desconfiado: atrapa más malignos, con más falsas alarmas.",
            Tabla(130, 270, [["umbral", "FP", "FN", "precisión", "recall"], ["0.1", "11", "0", "0.792", "1.000"],
                             ["0.3", "6", "2", "0.870", "0.952"], ["0.5", "1", "2", "0.976", "0.952"],
                             ["0.7", "0", "4", "1.000", "0.905"], ["0.9", "0", "5", "1.000", "0.881"]],
                  [180, 130, 130, 220, 200], tam=32),
            Marco(122, 335, 876, 72, VERDE),
        ),
        Paso(
            "¿Qué umbral usar? Lo decide el costo de cada error. En una detección de cáncer, dejar pasar un tumor "
            "maligno es mucho peor que pedir una biopsia de más: importa el recall.",
            Caja(1120, 270, 680, 200, "**cáncer**: importa el recall", VERDE, 38, subtitulo="un falso negativo es grave\n→ umbral bajo"),
        ),
        Paso(
            "En un filtro de spam es al revés: mandar a la papelera un correo importante es peor. "
            "Importa la precisión. Y en una competencia, la métrica viene en el enunciado: léela con cuidado.",
            Caja(1120, 510, 680, 200, "**spam**: importa la precisión", AZUL, 38, subtitulo="un falso positivo es grave\n→ umbral alto"),
            Texto(130, 800, "En competencia, **la métrica viene en el enunciado**: leerla bien es parte del problema.", 38, ancho=1660),
        ),
    ),
    # --------------------------------------------------------------- modelos
    Seccion("Parte 6 · Los cuatro modelos"),
    Escena(
        "Cuatro formas de dividir el plano",
        Paso(
            "Cada modelo divide el plano a su manera. Pintamos la especie que predice cada uno en cada punto. "
            "La regresión logística traza fronteras rectas: mucho sesgo si la frontera real es curva, poca varianza.",
            regiones(200, 280, 300, 330, logistica),
            Texto(350, 750, "**logística**", 34, alinear="centro"),
            Texto(350, 800, "prueba 0.965", 30, GRIS, alinear="centro"),
        ),
        Paso(
            "El kNN con un vecino dobla la frontera para alcanzar cada punto suelto: aprende de memoria, "
            "y en entrenamiento acierta el cien por ciento.",
            regiones(640, 280, 300, 330, knn(1)),
            Texto(790, 750, "**kNN, k = 1**", 34, alinear="centro"),
            Texto(790, 800, "entren. 1.000 · prueba 0.942", 30, GRIS, alinear="centro"),
        ),
        Paso(
            "Con quince vecinos la frontera se suaviza. Más vecinos, menos flexible.",
            regiones(1080, 280, 300, 330, knn(15)),
            Texto(1230, 750, "**kNN, k = 15**", 34, alinear="centro"),
            Texto(1230, 800, "prueba 0.965", 30, GRIS, alinear="centro"),
        ),
        Paso(
            "Y el árbol sin límite corta en escalones, con franjas finas para atrapar puntos sueltos. "
            "También acierta todo en entrenamiento, y es el peor en la prueba.",
            regiones(1520, 280, 300, 330, arbol(ARBOL_TODO)),
            Texto(1670, 750, "**árbol sin límite**", 34, alinear="centro"),
            Texto(1670, 800, "entren. 1.000 · prueba 0.919", 30, GRIS, alinear="centro"),
        ),
    ),
    Escena(
        "Escalar importa",
        Paso(
            "El kNN mide distancias, y en los tumores el área llega a dos mil quinientos, mientras la suavidad no pasa "
            "de cero coma dos. Sin escalar, la distancia la decide casi solo el área, como comparar casas sumando "
            "el precio en dólares con el número de baños. "
            "Con el estandarizador, cada medida pesa lo mismo, y el kNN sube de noventa y tres a noventa y siete por ciento.",
            Tabla(130, 270, [["medida", "mínimo", "máximo"], ["área", "143.5", "2501"], ["suavidad", "0.053", "0.163"]],
                  [300, 240, 240], tam=34),
            Terminal(980, 270, "kNN sin escalar          accuracy 0.934 ± 0.023\n"
                     "kNN con StandardScaler   accuracy 0.967 ± 0.023", 26),
        ),
        Paso(
            "Y el pipeline evita una trampa: la fuga de datos. Si ajustas el estandarizador con todos los datos antes "
            "de separar, la validación ya vio información de sí misma. Dentro del pipeline, se ajusta solo con "
            "el entrenamiento de cada vuelta.",
            Caja(130, 560, 1660, 200, "**Fuga de datos** (data leakage): la validación ya «vio» información de sí misma",
                 ROJO, 40, subtitulo="`make_pipeline(StandardScaler(), modelo)` ajusta el escalador solo con el entrenamiento"),
        ),
    ),
    # --------------------------------------------------------------- receta
    Seccion("Parte 7 · La receta completa"),
    Escena(
        "La receta del Sprint 0",
        *[
            Paso(decir, *grupo)
            for decir, grupo in zip(
                [
                    "Todo junto, en el orden que vas a usar en el Sprint cero. Primero, separa la prueba y no la toques. "
                    "Segundo, una línea base tonta, que diga siempre la clase más común. Tercero, compara los candidatos "
                    "con validación cruzada, sobre los datos de desarrollo.",
                    "Cuarto, reentrena el elegido con todos los datos de desarrollo. Y quinto, mide en la prueba "
                    "una sola vez, con varias métricas, contra la línea base.",
                ],
                (lambda v: [v[:3], v[3:]])(vinetas(130, 250, [
                    "Separar la prueba y **no tocarla**.",
                    "Una línea base tonta: `DummyClassifier`.",
                    "Comparar candidatos con **validación cruzada**.",
                    "Reentrenar el elegido con todo el desarrollo.",
                    "Medir en la prueba **una sola vez**, contra la línea base.",
                ], tam=38, sep=100, ancho=900, numeradas=True)),
            )
        ],
        Paso(
            "Con los tumores, comparando con efe uno, gana la regresión logística. En la prueba: efe uno de cero coma "
            "noventa y seis, contra cero de la línea base.",
            Terminal(1100, 250, "F1 en validación cruzada:\n  siempre benigno        0.000\n"
                     "  regresión logística    0.976\n  kNN, k = 5             0.954\n  árbol, max_depth = 3   0.908\n\n"
                     "elegido: regresión logística\nen la prueba: F1 0.964", 26, ancho=700),
            pausa=0.8,
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
                    "Repasemos. Se aprende de ejemplos con respuesta, con fit y predict. El entrenamiento ajusta, "
                    "la validación elige, y la prueba da la nota final, una sola vez.",
                    "Sobreajuste es ir bien en entrenamiento y mal con datos nuevos; la curva de validación elige "
                    "la flexibilidad justa. La validación cruzada rota y promedia.",
                    "Con clases raras, la accuracy engaña: mira la precisión, el recall y efe uno. Y cada modelo tiene "
                    "su forma: rectas, vecinos o escalones.",
                ],
                (lambda v: [v[0:2], v[2:4], v[4:6]])(vinetas(130, 250, [
                    "**Supervisado:** ejemplos con respuesta; `fit` y `predict`.",
                    "**Entrenamiento** ajusta, **validación** elige, **prueba** califica una vez.",
                    "**Sobreajuste:** bien en entrenamiento, mal en datos nuevos. Sesgo vs varianza.",
                    "**Validación cruzada:** $k$ pliegues, cada uno valida una vez.",
                    "**Métricas:** con clases raras, precisión, recall y $F_1$.",
                    "**Modelos:** rectas, vecinos (¡escalar!) y escalones.",
                ], tam=38, sep=118, ancho=1650)),
            )
        ],
    ),
    Escena(
        "Para la semana 6",
        Paso(
            "Al final de esta clase se publica el Sprint cero: limpiar un archivo desordenado, ajustar dos modelos "
            "y compararlos con una línea base, con la receta de hoy. Va en la carpeta Sprint cero de tu repositorio, "
            "para el inicio de la semana seis.",
            Vineta(130, 260, "**Sprint 0** (asincrónico): limpiar un CSV, dos modelos, métricas contra una línea base. "
                   "En `Sprint0/`, al inicio de la semana 6.", 40, NARANJA, numero=1, ancho=1600),
        ),
        Paso(
            "Como consulta, tienes tres notebooks de MÓLÓ, en inglés: regresión, árboles de decisión y kNN.",
            Vineta(130, 460, "**Consulta** (MÓLÓ): Regression; Decision Trees; Non-Parametric Learning.", 40, NARANJA,
                   numero=2, ancho=1600),
        ),
        Paso(
            "Y no olvides la cuota semanal en la plataforma. La semana seis empieza el bloque de redes neuronales: "
            "la regresión logística de hoy es una red de una sola capa. ¡Nos vemos en la semana seis!",
            Vineta(130, 600, "**Cuota semanal** en ioai.artix.tech.", 40, NARANJA, numero=3, ancho=1600),
            Texto(130, 740, "Los enlaces están al final de `sem05_ml_clasico.ipynb`.", 36, GRIS),
            Texto(960, 880, "**¡Nos vemos en la semana 6!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
