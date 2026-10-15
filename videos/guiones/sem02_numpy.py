"""Video de la semana 2: Nivel_Intermedio/clases/sem02_numpy.ipynb."""

from motor import (AZUL, GRIS, GRIS_CLARO, MORADO, NARANJA, ROJO, TINTA, VERDE, Caja, Circulo, Codigo, Escena,
                   Flecha, Linea, Marco, Matriz, Paso, Pixeles, Seccion, Tabla, Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 2 · NumPy para tensores y álgebra lineal · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}


# Imágenes del conjunto digits de sklearn (8×8, valores 0–16, un carácter hexadecimal
# por píxel), copiadas aquí porque el entorno de los videos no tiene sklearn.
def _imagen(s):
    return np.array([int(c, 17) for c in s]).reshape(8, 8)


PRIMERA = _imagen("005d910000dfaf5003f20b8004c008800580098004b01c7002e5ac00006da000")
PROMEDIOS = [_imagen(s) for s in [
    "004db30001ddbb1004e52c4005d2096006c1097003d22b6001daad20004ed500",
    "0029a610004de810017fe710029ee600017ce600005ad600005be8210029d931",
    "01aea20005ecd6000585c6000125c4000019a200015c721001becb7101aedc82",
    "018ee71004d9bc200243c9100019e6000016cb2000114c6001768d60019fe910",
    "0007b200003d821001ab564005e67b6009fade5006bcfa200138d4000018c200",
    "01adec4004fc862006e6210005ec940002889820001478200168b71001bf9300",
    "001ba100007f610001da100002e8420004fdca3002fb6a9000ad6bb1001bfd40",
    "005deb5101bcbc5101527b300146cc50019dfb40015cb410003c6000006c2000",
    "005cc61002cb9c3003c78c20019dd710007ed40001b9981001b88a20005dd710",
    "006cb62002daab3004c68e3002acde4000355c5000112a6001656b50006cd920",
]]


def fila_de_promedios(y, celda=17, etiquetas=None):
    """Los diez promedios en una fila, centrados."""
    paso = 8 * celda + 34
    x0 = 960 - (10 * paso - 34) / 2
    etiquetas = etiquetas or [str(c) for c in range(10)]
    return [Pixeles(x0 + c * paso, y, PROMEDIOS[c], celda, etiqueta=etiquetas[c]) for c in range(10)]


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la segunda clase del nivel intermedio: NumPy y álgebra lineal. "
            "Es el idioma en que se escriben todas las redes neuronales.",
            Texto(960, 250, "SEMANA 2 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**NumPy para tensores**", 92, alinear="centro"),
            Texto(960, 460, "formas, ejes, broadcasting y una capa como matriz × vector", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Vamos a ver cuatro ideas. Las formas de los arreglos y cómo elegir partes de ellos. "
            "Los ejes, que dicen en qué dirección se suma o se promedia. El broadcasting, para operar arreglos "
            "de formas distintas. Y por último, por qué una capa de una red es una multiplicación de matrices.",
            Caja(110, 640, 400, 200, "**1**\nFormas e índices", AZUL, 38),
            Caja(560, 640, 400, 200, "**2**\nEjes: `axis`", VERDE, 38),
            Caja(1010, 640, 400, 200, "**3**\nBroadcasting", MORADO, 38),
            Caja(1460, 640, 400, 200, "**4**\nUna capa = `W @ x`", NARANJA, 38),
        ),
    ),
    # ------------------------------------------------------------ por qué NumPy
    Seccion("Parte 1 · Por qué NumPy"),
    Escena(
        "Bucles vs arreglos",
        Paso(
            "Imagina pagar en el supermercado así: la cajera toma un producto, pregunta qué es, busca el precio, "
            "y recién ahí pasa al siguiente. Eso hace un bucle de Python: una operación por vuelta, "
            "y en cada vuelta revisa qué tipo de dato tiene.",
            Caja(130, 260, 780, 200, "**Bucle de Python**", ROJO, 40,
                 subtitulo="un número a la vez, preguntando su tipo en cada vuelta"),
        ),
        Paso(
            "NumPy, en cambio, pasa todo el carrito por el escáner de una vez. Guarda los números juntos en memoria, "
            "todos del mismo tipo, y hace el bucle en un lenguaje rápido. "
            "Escribir una operación sobre el arreglo entero, en vez de un bucle, se llama vectorizar.",
            Caja(1010, 260, 780, 200, "**NumPy**", VERDE, 40, subtitulo="todo el arreglo de una vez: **vectorizar**"),
        ),
        Paso(
            "Hagamos la prueba con la suma de los cuadrados de un millón de números. El bucle es la línea base: "
            "cualquier mejora se mide contra él.",
            Codigo(130, 540, "total = 0.0\nfor v in values_list:\n    total += v * v", 30, ancho=780, titulo="bucle"),
            Codigo(1010, 540, "total_np = np.sum(values**2)", 30, ancho=780, titulo="NumPy"),
        ),
        Paso(
            "Mismo resultado, unas dieciocho veces más rápido en esta computadora. En la tuya el número cambia, "
            "pero la diferencia siempre es grande. Por eso casi nunca escribimos un bucle sobre los ejemplos. "
            "Y de eso se trata el trabajo B1.",
            Terminal(130, 790, "bucle de Python:    27.7 ms  → 1001345.1228\n"
                     "NumPy:               1.5 ms  → 1001345.1228\n"
                     "NumPy es unas 18 veces más rápido", 28, ancho=1660),
            pausa=0.8,
        ),
    ),
    # ----------------------------------------------------------------- formas
    Seccion("Parte 2 · Arreglos y formas"),
    Escena(
        "Tres datos de todo arreglo",
        Paso(
            "Todo arreglo de NumPy tiene tres datos que conviene mirar siempre. Cuántos ejes tiene, "
            "que se llama ndim. Cuántos elementos hay a lo largo de cada eje, que es la forma, o shape. "
            "Y de qué tipo son los números, el dtype.",
            Terminal(130, 250, "scalar  ndim=0  shape=()      dtype=float64\n"
                     "vector  ndim=1  shape=(3,)    dtype=int64\n"
                     "matrix  ndim=2  shape=(2, 3)  dtype=float64", 30, ancho=1660),
        ),
        Paso(
            "En deep learning, a cualquier arreglo se le llama tensor. Un escalar no tiene ejes. "
            "Un vector tiene uno: por ejemplo, un ejemplo con sesenta y cuatro variables. "
            "Una matriz tiene dos: un lote de treinta y dos ejemplos.",
            Tabla(130, 480, [
                ["ndim", "Nombre", "Ejemplo", "shape"],
                ["0", "escalar", "una pérdida", "`()`"],
                ["1", "vector", "un ejemplo con 64 variables", "`(64,)`"],
                ["2", "matriz", "un lote de 32 ejemplos", "`(32, 64)`"],
            ], [160, 330, 760, 410], tam=32, id="tabla"),
        ),
        Paso(
            "Con tres ejes guardamos treinta y dos imágenes en escala de grises. Y con cuatro, "
            "imágenes a color, como las usa PyTorch: lote, canales, alto y ancho.",
            Tabla(130, 480, [
                ["ndim", "Nombre", "Ejemplo", "shape"],
                ["0", "escalar", "una pérdida", "`()`"],
                ["1", "vector", "un ejemplo con 64 variables", "`(64,)`"],
                ["2", "matriz", "un lote de 32 ejemplos", "`(32, 64)`"],
                ["3", "tensor", "32 imágenes en grises de 8×8", "`(32, 8, 8)`"],
                ["4", "tensor", "32 imágenes a color (PyTorch)", "`(32, 3, 224, 224)`"],
            ], [160, 330, 760, 410], tam=32),
            ocultar=["tabla"],
        ),
    ),
    Escena(
        "Una imagen es un arreglo",
        Paso(
            "Toda la clase usa el conjunto digits: mil setecientas noventa y siete imágenes de dígitos escritos a mano, "
            "de ocho por ocho píxeles. Cada píxel es un número de cero, blanco, a dieciséis, negro.",
            Pixeles(130, 260, PRIMERA, 70, numeros=True, etiqueta="la primera imagen: un 0", tam=26),
        ),
        Paso(
            "El conjunto viene en dos formatos. El arreglo de imágenes las guarda una encima de otra: "
            "forma mil setecientos noventa y siete, ocho, ocho.",
            Texto(830, 280, "`images`:  (1797, 8, 8)", 44, acento=AZUL),
            Texto(830, 360, "una imagen encima de otra", 32, GRIS),
        ),
        Paso(
            "Y X tiene las mismas imágenes aplanadas: cada una es una fila de sesenta y cuatro números. "
            "Su forma es mil setecientos noventa y siete por sesenta y cuatro. Además, y guarda el dígito de cada imagen.",
            Texto(830, 480, "`X`:  (1797, 64)", 44, acento=VERDE),
            Texto(830, 560, "cada imagen aplanada en una fila de 64", 32, GRIS),
            Matriz(830, 650, 3, 16, 40, VERDE),
            Texto(1500, 690, "…", 50, GRIS),
            Texto(830, 830, "`y`:  (1797,)  el dígito de cada imagen", 36, acento=NARANJA),
        ),
    ),
    Escena(
        "reshape: los mismos números, otra forma",
        Paso(
            "Piensa en doce huevos. Caben en un cartón de tres por cuatro, o en uno de dos por seis. "
            "Los huevos son los mismos; solo cambia cómo están ordenados. Eso hace reshape.",
            Matriz(130, 280, 1, 12, 90, AZUL, valores=[list(range(12))], tam=26, etiqueta="`np.arange(12)`:  (12,)"),
        ),
        Paso(
            "Con reshape de tres, cuatro, los doce números quedan en tres filas de cuatro. "
            "Y si pones menos uno en un eje, NumPy lo calcula por ti: dos, menos uno, da dos por seis.",
            Matriz(130, 500, 3, 4, 80, VERDE, valores=np.arange(12).reshape(3, 4), tam=28, etiqueta="`reshape(3, 4)`"),
            Matriz(560, 500, 2, 6, 80, MORADO, valores=np.arange(12).reshape(2, 6), tam=28,
                   etiqueta="`reshape(2, -1)` → (2, 6)"),
        ),
        Paso(
            "La única regla: la cantidad total no cambia. Doce huevos no caben en un cartón de cinco por tres, "
            "y NumPy te lo dice con un error.",
            Terminal(1150, 500, ">>> a.reshape(5, 3)\nValueError: cannot reshape\narray of size 12\ninto shape (5,3)",
                     28, ancho=650),
        ),
    ),
    Escena(
        "Tu turno 1: formas",
        Paso(
            "Tu turno. Reconstruye el arreglo de imágenes a partir de X con un reshape, usando menos uno para la cantidad de imágenes. "
            "Convierte un vector de cinco números en una columna. Y aplana la primera imagen. "
            "Pausa el video y resuélvelo en el notebook.",
            Codigo(130, 280, "images_from_X = None  # TODO: X → (1797, 8, 8), con -1\n"
                   "as_column = None      # TODO: np.arange(5) → (5, 1)\n"
                   "first_flat = None     # TODO: images[0] → (64,)", 32),
            Caja(130, 640, 1660, 140, "**Truco:** antes de escribir, di en voz alta la forma que esperas.", VERDE, 38),
            pausa=1.2,
        ),
    ),
    # -------------------------------------------------------------- indexación
    Seccion("Parte 3 · Indexación"),
    Escena(
        "Índices y cortes",
        Paso(
            "Para elegir partes de un arreglo usamos índices, como en las listas, pero uno por eje, separados por comas. "
            "En esta matriz de cuatro por cinco, M de uno, tres es la fila uno, columna tres: el ocho. "
            "Recuerda que se cuenta desde cero.",
            Matriz(200, 300, 4, 5, 100, AZUL, valores=np.arange(20).reshape(4, 5), tam=34),
            *[Texto(160, 335 + i * 100, str(i), 30, GRIS, alinear="centro") for i in range(4)],
            *[Texto(250 + j * 100, 250, str(j), 30, GRIS, alinear="centro") for j in range(5)],
            Marco(500, 400, 100, 100, NARANJA, id="m"),
            Texto(860, 330, "`M[1, 3]`  →  8", 44, acento=NARANJA, id="t"),
        ),
        Paso(
            "Un solo índice elige una fila entera. Y dos puntos significa todo el eje: M de dos puntos, tres, "
            "es la columna tres completa.",
            Marco(200, 400, 500, 100, VERDE, id="m"),
            Marco(500, 300, 100, 400, MORADO, id="m2"),
            Texto(860, 330, "`M[1]`  →  la fila 1", 44, acento=VERDE, id="t"),
            Texto(860, 420, "`M[:, 3]`  →  la columna 3", 44, acento=MORADO),
            ocultar=["m", "t"],
        ),
        Paso(
            "Con imágenes, un corte en los dos últimos ejes es un recorte: el arreglo de imágenes en cero, dos a seis, dos a seis, "
            "es un cuadrado de cuatro por cuatro del centro de la primera imagen.",
            Texto(860, 560, "`images[0, 2:6, 2:6]`  →  un recorte de (4, 4)", 40, acento=AZUL),
            ocultar=["m", "m2"],
        ),
    ),
    Escena(
        "Máscaras: un colador",
        Paso(
            "Una máscara funciona como un colador. La comparación y igual igual tres da un arreglo de verdadero y falso, "
            "uno por imagen: verdadero donde la imagen es un tres.",
            Texto(130, 270, "`mask = y == 3`", 44, acento=AZUL),
            Matriz(130, 380, 1, 12, 92, AZUL, valores=[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1]], tam=30,
                   etiqueta="`y`"),
            Matriz(130, 560, 1, 12, 92, GRIS, valores=[["F", "F", "F", "T", "F", "F", "F", "F", "F", "F", "F", "F"]],
                   tam=30, etiqueta="`mask`"),
            Marco(406, 552, 92, 92, NARANJA),
        ),
        Paso(
            "Si indexas con la máscara, solo pasan los elementos marcados: quedan todas las imágenes del tres. "
            "Y como verdadero cuenta como uno, mask punto sum dice cuántas hay: ciento ochenta y tres.",
            Texto(130, 760, "`images[mask]`  →  todas las imágenes del 3", 40, acento=VERDE),
            Texto(130, 850, "`mask.sum()`  →  183", 40, acento=VERDE),
        ),
        Paso(
            "También puedes indexar con una lista de posiciones: X de cero, once y veintidós devuelve esas tres filas, "
            "en ese orden.",
            Texto(1130, 760, "`X[[0, 11, 22]]`  →  3 filas", 40, acento=MORADO),
        ),
    ),
    Escena(
        "Cuidado: un corte no es una copia",
        Paso(
            "Una trampa común. Un corte no es una fotocopia, es una ventana: mira los mismos números en memoria. "
            "Si cambias algo a través de la ventana, cambia el original.",
            Codigo(130, 270, "a = np.arange(6)\nview = a[:3]\nview[0] = 100", 32, ancho=760, titulo="vista"),
            Terminal(130, 560, "a → [100   1   2   3   4   5]", 32, ancho=760),
            Caja(1000, 270, 790, 150, "**vista** = una ventana", ROJO, 40, subtitulo="cambia el original"),
        ),
        Paso(
            "Si necesitas una copia independiente, pídela con punto copy.",
            Codigo(130, 700, "copy = a[:3].copy()\ncopy[0] = 100", 32, ancho=760, titulo="copia"),
            Caja(1000, 700, 790, 150, "`.copy()` = una fotocopia", VERDE, 40, subtitulo="el original no cambia"),
        ),
    ),
    Escena(
        "Tu turno 2: elegir datos",
        Paso(
            "Tu turno. Con una máscara, elige todas las imágenes del siete y cuéntalas. Toma la mitad de arriba "
            "de todas las imágenes, y la última columna de X. Pausa el video y resuélvelo.",
            Codigo(130, 280, "sevens = None       # TODO: las imágenes del 7, (cantidad, 8, 8)\n"
                   "n_sevens = None     # TODO: cuántas son\n"
                   "top_half = None     # TODO: las 4 primeras filas de cada imagen, (1797, 4, 8)\n"
                   "last_column = None  # TODO: la última columna de X, (1797,)", 30),
            pausa=1.2,
        ),
    ),
    # -------------------------------------------------------------- reducciones
    Seccion("Parte 4 · Reducciones y axis"),
    Escena(
        "El eje que nombras desaparece",
        Paso(
            "Funciones como sum, mean, max o argmax reducen un arreglo. Sin argumentos, lo reducen todo a un solo número. "
            "Con axis, reducen solo a lo largo de ese eje.",
            Matriz(640, 270, 3, 4, 90, AZUL, valores=np.arange(12).reshape(3, 4), tam=32, etiqueta="`A`: (3, 4)"),
        ),
        Paso(
            "Con axis igual cero sumamos hacia abajo, a lo largo de las filas. El eje cero desaparece, "
            "y queda un número por columna: forma cuatro.",
            Flecha(820, 610, 820, 700, VERDE, 8, etiqueta="`axis=0`", lado="izquierda"),
            Matriz(640, 710, 1, 4, 90, VERDE, valores=[[12, 15, 18, 21]], tam=32, etiqueta="`A.sum(axis=0)`: (4,)"),
        ),
        Paso(
            "Con axis igual uno sumamos hacia la derecha. Desaparece el eje uno y queda un número por fila: forma tres. "
            "La regla es una sola: el eje que nombras, desaparece.",
            Flecha(1020, 405, 1150, 405, MORADO, 8, etiqueta="`axis=1`", lado="arriba"),
            Matriz(1170, 270, 3, 1, 90, MORADO, valores=[[6], [22], [38]], tam=32, etiqueta="`A.sum(axis=1)`: (3,)"),
            Caja(1300, 700, 520, 200, "**El eje que nombras desaparece.**", NARANJA, 38),
        ),
    ),
    Escena(
        "La imagen promedio",
        Paso(
            "Con el arreglo de imágenes, de forma mil setecientos noventa y siete, ocho, ocho, el promedio en el eje cero "
            "borra el eje de las imágenes y deja una imagen de ocho por ocho. El promedio en los ejes uno y dos "
            "deja un número por imagen: su brillo.",
            Tabla(130, 250, [
                ["Expresión", "Forma", "Qué es"],
                ["`images.mean()`", "`()`", "el brillo promedio de todo"],
                ["`images.mean(axis=0)`", "`(8, 8)`", "la imagen promedio"],
                ["`images.mean(axis=(1, 2))`", "`(1797,)`", "el brillo de cada imagen"],
            ], [720, 300, 640], tam=32),
        ),
        Paso(
            "Si combinamos una máscara con un promedio en el eje cero, obtenemos la imagen promedio de cada dígito. "
            "Son borrosas, pero cada una se reconoce. Más adelante, estas imágenes van a ser los pesos de una capa.",
            *fila_de_promedios(640),
        ),
    ),
    Escena(
        "argmax: de puntajes a predicciones",
        Paso(
            "Una red que clasifica produce un puntaje por clase, para cada ejemplo. Aquí hay dos ejemplos y tres clases.",
            Matriz(200, 320, 2, 3, 120, AZUL, valores=[[0.1, 2.0, 0.3], [1.5, 0.2, 0.9]], tam=34,
                   etiqueta="`scores`: (2 ejemplos, 3 clases)"),
            *[Texto(260 + j * 120, 270, f"clase {j}", 26, GRIS, alinear="centro") for j in range(3)],
        ),
        Paso(
            "La función argmax devuelve la posición del máximo, no su valor. Con axis igual uno, da la clase ganadora "
            "de cada ejemplo: la clase uno para el primero y la clase cero para el segundo. "
            "Vas a escribir esta línea todo el año.",
            Marco(320, 320, 120, 120, NARANJA),
            Marco(200, 440, 120, 120, NARANJA),
            Texto(800, 360, "`scores.argmax(axis=1)`  →  [1, 0]", 44, acento=NARANJA),
            Texto(800, 450, "la clase predicha para cada ejemplo", 34, GRIS),
        ),
    ),
    Escena(
        "Tu turno 3: reducciones",
        Paso(
            "Tu turno. Con X, calcula el promedio de cada píxel, el brillo de cada imagen, la posición de la imagen "
            "con más tinta, y cuántas imágenes hay de cada dígito. Antes de cada línea, di qué forma esperas. "
            "Pausa el video.",
            Codigo(130, 280, "pixel_mean = None    # TODO: (64,)\n"
                   "brightness = None    # TODO: (1797,)\n"
                   "darkest = None       # TODO: la posición del máximo de brightness\n"
                   "class_counts = None  # TODO: (10,); pista: np.bincount", 32),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------ broadcasting
    Seccion("Parte 5 · Broadcasting"),
    Escena(
        "Estirar sin copiar",
        Paso(
            "¿Qué pasa si sumamos arreglos de formas distintas? A veces NumPy estira el más chico para que encaje, "
            "sin copiar datos. Eso es el broadcasting. Ya lo usaste: en x más uno, el uno se estira a la forma de x.",
            Texto(130, 260, "`x + 1`: el 1 se estira a la forma de `x`", 40, acento=AZUL),
        ),
        Paso(
            "Aquí sumamos a una matriz de tres por cuatro una fila de cuatro números. NumPy repite la fila "
            "tres veces, como si la estampara en cada fila de la matriz.",
            Matriz(130, 400, 3, 4, 80, AZUL, valores=np.zeros((3, 4), int), tam=30, etiqueta="`A`: (3, 4)"),
            Texto(510, 490, "+", 60, alinear="centro"),
            Matriz(570, 400, 1, 4, 80, VERDE, valores=[[1, 2, 3, 4]], tam=30, etiqueta="`row`: (4,)"),
            Matriz(570, 480, 2, 4, 80, GRIS_CLARO, valores=[[1, 2, 3, 4]] * 2, tam=30),
            Texto(950, 490, "=", 60, alinear="centro"),
            Matriz(1010, 400, 3, 4, 80, VERDE, valores=[[1, 2, 3, 4]] * 3, tam=30, etiqueta="`A + row`: (3, 4)"),
            Texto(570, 730, "las filas grises no existen en memoria: NumPy hace como si estuvieran", 30, GRIS),
        ),
    ),
    Escena(
        "Las reglas del broadcasting",
        Paso(
            "Las reglas. Compara las formas alineadas por la derecha, eje por eje. "
            "Dos tamaños son compatibles si son iguales, o si uno de ellos es uno: el uno se estira. "
            "Y si a un arreglo le faltan ejes a la izquierda, se le agregan con tamaño uno.",
            *vinetas(130, 260, [
                "Alinea las formas **por la derecha**.",
                "En cada eje: tamaños **iguales**, o **uno de ellos es 1** (el 1 se estira).",
                "Si faltan ejes a la izquierda, se agregan con tamaño 1.",
            ], tam=38, sep=80, color=MORADO),
        ),
        Paso(
            "Mira los ejemplos. Tres por cuatro más cuatro funciona: la fila se suma a cada fila. "
            "Tres por cuatro más tres por uno también: la columna se suma a cada columna. "
            "Y tres por uno más uno por cuatro estira los dos, y da tres por cuatro.",
            Tabla(130, 560, [
                ["A", "B", "A + B"],
                ["`(3, 4)`", "`(4,)`", "`(3, 4)` ✓ la fila se suma a cada fila"],
                ["`(3, 4)`", "`(3, 1)`", "`(3, 4)` ✓ la columna se suma a cada columna"],
                ["`(3, 1)`", "`(1, 4)`", "`(3, 4)` ✓ se estiran los dos"],
            ], [260, 260, 1140], tam=32, id="t"),
        ),
        Paso(
            "Pero tres por cuatro más tres da error: alineados por la derecha, el cuatro y el tres no coinciden, "
            "y ninguno es uno.",
            Tabla(130, 560, [
                ["A", "B", "A + B"],
                ["`(3, 4)`", "`(4,)`", "`(3, 4)` ✓ la fila se suma a cada fila"],
                ["`(3, 4)`", "`(3, 1)`", "`(3, 4)` ✓ la columna se suma a cada columna"],
                ["`(3, 1)`", "`(1, 4)`", "`(3, 4)` ✓ se estiran los dos"],
                ["`(3, 4)`", "`(3,)`", "✗ error: 4 y 3 no son compatibles"],
            ], [260, 260, 1140], tam=32),
            Marco(122, 826, 1676, 72, ROJO),
            ocultar=["t"],
        ),
    ),
    Escena(
        "None agrega un eje",
        Paso(
            "Para agregar un eje de tamaño uno donde lo necesites, se escribe None en el índice. "
            "Si v tiene forma cinco, v con dos puntos coma None tiene forma cinco por uno: una columna. "
            "Y v con None coma dos puntos es una fila de uno por cinco.",
            Texto(130, 270, "`v[:, None]`  →  (5, 1)   una columna", 40, acento=AZUL),
            Texto(130, 350, "`v[None, :]`  →  (1, 5)   una fila", 40, acento=VERDE),
        ),
        Paso(
            "Multiplica la columna por la fila, y el broadcasting estira las dos: sale la tabla de multiplicar "
            "del uno al cinco, sin un solo bucle.",
            Matriz(400, 500, 5, 1, 70, AZUL, valores=[[i] for i in range(1, 6)], tam=28),
            Texto(520, 650, "×", 56, alinear="centro"),
            Matriz(580, 640, 1, 5, 70, VERDE, valores=[list(range(1, 6))], tam=28),
            Texto(990, 650, "=", 56, alinear="centro"),
            Matriz(1060, 500, 5, 5, 70, MORADO, valores=np.outer(range(1, 6), range(1, 6)), tam=28,
                   etiqueta="(5, 1) × (1, 5) → (5, 5)"),
        ),
    ),
    Escena(
        "Estandarizar: restar la media de cada columna",
        Paso(
            "Un uso real. Antes de entrenar casi siempre estandarizamos cada variable: le restamos su media "
            "y la dividimos por su desviación estándar, para que todas queden en una escala parecida.",
            Codigo(130, 260, "mu = X.mean(axis=0)       # (64,)\nsigma = X.std(axis=0)     # (64,)\n"
                   "X_std = (X - mu) / sigma  # (1797, 64) - (64,) → (1797, 64)", 32, id="c"),
        ),
        Paso(
            "X tiene forma mil setecientos noventa y siete por sesenta y cuatro, y la media tiene forma sesenta y cuatro. "
            "Gracias al broadcasting, la media de cada columna se resta en todas las filas.",
            Codigo(130, 260, "mu = X.mean(axis=0)       # (64,)\nsigma = X.std(axis=0)     # (64,)\n"
                   "X_std = (X - mu) / sigma  # (1797, 64) - (64,) → (1797, 64)", 32, resaltar={3}, id="c"),
            ocultar=["c"],
        ),
        Paso(
            "Un detalle de datos reales: tres píxeles del borde valen cero en todas las imágenes, "
            "así que su desviación es cero, y dividir por cero da un valor llamado nan, que significa no es un número. "
            "La solución es reemplazar esas desviaciones por uno.",
            Terminal(130, 560, "píxeles con desviación 0: 3", 30, ancho=1660),
            Codigo(130, 700, "sigma = np.where(sigma == 0, 1.0, sigma)", 32),
        ),
    ),
    Escena(
        "Tu turno 4: todas las distancias de una vez",
        Paso(
            "Para comparar dos imágenes, una medida natural es la distancia al cuadrado: la suma de las diferencias "
            "al cuadrado, píxel por píxel. Tu tarea es calcularla entre todas las imágenes de A y todas las de B, "
            "sin bucles.",
            Texto(130, 260, "‖a − b‖² = Σₖ (aₖ − bₖ)²", 46, acento=AZUL),
            Texto(130, 350, "`pairwise_sq_dists(A, B)`:  A (n, d),  B (m, d)  →  D (n, m)", 38, acento=AZUL),
        ),
        Paso(
            "El plan: a A le agregas un eje en el medio, y a B uno al principio. Al restarlos, el broadcasting "
            "produce todas las diferencias, con forma n, m, d. Luego elevas al cuadrado y sumas sobre el último eje. "
            "Pausa el video y escríbelo.",
            Caja(130, 480, 370, 150, "`A[:, None, :]`", AZUL, 34, subtitulo="(n, 1, d)"),
            Texto(545, 520, "−", 60, alinear="centro"),
            Caja(590, 480, 370, 150, "`B[None, :, :]`", VERDE, 34, subtitulo="(1, m, d)"),
            Flecha(970, 555, 1060, 555, MORADO),
            Caja(1070, 480, 300, 150, "**diferencias**", MORADO, 34, subtitulo="(n, m, d)"),
            Flecha(1380, 555, 1560, 555, NARANJA, etiqueta="² y suma", lado="arriba"),
            Caja(1570, 480, 230, 150, "**D**", NARANJA, 36, subtitulo="(n, m)"),
            pausa=1.2,
        ),
    ),
    Escena(
        "Un clasificador con lo que ya sabes",
        Paso(
            "Con esa función ya tienes un clasificador completo. Para cada imagen nueva, buscas la imagen de entrenamiento "
            "más parecida, la de menor distancia, y copias su etiqueta. Se llama vecino más cercano, "
            "y lo veremos a fondo en la semana cinco.",
            Caja(130, 280, 480, 170, "imagen nueva", AZUL, 38),
            Flecha(620, 365, 760, 365, etiqueta="distancias", tam=28),
            Caja(770, 280, 480, 170, "la más parecida\ndel entrenamiento", VERDE, 36),
            Flecha(1260, 365, 1400, 365, etiqueta="su etiqueta", tam=28),
            Caja(1410, 280, 380, 170, "**predicción**", NARANJA, 38),
        ),
        Paso(
            "Primero, la línea base: predecir siempre el dígito más común acierta solo el siete por ciento. "
            "El vecino más cercano acierta casi el noventa y nueve por ciento. Y toda la predicción son tres líneas "
            "sin bucles.",
            Terminal(130, 560, "línea base (siempre 6): accuracy 0.074\n"
                     "vecino más cercano:        accuracy 0.987", 32, ancho=1660),
            Codigo(130, 760, "D = pairwise_sq_dists(X_test, X_train)  # (297, 1500)\n"
                   "pred = y_train[D.argmin(axis=1)]        # la etiqueta del vecino más cercano", 30),
        ),
    ),
    # ------------------------------------------------------------ álgebra lineal
    Seccion("Parte 6 · Álgebra lineal"),
    Escena(
        "El producto punto",
        Paso(
            "Última parte: álgebra lineal. El producto punto multiplica dos vectores elemento a elemento, y suma. "
            "Uno por cuatro, más dos por menos uno, más tres por cero coma cinco: da tres coma cinco.",
            Matriz(200, 300, 1, 3, 100, AZUL, valores=[[1.0, 2.0, 3.0]], tam=34, etiqueta="`a`"),
            Texto(560, 320, "·", 70, alinear="centro"),
            Matriz(620, 300, 1, 3, 100, VERDE, valores=[[4.0, -1.0, 0.5]], tam=34, etiqueta="`b`"),
            Texto(1000, 320, "= 4 − 2 + 1.5 = **3.5**", 46),
        ),
        Paso(
            "En NumPy se escribe a arroba b. Y es la operación más común de todo el deep learning: "
            "una neurona multiplica cada entrada por su peso, suma todo, y le agrega un sesgo.",
            Texto(200, 470, "`a @ b`  →  3.5", 42, acento=AZUL),
            Linea(440, 640, 740, 740, "#B9C7E8", 4), Linea(440, 740, 740, 740, "#B9C7E8", 4),
            Linea(440, 840, 740, 740, "#B9C7E8", 4),
            Circulo(400, 640, 40, "x₁", AZUL), Circulo(400, 740, 40, "x₂", AZUL), Circulo(400, 840, 40, "x₃", AZUL),
            Circulo(780, 740, 50, "Σ", VERDE),
            Texto(900, 710, "una neurona:  **w · x + b**", 44, acento=VERDE),
        ),
    ),
    Escena(
        "Matriz × vector = una capa",
        Paso(
            "Una capa tiene varias neuronas, cada una con su vector de pesos. Si apilamos esos vectores "
            "como filas de una matriz W, la capa entera es una multiplicación de matriz por vector: "
            "cada fila hace su producto punto con x.",
            Matriz(200, 300, 3, 4, 80, MORADO, etiqueta="`W`: (3, 4)"),
            Texto(580, 380, "@", 56, alinear="centro"),
            Matriz(640, 260, 4, 1, 80, AZUL, etiqueta="`x`: (4,)"),
            Texto(800, 380, "=", 56, alinear="centro"),
            Matriz(860, 300, 3, 1, 80, VERDE, etiqueta="(3,)"),
            Marco(195, 295, 330, 90, NARANJA),
            Marco(855, 295, 90, 90, NARANJA),
        ),
        Paso(
            "Piensa en cada fila de W como un detector. Su producto punto es grande cuando x se parece a esa fila. "
            "Tres filas, tres detectores, tres salidas.",
            Caja(1100, 280, 700, 260, "Cada fila de `W` es un **detector**", NARANJA, 38,
                 subtitulo="su producto punto es grande\ncuando x se parece a esa fila"),
            Texto(200, 720, "(k, d) @ (d,) → (k,)", 46, acento=MORADO),
        ),
    ),
    Escena(
        "Matriz × matriz = un lote",
        Paso(
            "Nunca pasamos un ejemplo a la vez, sino un lote. Si ponemos cinco ejemplos como filas de una matriz X, "
            "toda la capa sobre todo el lote es un solo producto de matrices: X por W transpuesta, más el sesgo. "
            "Es la capa Linear de la semana uno, y la de PyTorch.",
            Matriz(130, 300, 5, 4, 60, AZUL, etiqueta="`X`: (5, 4)"),
            Texto(420, 420, "@", 56, alinear="centro"),
            Matriz(480, 330, 4, 3, 60, MORADO, etiqueta="`W.T`: (4, 3)"),
            Texto(710, 420, "+", 56, alinear="centro"),
            Matriz(770, 390, 1, 3, 60, NARANJA, etiqueta="`b`: (3,)"),
            Texto(1000, 420, "=", 56, alinear="centro"),
            Matriz(1060, 300, 5, 3, 60, VERDE, etiqueta="`Y`: (5, 3)"),
        ),
        Paso(
            "La regla de formas del producto de matrices: los dos ejes de adentro deben coincidir, y desaparecen. "
            "Cinco por cuatro, por cuatro por tres, da cinco por tres. Y el sesgo se suma a cada fila por broadcasting.",
            Texto(130, 720, "(5, **4**) @ (**4**, 3) → (5, 3)", 50, acento=ROJO),
            Texto(130, 810, "los ejes de adentro deben coincidir, y desaparecen", 36, GRIS),
            Texto(1300, 720, "+ `b` por broadcasting", 38, acento=NARANJA),
        ),
    ),
    Escena(
        "Tu turno 5: el producto de matrices por dentro",
        Paso(
            "Tu turno, en dos partes. Primero, escribe el producto de matrices con bucles: el elemento i, j "
            "es el producto punto de la fila i de A con la columna j de B. Si los ejes de adentro no coinciden, "
            "lanza un ValueError.",
            Codigo(130, 260, "def matmul_loops(A, B):\n    # TODO: C[i, j] = fila i de A · columna j de B\n"
                   "    raise NotImplementedError", 32),
        ),
        Paso(
            "Segundo, sin ejecutar nada, predice la forma de cada expresión, o escribe error si NumPy no puede calcularla. "
            "Usa la regla de los ejes de adentro y la del broadcasting. Pausa el video.",
            Codigo(130, 520, "# a: (2, 3)   b: (3, 4)   v: (3,)   w: (4,)\n"
                   'predicted = {"a @ b": None, "a @ v": None, "b @ a": None,\n'
                   '             "a.T @ a": None, "a @ b + w": None, "a @ b + v": None}', 30),
            pausa=1.2,
        ),
        Paso(
            "Cuando termines, el notebook compara tu versión con bucles contra la arroba, con matrices de doscientos "
            "por doscientos. Ahí vas a ver por qué nadie escribe estos bucles a mano.",
            Caja(130, 820, 1660, 120, "Compara tus bucles con `A @ B` en matrices de 200 × 200.", NARANJA, 36),
        ),
    ),
    Escena(
        "Una capa que reconoce dígitos",
        Paso(
            "Para cerrar, una capa con pesos elegidos a mano. Cada fila de W es la imagen promedio de un dígito: "
            "un detector por dígito.",
            *fila_de_promedios(280, etiquetas=[f"fila {c}" for c in range(10)]),
        ),
        Paso(
            "El producto punto mide parecido, pero favorece a las imágenes con más tinta. Al desarrollar la distancia "
            "a cada promedio, sale exactamente una capa lineal: pesos iguales a dos veces el promedio, "
            "y un sesgo igual a menos su norma al cuadrado.",
            Texto(960, 520, "−‖x − μ_c‖²  =  2 μ_c · x  −  ‖μ_c‖²  −  ‖x‖²", 44, alinear="centro"),
            Texto(960, 600, "pesos  W_c = 2μ_c  ·  sesgo  b_c = −‖μ_c‖²  ·  el último término es igual para todos",
                  32, GRIS, alinear="centro"),
        ),
        Paso(
            "Una sola multiplicación de matrices clasifica doscientas noventa y siete imágenes a la vez, "
            "y acierta el noventa y dos por ciento, contra el siete por ciento de la línea base. "
            "Los pesos los elegimos nosotros. Entrenar una red es dejar que el gradiente los elija, "
            "y eso empieza en la semana cuatro.",
            Terminal(130, 700, "scores = X_test @ W_digits.T + b_digits   # (297, 10)\n"
                     "línea base:                accuracy 0.074\n"
                     "capa con pesos a mano:     accuracy 0.916", 30, ancho=1660),
            pausa=0.8,
        ),
    ),
    # ------------------------------------------------------------------ cierre
    Seccion("Cierre"),
    Escena(
        "Lo que vimos hoy",
        *[
            Paso(decir, vineta)
            for decir, vineta in zip(
                [
                    "Repasemos. Vectorizar es operar sobre arreglos enteros en vez de usar bucles: muchas veces más rápido.",
                    "Todo arreglo tiene una forma. La función reshape la cambia sin cambiar los números, y una imagen también es un arreglo.",
                    "Para elegir datos hay cortes, que son ventanas y no copias, máscaras y listas de posiciones.",
                    "En las reducciones, el eje que nombras desaparece. Y argmax convierte puntajes en predicciones.",
                    "Con broadcasting, las formas se alinean por la derecha y los unos se estiran.",
                    "Y el álgebra lineal de una red: el producto punto es una neurona, matriz por vector es una capa, "
                    "y matriz por matriz es un lote que pasa por esa capa.",
                ],
                vinetas(130, 250, [
                    "**Vectorizar:** arreglos enteros en vez de bucles.",
                    "**Formas:** `shape`, `reshape` con `-1`. Una imagen es un arreglo.",
                    "**Indexación:** cortes (vistas, no copias), máscaras y listas.",
                    "**Reducciones:** el eje de `axis` desaparece; `argmax(axis=1)` predice.",
                    "**Broadcasting:** alinear por la derecha; el 1 se estira.",
                    "**Álgebra lineal:** punto = neurona · matriz × vector = capa · matriz × matriz = lote.",
                ], tam=38, sep=118, ancho=1650),
            )
        ],
    ),
    Escena(
        "Para la semana 3",
        Paso(
            "Para la próxima clase, entrega el trabajo B1, ejercicios de NumPy: vas a vectorizar seis funciones "
            "escritas con bucles y medir cuánto más rápidas quedan. Se guarda en la carpeta B1 de tu repositorio de entregas.",
            Vineta(130, 260, "**Entrega al inicio de la clase:** B1, Ejercicios de NumPy, en "
                   "`B1/B1_ejercicios_numpy.ipynb`.", 40, NARANJA, numero=1, ancho=1600),
        ),
        Paso(
            "Haz la lectura previa de MÓLÓ, en inglés: NumPy avanzado, que te ayuda con el B1, "
            "análisis de datos con pandas, y estadística descriptiva.",
            Vineta(130, 440, "**Lectura previa** (MÓLÓ): Advanced NumPy, Data Analysis with Pandas y "
                   "Descriptive statistics.", 40, NARANJA, numero=2, ancho=1600),
        ),
        Paso(
            "Y no olvides la cuota semanal en la plataforma. La semana tres es un laboratorio con datos reales.",
            Vineta(130, 620, "**Cuota semanal** en ioai.artix.tech.", 40, NARANJA, numero=3, ancho=1600),
            Texto(130, 760, "Los enlaces están al final de `sem02_numpy.ipynb`.", 36, GRIS),
        ),
        Paso(
            "Eso es todo por hoy. ¡Nos vemos en la semana tres!",
            Texto(960, 880, "**¡Nos vemos en la semana 3!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
