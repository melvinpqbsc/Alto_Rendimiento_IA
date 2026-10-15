"""Video de la semana 3: Nivel_Intermedio/clases/sem03_pandas.ipynb."""

from motor import (AZUL, GRIS, MORADO, NARANJA, ROJO, TINTA, VERDE, Caja, Codigo, Dispersion, Escena, Flecha,
                   Histograma, Marco, Matriz, Paso, Pixeles, Seccion, Tabla, Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 3 · Datos reales: pandas, gráficos y estadística descriptiva · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}

# Palmer Penguins sin las dos filas vacías (342 pingüinos), copiado aquí porque el entorno
# de los videos no tiene pandas. Por pingüino: largo del pico ×10, alto del pico ×10 (mm),
# largo de la aleta (mm) y masa / 25 (g).
_PINGUINOS = {
    "Adelie": (
        "391,187,181,150 395,174,186,152 403,180,195,130 367,193,193,138 393,206,190,146 389,178,181,145 "
        "392,196,195,187 341,181,193,139 420,202,190,170 378,171,186,132 378,173,180,148 411,176,182,128 "
        "386,212,191,152 346,211,198,176 366,178,185,148 387,190,195,138 425,207,197,180 344,184,184,133 "
        "460,215,194,168 378,183,174,136 377,187,180,144 359,192,189,152 382,181,185,158 388,172,180,152 "
        "353,189,187,152 406,186,183,142 405,179,187,128 379,186,172,126 405,189,180,158 395,167,178,130 "
        "372,181,178,156 395,178,188,132 409,189,184,156 364,170,195,133 392,211,196,166 388,200,190,158 "
        "422,185,180,142 376,193,181,132 398,191,184,186 365,180,182,126 408,184,195,156 360,185,186,124 "
        "441,197,196,176 370,169,185,120 396,188,190,184 411,190,182,137 375,189,179,119 360,179,190,138 "
        "423,212,191,166 396,177,186,140 401,189,188,172 350,179,190,138 420,195,200,162 345,181,187,116 "
        "414,186,191,148 390,175,186,142 406,188,193,152 365,166,181,114 376,191,194,150 357,169,185,126 "
        "413,211,195,176 376,170,185,144 411,182,192,162 364,171,184,114 416,180,192,158 355,162,195,134 "
        "411,191,188,164 359,166,190,122 418,194,198,178 335,190,190,144 397,184,190,156 396,172,196,142 "
        "458,189,197,166 355,175,190,148 428,185,195,170 409,168,191,148 372,194,184,156 362,161,187,142 "
        "421,191,195,160 346,172,189,128 429,176,196,188 367,188,187,152 351,194,193,168 373,178,191,134 "
        "413,203,194,142 363,195,190,152 369,186,189,140 383,192,189,158 389,188,190,144 357,180,202,142 "
        "411,181,205,172 340,171,185,136 396,181,186,178 362,173,187,132 408,189,208,172 381,186,190,148 "
        "403,185,196,174 331,161,178,116 432,185,192,164 350,179,192,149 410,200,203,189 377,160,183,123 "
        "378,200,190,170 379,186,193,117 397,189,184,142 386,172,199,150 382,200,190,156 381,170,181,127 "
        "432,190,197,191 381,165,198,153 456,203,191,184 397,177,193,128 422,195,197,171 396,207,191,156 "
        "427,183,196,163 386,170,188,116 373,205,199,151 357,170,189,134 411,186,189,133 362,172,187,126 "
        "377,198,198,140 402,170,176,138 414,185,202,155 352,159,186,122 406,190,199,160 388,176,191,131 "
        "415,183,195,172 390,171,191,122 441,180,210,160 385,179,190,133 431,192,197,140 368,185,193,140 "
        "375,185,199,179 381,176,187,137 411,175,190,156 356,175,191,127 402,201,200,159 370,165,185,136 "
        "397,179,193,170 402,171,193,136 406,172,187,139 321,155,188,122 407,170,190,149 373,168,192,120 "
        "390,187,185,146 392,186,190,170 366,184,184,139 360,178,195,138 378,181,193,150 360,171,187,148 "
        "415,185,201,160 "
    ),
    "Chinstrap": (
        "465,179,192,140 500,195,196,156 513,192,193,146 454,187,188,141 527,198,197,149 452,178,198,158 "
        "461,182,178,130 513,182,197,150 460,189,195,166 513,199,198,148 466,178,193,152 517,203,194,151 "
        "470,173,185,148 520,181,201,162 459,171,190,143 505,196,201,162 503,200,197,132 580,178,181,148 "
        "464,186,190,138 492,182,195,176 424,173,181,144 485,175,191,136 432,166,187,116 506,194,193,152 "
        "467,179,195,132 520,190,197,166 505,184,200,136 495,190,200,152 464,178,191,148 528,200,205,182 "
        "409,166,187,128 542,208,201,172 425,167,187,134 510,188,203,164 497,186,195,144 475,168,199,156 "
        "476,183,195,154 520,207,210,192 469,166,192,108 535,199,205,180 490,195,210,158 462,175,187,146 "
        "509,191,196,142 455,170,196,140 509,179,196,147 508,185,201,178 501,179,190,136 490,196,212,172 "
        "515,187,187,130 498,173,198,147 481,164,199,133 514,190,201,158 457,173,193,144 507,197,203,162 "
        "425,173,187,134 522,188,197,138 452,166,191,130 493,199,203,162 502,188,202,152 456,194,194,141 "
        "519,195,206,158 468,165,189,146 457,170,195,146 558,198,207,160 435,181,202,136 496,182,193,151 "
        "508,190,210,164 502,187,198,151 "
    ),
    "Gentoo": (
        "461,132,211,180 500,163,230,228 487,141,210,178 500,152,218,228 476,145,215,216 465,135,210,182 "
        "454,146,211,192 467,153,219,208 433,134,209,176 468,154,215,206 409,137,214,186 490,161,216,222 "
        "455,137,214,186 484,146,213,234 458,146,210,168 493,157,217,234 420,135,210,166 492,152,221,252 "
        "462,145,209,192 487,151,222,214 502,143,218,228 451,145,215,200 465,145,213,176 463,158,215,202 "
        "429,131,215,200 461,151,215,204 445,143,216,164 478,150,215,226 482,143,210,184 500,153,220,222 "
        "473,153,222,210 428,142,209,188 451,145,207,202 596,170,230,242 491,148,220,206 484,163,220,216 "
        "426,137,213,198 444,173,219,210 440,136,208,174 487,157,208,214 427,137,208,158 496,160,225,228 "
        "453,137,210,172 496,150,216,190 505,159,222,222 436,139,217,196 455,139,210,168 505,159,225,216 "
        "449,133,213,204 452,158,215,212 466,142,210,194 485,141,220,212 451,144,210,176 501,150,225,200 "
        "465,144,217,196 450,154,220,202 438,139,208,172 455,150,220,200 432,145,208,178 504,153,224,222 "
        "453,138,208,168 462,149,221,212 457,139,214,176 543,157,231,226 458,142,219,188 498,168,230,228 "
        "462,144,214,186 495,162,229,232 435,142,220,188 507,150,223,222 477,150,216,190 464,156,221,200 "
        "482,156,221,204 465,148,217,208 464,150,216,188 486,160,230,232 475,142,209,184 511,163,220,240 "
        "452,138,215,190 452,164,223,238 491,145,212,185 525,156,221,218 474,146,212,189 500,159,224,214 "
        "449,138,212,190 508,173,228,224 434,144,218,184 513,142,218,212 475,140,212,195 521,170,230,222 "
        "475,150,218,198 522,171,228,216 455,145,212,190 495,161,224,226 445,147,214,194 508,157,226,208 "
        "494,158,216,197 469,146,222,195 484,144,203,185 511,165,225,210 485,150,219,194 559,170,228,224 "
        "472,155,215,199 491,150,228,220 473,138,216,189 468,161,215,220 417,147,210,188 534,158,219,220 "
        "433,140,208,183 481,151,209,220 505,152,216,200 498,159,229,238 435,152,213,186 515,163,230,220 "
        "462,141,217,175 551,160,230,234 445,157,217,195 488,162,222,240 472,137,214,197 468,143,215,194 "
        "504,157,222,230 452,148,212,208 499,161,213,216 "
    ),
}
ESPECIES = ["Adelie", "Chinstrap", "Gentoo"]
COLOR = {"Adelie": AZUL, "Chinstrap": NARANJA, "Gentoo": VERDE}


def _columnas(s):
    v = np.array([[int(n) for n in p.split(",")] for p in s.split()], float)
    return {"pico": v[:, 0] / 10, "alto": v[:, 1] / 10, "aleta": v[:, 2], "masa": v[:, 3] * 25}


DATOS = {e: _columnas(s) for e, s in _PINGUINOS.items()}
BORDES = list(range(2600, 6600, 200))


def conteos(valores):
    return np.histogram(valores, BORDES)[0].tolist()


MASA_TODOS = conteos(np.concatenate([DATOS[e]["masa"] for e in ESPECIES]))
MASA_POR_ESPECIE = [conteos(DATOS[e]["masa"]) for e in ESPECIES]


def histograma_masa(por_especie, id=None):
    comun = dict(ymax=44, marcas_x=range(3000, 6500, 500), marcas_y=range(0, 45, 10),
                 etiqueta_x="masa corporal (g)", etiqueta_y="cantidad de pingüinos", id=id)
    if por_especie:
        return Histograma(230, 270, 1400, 500, BORDES, MASA_POR_ESPECIE, [COLOR[e] for e in ESPECIES], ESPECIES,
                          **comun)
    return Histograma(230, 270, 1400, 500, BORDES, [MASA_TODOS], **comun)


def picos(x, w, por_especie, tendencia=False, id=None):
    comun = dict(rx=(31, 61), ry=(12.5, 24.5), marcas_x=range(35, 61, 5), marcas_y=range(14, 22, 2),
                 etiqueta_x="largo del pico (mm)", etiqueta_y="alto del pico (mm)", tendencia=tendencia, radio=5, id=id)
    if por_especie:
        return Dispersion(x, 290, w, 480, [(DATOS[e]["pico"], DATOS[e]["alto"]) for e in ESPECIES],
                          [COLOR[e] for e in ESPECIES], ESPECIES, **comun)
    todos = (np.concatenate([DATOS[e]["pico"] for e in ESPECIES]), np.concatenate([DATOS[e]["alto"] for e in ESPECIES]))
    return Dispersion(x, 290, w, 480, [todos], **comun)


# Una imagen del conjunto digits (8×8, valores 0–16): el primer 0, como en la semana 2.
PRIMERA = np.array([int(c, 17) for c in "005d910000dfaf5003f20b8004c008800580098004b01c7002e5ac00006da000"]).reshape(8, 8)

CABECERA = ["species", "island", "bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g", "sex"]
FILAS = [
    ["Adelie", "Torgersen", "39.1", "18.7", "181.0", "3750.0", "MALE"],
    ["Adelie", "Torgersen", "39.5", "17.4", "186.0", "3800.0", "FEMALE"],
    ["Adelie", "Torgersen", "40.3", "18.0", "195.0", "3250.0", "FEMALE"],
    ["Adelie", "Torgersen", "NaN", "NaN", "NaN", "NaN", "NaN"],
    ["Adelie", "Torgersen", "36.7", "19.3", "193.0", "3450.0", "FEMALE"],
]
ANCHOS = [190, 230, 250, 250, 290, 250, 190]


def tabla_pinguinos(y, filas=FILAS, id=None):
    return Tabla(110, y, [CABECERA] + filas, ANCHOS, tam=24, id=id)


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la tercera clase del nivel intermedio. Es un laboratorio con datos reales: "
            "pandas, gráficos y estadística descriptiva.",
            Texto(960, 250, "SEMANA 3 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**Datos reales con pandas**", 92, alinear="centro"),
            Texto(960, 460, "tablas, datos faltantes, grupos, gráficos y correlación", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Vamos a trabajar con trescientos cuarenta y cuatro pingüinos de la Antártida, de tres especies. "
            "Aprenderemos a cargar y limpiar la tabla, a resumirla con números, a compararla por grupos, "
            "y a mirarla con gráficos.",
            Caja(110, 640, 400, 200, "**1**\nCargar y limpiar", AZUL, 38),
            Caja(560, 640, 400, 200, "**2**\nResumir con números", VERDE, 38),
            Caja(1010, 640, 400, 200, "**3**\nComparar grupos", MORADO, 38),
            Caja(1460, 640, 400, 200, "**4**\nGráficos y correlación", NARANJA, 38),
        ),
    ),
    # ------------------------------------------------------------- cargar un CSV
    Seccion("Parte 1 · Cargar un CSV"),
    Escena(
        "Un CSV es una tabla escrita como texto",
        Paso(
            "Piensa en una lista de asistencia escrita en un cuaderno: una línea por estudiante, y sus datos "
            "separados por comas. Eso es un archivo CSV. Es el formato más común de las tablas de datos, "
            "y el de muchos problemas de competencia.",
            Codigo(110, 260, "species,island,bill_length_mm,bill_depth_mm,flipper_length_mm,body_mass_g,sex\n"
                   "Adelie,Torgersen,39.1,18.7,181.0,3750.0,MALE\n"
                   "Adelie,Torgersen,39.5,17.4,186.0,3800.0,FEMALE\n"
                   "Adelie,Torgersen,40.3,18.0,195.0,3250.0,FEMALE", 28, titulo="penguins.csv, como texto"),
        ),
        Paso(
            "Con pandas, una sola línea lo lee y lo convierte en un DataFrame: una tabla con un nombre en cada columna, "
            "como una hoja de cálculo. Cada columna, por separado, se llama Series.",
            Codigo(110, 590, "df = pd.read_csv(URL)", 32),
            Texto(700, 610, "→  un **DataFrame**: una hoja de cálculo en Python", 38, acento=AZUL),
        ),
        Paso(
            "Nuestra tabla tiene trescientas cuarenta y cuatro filas y siete columnas. En machine learning, "
            "cada fila es un ejemplo, aquí un pingüino, y cada columna es una variable.",
            Terminal(110, 740, "forma: (344, 7)      344 pingüinos × 7 variables", 30, ancho=1700),
        ),
    ),
    Escena(
        "Filas, columnas y tipos",
        Paso(
            "Estas son las primeras filas. La primera línea de la tabla son los nombres de las columnas.",
            tabla_pinguinos(260),
        ),
        Paso(
            "Lo primero que hay que mirar en una tabla nueva es el tipo de cada columna. Las medidas, como el largo "
            "del pico o la masa, son números: float sesenta y cuatro.",
            Marco(525, 255, 1050, 320, AZUL),
            Texto(530, 610, "**numéricas**: `float64`", 36, acento=AZUL),
        ),
        Paso(
            "La especie, la isla y el sexo son texto. Toman unos pocos valores fijos, así que se llaman variables "
            "categóricas.",
            Marco(105, 255, 425, 320, MORADO),
            Marco(1565, 255, 200, 320, MORADO),
            Texto(110, 700, "**categóricas**: `object`, unos pocos valores fijos", 36, acento=MORADO),
        ),
    ),
    Escena(
        "Ejercicio 1: conocer la tabla",
        Paso(
            "Primer ejercicio. Guarda cuántas filas y columnas hay, y haz la lista de las columnas numéricas y la de "
            "las categóricas. Pausa el video y resuélvelo en el notebook.",
            Codigo(110, 280, "n_rows = None            # TODO\n"
                   "n_cols = None            # TODO\n"
                   "numeric_cols = None      # TODO: pista, df.select_dtypes(\"number\")\n"
                   "categorical_cols = None  # TODO", 32),
            Caja(110, 640, 1700, 140, "Cada ejercicio tiene una celda que lo revisa: ✓, ✗ o «todavía sin resolver».",
                 VERDE, 36),
            pausa=1.2,
        ),
    ),
    # ---------------------------------------------------------------- filtrar
    Seccion("Parte 2 · Seleccionar y filtrar"),
    Escena(
        "Una máscara, ahora con nombres",
        Paso(
            "Elegir columnas es como en NumPy, pero con nombres en vez de posiciones. Con un nombre entre corchetes, "
            "sale una columna. Con una lista de nombres, sale una tabla más chica.",
            Texto(110, 260, "`df[\"species\"]`  →  una columna, una **Series**", 38, acento=AZUL),
            Texto(110, 340, "`df[[\"species\", \"body_mass_g\"]]`  →  dos columnas, un **DataFrame**", 38, acento=AZUL),
        ),
        Paso(
            "Para elegir filas usamos el mismo colador de la semana dos. La comparación body mass mayor que cinco mil "
            "da verdadero o falso para cada pingüino.",
            Matriz(110, 480, 1, 8, 140, AZUL, valores=[[3750, 5700, 3250, 4500, 5850, 3450, 6300, 4100]], tam=28,
                   etiqueta="`df[\"body_mass_g\"]`"),
            Matriz(110, 680, 1, 8, 140, GRIS, valores=[["F", "T", "F", "F", "T", "F", "T", "F"]], tam=28,
                   etiqueta="`df[\"body_mass_g\"] > 5000`"),
        ),
        Paso(
            "Y al indexar la tabla con esa máscara, solo pasan las filas marcadas: los pingüinos pesados.",
            Marco(250, 672, 140, 156, NARANJA), Marco(670, 672, 140, 156, NARANJA), Marco(950, 672, 140, 156, NARANJA),
            Texto(110, 890, "`df[df[\"body_mass_g\"] > 5000]`  →  solo las filas con T", 38, acento=NARANJA),
        ),
    ),
    Escena(
        "Dos condiciones y value_counts",
        Paso(
            "Para dos condiciones a la vez se usa el ampersand, que significa y. O la barra vertical, que significa o. "
            "Cuidado: cada condición va entre paréntesis, o pandas se confunde.",
            Codigo(110, 260, 'heavy_gentoo = df[(df["species"] == "Gentoo") & (df["body_mass_g"] > 5500)]', 30),
            Terminal(110, 400, "Gentoo de más de 5,5 kg: 28", 30, ancho=1700),
        ),
        Paso(
            "Y para una columna categórica, value counts cuenta cuántas veces aparece cada valor. "
            "Hay ciento cincuenta y dos Adelie, ciento veinticuatro Gentoo y sesenta y ocho Chinstrap.",
            Codigo(110, 560, 'df["species"].value_counts()', 30),
            Terminal(110, 690, "species\nAdelie       152\nGentoo       124\nChinstrap     68", 30, ancho=760),
        ),
    ),
    Escena(
        "Ejercicio 2: filtrar",
        Paso(
            "Tu turno. Cuenta los Gentoo y los pingüinos de más de cinco mil gramos. Calcula qué fracción vive en la "
            "isla Biscoe: el promedio de una máscara es la fracción de verdaderos. Y busca la especie del pingüino con "
            "la aleta más larga. Pausa el video.",
            Codigo(110, 280, "n_gentoo = None                 # TODO\n"
                   "n_heavy = None                  # TODO: body_mass_g > 5000\n"
                   "biscoe_share = None             # TODO: una fracción entre 0 y 1\n"
                   "longest_flipper_species = None  # TODO: pista, idxmax()", 32),
            pausa=1.2,
        ),
    ),
    # ---------------------------------------------------------------- faltantes
    Seccion("Parte 3 · Datos faltantes"),
    Escena(
        "Huecos en la tabla",
        Paso(
            "Imagina una encuesta en papel en la que alguien dejó casillas en blanco. Los datos reales siempre tienen "
            "huecos así: un sensor que falló, una medida que nadie anotó. Pandas los marca con NaN, "
            "que significa no es un número.",
            tabla_pinguinos(260),
            Marco(525, 462, 1240, 60, ROJO),
        ),
        Paso(
            "Antes de calcular nada, hay que saber cuántos hay y dónde. Is na, punto sum, los cuenta por columna. "
            "Dos pingüinos no tienen ninguna medida, y a otros nueve solo les falta el sexo.",
            Codigo(110, 720, "df.isna().sum()", 30),
            Terminal(700, 700, "bill_length_mm    2    body_mass_g    2\n"
                     "bill_depth_mm     2    sex           11\n"
                     "flipper_length_mm 2", 28, ancho=1110),
        ),
    ),
    Escena(
        "Borrar, rellenar o dejar",
        Paso(
            "Hay tres salidas, y elegir es parte del análisis. Borrar las filas, si son pocas y no tienen nada útil. "
            "Rellenar con un valor: la mediana para un número, o desconocido para una categoría. "
            "O dejarlas, si esa columna no se va a usar.",
            Caja(110, 270, 520, 230, "**Borrar**", ROJO, 40, subtitulo="`dropna(subset=[...])`\npocas filas, sin nada útil"),
            Caja(700, 270, 520, 230, "**Rellenar**", VERDE, 40, subtitulo="`fillna(valor)`\nla mediana, «UNKNOWN»"),
            Caja(1290, 270, 520, 230, "**Dejar**", GRIS, 40, subtitulo="si la columna\nno se usa"),
        ),
        Paso(
            "En el laboratorio borramos los dos pingüinos sin medidas, y al sexo faltante le ponemos unknown. "
            "Y un detalle que causa muchos errores: pandas ignora los NaN al calcular, pero NumPy no. "
            "Un solo NaN contamina todo el promedio.",
            Terminal(110, 600, "pandas: 4201.754385964912\nNumPy:  nan", 32, ancho=900),
            Caja(1080, 600, 730, 160, "un solo `NaN` contamina\ntodo el resultado de NumPy", ROJO, 34),
        ),
    ),
    Escena(
        "Ejercicio 3: tratar los faltantes",
        Paso(
            "Tu turno. Cuenta los faltantes de cada columna y las filas completas. Después borra las filas sin medidas "
            "numéricas, y rellena el sexo faltante con unknown. Pausa el video.",
            Codigo(110, 280, "missing = None     # TODO: faltantes por columna\n"
                   "n_complete = None  # TODO: filas sin ningún faltante\n"
                   "df_clean = None    # TODO: sin las filas a las que les falta una medida numérica\n"
                   "sex_filled = None  # TODO: la columna sex de df_clean, con \"UNKNOWN\"", 30),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------- estadística
    Seccion("Parte 4 · Estadística descriptiva"),
    Escena(
        "Media y mediana",
        Paso(
            "Tres números resumen una variable. El primero es la media. Piensa en un balancín: la media es el punto "
            "donde los datos quedan en equilibrio. Con estos cinco pesos, la suma es dieciocho mil quinientos, "
            "y dividida entre cinco da tres mil setecientos.",
            Matriz(110, 300, 1, 5, 200, AZUL, valores=[[3300, 3400, 3600, 3800, 4400]], tam=36,
                   etiqueta="cinco pingüinos (g)"),
            Texto(1180, 300, "**media** = 18 500 / 5", 42, acento=AZUL),
            Texto(1180, 380, "= **3700 g**", 42, acento=AZUL),
        ),
        Paso(
            "El segundo es la mediana: ordena los datos y quédate con el del medio, como la persona del centro "
            "en una fila ordenada por estatura. Aquí es tres mil seiscientos. La mitad de los datos queda por debajo.",
            Marco(510, 300, 200, 200, VERDE),
            Texto(1180, 470, "**mediana** = el del medio", 42, acento=VERDE),
            Texto(1180, 550, "= **3600 g**", 42, acento=VERDE),
        ),
    ),
    Escena(
        "Un dato malo",
        Paso(
            "¿Para qué dos números? Imagina cinco amigos en una reunión, y llega un millonario: el ingreso promedio "
            "se dispara, pero el de la persona del medio no cambia. Lo mismo pasa si alguien anota mal un peso, "
            "con dos ceros de más.",
            Matriz(110, 300, 1, 5, 200, AZUL, valores=[[3300, 3400, 3600, 3800, 440000]], tam=34,
                   etiqueta="el último peso, con dos ceros de más"),
            Marco(910, 300, 200, 200, ROJO),
            Texto(1180, 300, "media = **90 820 g**", 42, acento=ROJO),
            Texto(1180, 380, "mediana = **3600 g**", 42, acento=VERDE),
        ),
        Paso(
            "En el laboratorio lo pruebas con los trescientos cuarenta y dos pingüinos: un solo error mueve la media "
            "unos mil cien gramos, y la mediana ni se entera. La mediana es robusta a los valores atípicos.",
            Terminal(110, 650, "datos buenos: media    4202 g · mediana  4050 g\n"
                     "con el error: media    5302 g · mediana  4050 g", 32, ancho=1700),
        ),
    ),
    Escena(
        "Desviación estándar: cuánto se alejan",
        Paso(
            "El tercer número mide cuánto se alejan los datos de la media. Dos cursos con promedio siete: en uno todos "
            "sacaron siete; en el otro, la mitad cuatro y la mitad diez. Mismo promedio, historias muy distintas.",
            Matriz(110, 300, 1, 6, 110, AZUL, valores=[[7] * 6], tam=36, etiqueta="curso A: desviación **0**"),
            Matriz(960, 300, 1, 6, 110, NARANJA, valores=[[4, 10, 4, 10, 4, 10]], tam=36,
                   etiqueta="curso B: desviación **3,3**"),
        ),
        Paso(
            "Para medirlo, resta la media a cada dato. En el curso B, cada nota queda a tres puntos del siete: "
            "menos tres, o más tres.",
            Matriz(960, 500, 1, 6, 110, ROJO, valores=[[-3, 3, -3, 3, -3, 3]], tam=36, etiqueta="x − x̄"),
        ),
        Paso(
            "Eleva al cuadrado, para que los negativos no cancelen a los positivos: seis veces nueve, cincuenta y cuatro. "
            "Divide entre n menos uno, que es cinco, y saca la raíz: unos tres coma tres. Esa es la desviación estándar, "
            "y queda en las mismas unidades que los datos.",
            Texto(110, 520, "(9 · 6) / 5 = 10,8  →  √10,8 ≈ **3,3**", 40, acento=ROJO),
            Texto(110, 640, "s = √( Σ (xᵢ − x̄)² / (n − 1) )", 48),
        ),
        Paso(
            "Un detalle: pandas divide entre n menos uno, y NumPy entre n, salvo que le pidas otra cosa con el "
            "argumento ddof. Con muchos datos casi no importa, pero una comparación exacta va a fallar.",
            Texto(960, 780, "pandas: n − 1   ·   NumPy: n   (salvo `ddof=1`)", 40, NARANJA, NARANJA, alinear="centro"),
            Texto(960, 870, "`describe()` da media, desviación, mínimo, cuartiles y máximo de una vez", 34, GRIS,
                  alinear="centro"),
        ),
    ),
    Escena(
        "Ejercicio 4: estadística a mano",
        Paso(
            "Tu turno. Escribe la media y la desviación estándar con NumPy, usando solo sumas, len y la raíz cuadrada. "
            "La celda de revisión las compara con pandas. Después calcula la media y la mediana de los pesos con el error. "
            "Pausa el video.",
            Codigo(110, 280, "def my_mean(x):\n    # TODO: solo sumas y len\n\n"
                   "def my_std(x):\n    # TODO: divide entre n - 1\n\n"
                   "mean_bad = None    # TODO\nmedian_bad = None  # TODO", 32),
            pausa=1.2,
        ),
    ),
    # ----------------------------------------------------------------- groupby
    Seccion("Parte 5 · Comparar grupos"),
    Escena(
        "groupby: separar, aplicar, juntar",
        Paso(
            "¿Los Gentoo pesan más que los Adelie? Para responder, piensa en la ropa sucia: primero la separas "
            "en montones por color. Group by separa las filas según el valor de una columna, aquí la especie.",
            Caja(110, 280, 380, 420, "**la tabla**\n342 filas", GRIS, 38),
            Flecha(500, 360, 680, 300, GRIS),
            Flecha(500, 490, 680, 490, GRIS, etiqueta="separar", tam=28),
            Flecha(500, 620, 680, 680, GRIS),
            Caja(690, 240, 360, 130, "Adelie", AZUL, 36, subtitulo="151 filas"),
            Caja(690, 425, 360, 130, "Chinstrap", NARANJA, 36, subtitulo="68 filas"),
            Caja(690, 610, 360, 130, "Gentoo", VERDE, 36, subtitulo="123 filas"),
        ),
        Paso(
            "Luego aplica una función a cada montón, como pesar cada uno: aquí, el promedio de la masa. "
            "Y por último junta los resultados en una tabla nueva, con una fila por especie.",
            Flecha(1060, 305, 1240, 305, GRIS),
            Flecha(1060, 490, 1240, 490, GRIS, etiqueta="`mean`", tam=28),
            Flecha(1060, 675, 1240, 675, GRIS),
            Caja(1250, 240, 300, 130, "**3701 g**", AZUL, 36),
            Caja(1250, 425, 300, 130, "**3733 g**", NARANJA, 36),
            Caja(1250, 610, 300, 130, "**5076 g**", VERDE, 36),
            Codigo(110, 830, 'penguins.groupby("species")["body_mass_g"].mean()', 32),
        ),
    ),
    Escena(
        "Grupos cruzados",
        Paso(
            "Con dos columnas en group by salen los grupos cruzados: especie y sexo. La función unstack pasa el sexo "
            "a columnas, y la tabla se lee de un vistazo. En las tres especies, los machos pesan más.",
            Codigo(110, 260, 'penguins.groupby(["species", "sex"])["body_mass_g"].mean().unstack()', 30),
            Tabla(110, 420, [
                ["species", "FEMALE", "MALE"],
                ["Adelie", "3369", "4043"],
                ["Chinstrap", "3527", "3939"],
                ["Gentoo", "4680", "5485"],
            ], [360, 300, 300], tam=34),
        ),
    ),
    Escena(
        "Ejercicio 5: grupos",
        Paso(
            "Tu turno. Calcula la aleta media de cada especie y quédate con la más larga. Cuenta cuántos pingüinos "
            "de cada especie viven en cada isla, y encuentra la isla donde solo hay Adelie. Pausa el video.",
            Codigo(110, 280, "flipper_by_species = None  # TODO: una Series, una fila por especie\n"
                   "longest_species = None     # TODO: pista, idxmax()\n"
                   "counts = None              # TODO: islas × especies, con 0 donde no hay\n"
                   "only_adelie_island = None  # TODO: mira counts", 32),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- histogramas
    Seccion("Parte 6 · Histogramas"),
    Escena(
        "Un histograma cuenta por intervalos",
        Paso(
            "Un número resume, pero un gráfico muestra la forma de los datos. Un histograma es como repartir a los "
            "pingüinos en cajones según su peso: de tres mil a tres mil doscientos gramos, a este cajón; "
            "de tres mil doscientos a tres mil cuatrocientos, al siguiente. La altura de cada barra es cuántos cayeron ahí.",
            histograma_masa(False, id="h"),
        ),
        Paso(
            "Mira: el histograma tiene dos jorobas. Algo pasa aquí.",
            Marco(355, 280, 560, 500, NARANJA, id="m1"),
            Marco(1080, 400, 420, 380, NARANJA, id="m2"),
        ),
        Paso(
            "Al separar por especie se entiende: los Gentoo son mucho más pesados, y los Adelie y los Chinstrap casi "
            "se superponen. El promedio de todos, cuatro mil doscientos gramos, describe un pingüino que casi no existe.",
            histograma_masa(True),
            ocultar=["h", "m1", "m2"],
        ),
        Paso(
            "Una regla para todo el curso: cada gráfico lleva nombre en los ejes, con unidades, y una leyenda si hay "
            "más de un grupo. Un gráfico sin unidades no se puede leer.",
            Caja(1690, 300, 200, 420, "ejes\n+\nunidades\n+\nleyenda", NARANJA, 30),
        ),
    ),
    Escena(
        "Ejercicio 6: tu histograma",
        Paso(
            "Tu turno. Haz el mismo gráfico, pero con el largo del pico: un histograma por especie, con ejes, unidades, "
            "título y leyenda. Después, mira el gráfico y responde: ¿qué especie se separa de las otras por el pico? "
            "Pausa el video.",
            Codigo(110, 280, "fig, ax = plt.subplots(figsize=(6, 3.5))\n"
                   "# TODO: un ax.hist(...) por especie, con label=...\n"
                   "# TODO: ax.set_xlabel, ax.set_ylabel, ax.set_title, ax.legend\n"
                   "plt.show()", 32),
            pausa=1.2,
        ),
    ),
    # --------------------------------------------------------------- correlación
    Seccion("Parte 7 · Dispersión y correlación"),
    Escena(
        "Un punto por pingüino",
        Paso(
            "Un diagrama de dispersión pone un punto por pingüino: el largo del pico en un eje, el alto del pico en el "
            "otro. Sirve para ver si dos variables se mueven juntas, como dos amigos que caminan juntos: "
            "cuando uno avanza, el otro también.",
            picos(230, 1200, False, id="d"),
        ),
        Paso(
            "La correlación mide esa tendencia con un número entre menos uno y uno. Traza la media de cada variable: "
            "los puntos arriba a la derecha, o abajo a la izquierda, suman. Los de las otras dos esquinas, restan.",
            Dispersion(230, 290, 1200, 480,
                       [(np.concatenate([DATOS[e]["pico"] for e in ESPECIES]),
                         np.concatenate([DATOS[e]["alto"] for e in ESPECIES]))],
                       rx=(31, 61), ry=(12.5, 24.5), marcas_x=range(35, 61, 5), marcas_y=range(14, 22, 2),
                       etiqueta_x="largo del pico (mm)", etiqueta_y="alto del pico (mm)", cuadrantes=True, radio=5),
            ocultar=["d"],
        ),
        Paso(
            "Cerca de uno, suben juntas. Cerca de menos uno, cuando una sube la otra baja. Cerca de cero, "
            "no hay relación en línea recta.",
            Texto(1550, 330, "**r ≈ 1**\nsuben juntas", 32, VERDE, VERDE),
            Texto(1550, 480, "**r ≈ −1**\nuna sube,\nla otra baja", 32, ROJO, ROJO),
            Texto(1550, 680, "**r ≈ 0**\nsin relación\nlineal", 32, GRIS),
        ),
    ),
    Escena(
        "Correlación no es causalidad",
        Paso(
            "Cuidado con una trampa. En verano se venden más helados y hay más quemaduras de sol. "
            "Las dos cosas están correlacionadas, pero los helados no queman la piel: las dos las causa el calor.",
            Caja(110, 450, 480, 170, "helados vendidos", AZUL, 38),
            Caja(1330, 450, 480, 170, "quemaduras de sol", AZUL, 38),
            Caja(720, 260, 480, 150, "**el calor del verano**", NARANJA, 38),
            Flecha(860, 420, 500, 440, NARANJA),
            Flecha(1060, 420, 1420, 440, NARANJA),
            Flecha(600, 535, 1320, 535, GRIS, etiqueta="correlación, no causa", tam=30, lado="abajo"),
        ),
        Paso(
            "Que dos variables se muevan juntas no dice cuál causa a cuál, ni si las dos dependen de una tercera.",
            Caja(110, 780, 1700, 130, "**Correlación no es causalidad.**", ROJO, 42),
        ),
    ),
    Escena(
        "Ejercicio 7: correlación a mano",
        Paso(
            "Tu turno. Escribe la correlación con NumPy: primero resta las medias. Luego calcúlala entre el largo y el "
            "alto del pico, primero con todos los pingüinos, y después dentro de cada especie. Compara los signos. "
            "Pausa el video.",
            Codigo(110, 260, "def my_corr(x, y):\n    # TODO: la fórmula de Pearson, restando primero las medias\n\n"
                   "r_all = None         # TODO: todos los pingüinos\n"
                   "r_by_species = None  # TODO: {\"Adelie\": ..., \"Chinstrap\": ..., \"Gentoo\": ...}", 32),
            Texto(110, 640, "r = Σ (xᵢ − x̄)(yᵢ − ȳ)  /  ( √Σ (xᵢ − x̄)²  ·  √Σ (yᵢ − ȳ)² )", 40),
            pausa=1.2,
        ),
    ),
    Escena(
        "La sorpresa: la paradoja de Simpson",
        Paso(
            "Cuando lo resuelvas, vas a ver algo raro. Con todos los pingüinos juntos, la tendencia baja: "
            "parece que un pico más largo es más bajo.",
            picos(200, 680, False, tendencia=True),
        ),
        Paso(
            "Pero separados por especie, en las tres la tendencia sube: un pico más largo también es más alto. "
            "Se invierte porque los Gentoo tienen picos largos y bajos, y al mezclarlos inclinan la nube entera.",
            picos(1110, 680, True, tendencia=True),
        ),
        Paso(
            "Esto se llama paradoja de Simpson. Y deja una lección para todo el año: antes de confiar en un número "
            "de toda la tabla, míralo por grupos.",
            Caja(110, 895, 1700, 85, "Antes de confiar en un número de toda la tabla, **míralo por grupos**.", MORADO, 36),
        ),
    ),
    # ------------------------------------------------------------------- regla
    Seccion("Parte 8 · De un gráfico a una regla"),
    Escena(
        "Primero, la línea base",
        Paso(
            "Los gráficos sugieren un clasificador sin entrenar nada. Pero antes, la línea base: predecir siempre "
            "la especie más común. Acierta el cuarenta y cuatro por ciento. Cualquier regla que valga la pena "
            "tiene que superarla con holgura.",
            Terminal(110, 280, "línea base (siempre Adelie): accuracy 0.442", 34, ancho=1700),
            Texto(110, 440, "accuracy = la fracción de aciertos", 36, GRIS),
        ),
    ),
    Escena(
        "Ejercicio 8: una regla de dos preguntas",
        Paso(
            "La regla funciona como el juego Adivina quién: preguntas de sí o no. Primero: ¿la aleta mide más de "
            "cierto valor? Si sí, es un Gentoo. Si no, segunda pregunta: ¿el pico mide más de cierto valor? "
            "Si sí, Chinstrap; si no, Adelie.",
            Caja(110, 300, 520, 150, "¿aleta > `FLIPPER_CUT`?", MORADO, 34),
            Flecha(630, 375, 820, 375, VERDE, etiqueta="sí", tam=30),
            Caja(830, 300, 380, 150, "**Gentoo**", VERDE, 40),
            Flecha(370, 460, 370, 560, ROJO, etiqueta="no", tam=30, lado="derecha"),
            Caja(110, 570, 520, 150, "¿pico > `BILL_CUT`?", MORADO, 34),
            Flecha(630, 645, 820, 645, VERDE, etiqueta="sí", tam=30),
            Caja(830, 570, 380, 150, "**Chinstrap**", NARANJA, 40),
            Flecha(370, 730, 370, 830, ROJO, etiqueta="no", tam=30, lado="derecha"),
            Caja(110, 840, 520, 120, "**Adelie**", AZUL, 40),
        ),
        Paso(
            "Tu turno: elige los dos umbrales mirando los histogramas, y escribe la regla con np where. "
            "La meta es superar el noventa por ciento. Pausa el video.",
            Codigo(1290, 300, "FLIPPER_CUT = None  # TODO\n"
                   "BILL_CUT = None     # TODO\n\n"
                   "def predict_rule(data):\n"
                   "    # TODO: np.where(...)", 28, ancho=540),
            pausa=1.2,
        ),
        Paso(
            "Un aviso: elegiste los umbrales mirando los mismos datos con los que mides. Con pingüinos nuevos, "
            "probablemente acertarías un poco menos. Medir bien es el tema de la semana cinco.",
            Caja(1290, 640, 540, 320, "Elegiste los umbrales con **los mismos datos** con que mides.",
                 ROJO, 34, subtitulo="medir bien: semana 5"),
        ),
    ),
    # ----------------------------------------------------- estructurados vs no
    Seccion("Parte 9 · Datos estructurados vs no estructurados"),
    Escena(
        "Todo termina siendo un arreglo",
        Paso(
            "Los pingüinos son datos estructurados: una tabla en la que cada columna significa algo, como el largo del "
            "pico. Las imágenes y los textos son no estructurados: ningún número suelto significa nada por sí solo. "
            "Pero para un modelo, todo termina siendo un arreglo.",
            Caja(110, 260, 520, 150, "**tabla**", AZUL, 40, subtitulo="cada columna significa algo"),
            Caja(700, 260, 520, 150, "**imagen**", VERDE, 40, subtitulo="píxeles"),
            Caja(1290, 260, 520, 150, "**texto**", MORADO, 40, subtitulo="palabras"),
        ),
        Paso(
            "Una imagen ya la viste en la semana dos: un dígito de ocho por ocho es una matriz, y aplanado es una fila "
            "de sesenta y cuatro números. Mil imágenes son una tabla de sesenta y cuatro columnas, una por píxel.",
            Pixeles(780, 460, PRIMERA, 44),
            Flecha(970, 820, 970, 880, VERDE),
            Texto(960, 900, "una fila: `px_0 … px_63`", 32, acento=VERDE, alinear="centro"),
        ),
        Paso(
            "Un texto se puede convertir así: vacías sus palabras en una bolsa y cuentas cuántas hay de cada una del "
            "vocabulario. Se pierde el orden, pero cada frase queda como un vector del mismo largo. "
            "Se llama bolsa de palabras.",
            Texto(1290, 460, "«el pez nada»", 36, MORADO),
            Matriz(1290, 560, 1, 7, 70, MORADO, valores=[[1, 0, 0, 1, 0, 0, 1]], tam=28),
            Texto(1290, 650, "el · pingüino · pesa · nada\nmucho · come · pez", 26, GRIS),
            Texto(1290, 760, "**bolsa de palabras**", 32, MORADO, MORADO),
        ),
    ),
    Escena(
        "Ejercicio 9: un texto como vector",
        Paso(
            "Último ejercicio. Pasa el texto a minúsculas, sepáralo en palabras, y cuenta cuántas veces aparece cada "
            "palabra del vocabulario. Las palabras que no están en el vocabulario se ignoran. Pausa el video.",
            Codigo(110, 280, 'vocab = ["el", "pingüino", "pesa", "nada", "mucho", "come", "pez"]\n\n'
                   "def text_to_counts(text, vocab):\n"
                   "    # TODO: un arreglo de largo len(vocab) con los conteos", 32),
            pausa=1.2,
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
                    "Repasemos. Con una tabla nueva, primero read csv, la forma y los tipos de cada columna.",
                    "Para filtrar, máscaras como en NumPy, con ampersand, barra y paréntesis.",
                    "Los datos faltantes se cuentan con is na, y se borran o se rellenan. NumPy no ignora un NaN.",
                    "La media, la mediana y la desviación estándar resumen una variable. La mediana resiste los errores.",
                    "Group by separa, aplica y junta: un número de toda la tabla puede esconder grupos muy distintos.",
                    "Los gráficos llevan ejes, unidades y leyenda. La correlación no es causa, y puede cambiar de signo "
                    "por grupos.",
                    "Y todo modelo se compara con una línea base. Tabla, imagen o texto: al final, un arreglo.",
                ],
                vinetas(110, 250, [
                    "**Cargar:** `read_csv`, `shape`, tipos numéricos y categóricos.",
                    "**Filtrar:** máscaras con `&`, `|` y paréntesis; `value_counts()`.",
                    "**Faltantes:** `isna().sum()`, `dropna`, `fillna`. NumPy no ignora `NaN`.",
                    "**Resumir:** media, mediana (robusta), desviación estándar.",
                    "**Grupos:** `groupby` separa, aplica y junta.",
                    "**Gráficos:** ejes, unidades, leyenda. Correlación ≠ causa; Simpson.",
                    "**Línea base** primero. Tabla, imagen o texto: un arreglo.",
                ], tam=36, sep=104, ancho=1650),
            )
        ],
    ),
    Escena(
        "Para la semana 4",
        Paso(
            "Para la próxima clase, haz la lectura previa de MÓLÓ, en inglés: cálculo básico, funciones y parámetros, "
            "y probabilidad.",
            Vineta(110, 260, "**Lectura previa** (MÓLÓ): Calculus basics for ML, Functions and Parameters y "
                   "Probability and Distributions.", 40, NARANJA, numero=1, ancho=1600),
        ),
        Paso(
            "No olvides la cuota semanal en la plataforma.",
            Vineta(110, 440, "**Cuota semanal** en ioai.artix.tech.", 40, NARANJA, numero=2, ancho=1600),
        ),
        Paso(
            "Y si no terminaste los ejercicios de hoy, termínalos: el Sprint cero, en las semanas cinco y seis, "
            "es limpiar y analizar un CSV desordenado con estas mismas herramientas.",
            Vineta(110, 560, "**Termina los ejercicios:** el Sprint 0 (semanas 5 y 6) es limpiar y analizar un CSV "
                   "desordenado.", 40, NARANJA, numero=3, ancho=1600),
            Texto(110, 760, "Los enlaces están al final de `sem03_pandas.ipynb`.", 36, GRIS),
        ),
        Paso(
            "La semana cuatro es cálculo y probabilidad: derivadas, gradiente y descenso de gradiente a mano. "
            "¡Nos vemos ahí!",
            Texto(960, 880, "**¡Nos vemos en la semana 4!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
