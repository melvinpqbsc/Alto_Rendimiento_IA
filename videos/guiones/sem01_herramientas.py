"""Video de la semana 1: Nivel_Intermedio/clases/sem01_herramientas.ipynb."""

from motor import (AZUL, GRIS, GRIS_CLARO, MORADO, NARANJA, ROJO, TINTA, VERDE, Caja, Circulo, Codigo, Escena,
                   Seccion,
                   Flecha, Grafica, Linea, Marco, Matriz, Paso, Tabla, Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 1 · Las herramientas del curso · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}


def red(x0, y0, entradas=3, salidas=2, sep=150, dx=420):
    """Diagrama de una capa lineal: círculos de entrada, de salida y una línea por peso."""
    ys_in = [y0 + i * sep for i in range(entradas)]
    alto = (entradas - 1) * sep
    ys_out = [y0 + alto / 2 + (j - (salidas - 1) / 2) * sep for j in range(salidas)]
    elementos = [Linea(x0, yi, x0 + dx, yo, color="#B9C7E8", grosor=4) for yi in ys_in for yo in ys_out]
    elementos += [Circulo(x0, y, 46, f"x{'₁₂₃₄₅'[i]}", AZUL) for i, y in enumerate(ys_in)]
    elementos += [Circulo(x0 + dx, y, 46, f"y{'₁₂₃₄₅'[j]}", VERDE) for j, y in enumerate(ys_out)]
    return elementos


ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Te damos la bienvenida al Programa de Alto Rendimiento en Inteligencia Artificial. "
            "Este video resume la primera clase del nivel intermedio: las herramientas que vamos a usar todo el año.",
            Texto(960, 250, "SEMANA 1 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**Las herramientas del curso**", 92, alinear="centro"),
            Texto(960, 460, "Git, GitHub y Python orientado a objetos", 50, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Son tres ideas. Primero, qué es la olimpiada y cómo funciona el curso. "
            "Segundo, Git y GitHub: así vas a guardar y entregar tus trabajos. "
            "Y tercero, las clases de Python, que son la base de cómo se escriben las redes neuronales en PyTorch.",
            Caja(150, 640, 500, 200, "**1**\nEl curso y la IOAI", AZUL, 38),
            Caja(710, 640, 500, 200, "**2**\nGit y GitHub", VERDE, 38),
            Caja(1270, 640, 500, 200, "**3**\nClases en Python", MORADO, 38),
        ),
    ),
    # ---------------------------------------------------------------- la IOAI
    Seccion("Parte 1 · El curso"),
    Escena(
        "¿Qué es la IOAI?",
        Paso(
            "La IOAI es la Olimpiada Internacional de Inteligencia Artificial, una competencia para estudiantes de colegio. "
            "Su programa cubre machine learning clásico, redes neuronales, visión por computadora y lenguaje.",
            Texto(130, 250, "**I**nternational **O**lympiad in **A**rtificial **I**ntelligence: "
                  "una competencia de IA para estudiantes de colegio.", 42, ancho=1650),
        ),
        Paso(
            "Un problema de la competencia se ve así. Te dan datos y una métrica, todo en inglés. "
            "Tú entrenas un modelo y haces predicciones. Esas predicciones se comparan con un conjunto de prueba oculto, "
            "que tú nunca ves, y de ahí sale tu puntaje.",
            Caja(130, 470, 330, 170, "**Datos**", AZUL, 40, subtitulo="en inglés"),
            Flecha(470, 555, 560, 555),
            Caja(570, 470, 330, 170, "**Tu modelo**", MORADO, 40),
            Flecha(910, 555, 1000, 555),
            Caja(1010, 470, 330, 170, "**Predicciones**", VERDE, 40),
            Flecha(1350, 555, 1440, 555),
            Caja(1450, 470, 340, 170, "**Puntaje**", NARANJA, 40, subtitulo="contra la prueba oculta"),
        ),
        Paso(
            "Cada día de competencia dura seis horas, en computadoras con GPU, "
            "los procesadores que aceleran el entrenamiento de redes neuronales.",
            Vineta(150, 760, "**6 horas** por día de competencia, en computadoras con **GPU**", 40, NARANJA),
        ),
    ),
    # ---------------------------------------------------------- cinco bloques
    Escena(
        "El año en cinco bloques",
        Paso(
            "El curso tiene cinco bloques. El primero es el puente, de la semana uno a la cinco: "
            "repasamos las herramientas que el deep learning da por sabidas, como NumPy, cálculo, probabilidad "
            "y machine learning clásico.",
            Caja(100, 330, 310, 260, "**Puente**", AZUL, 38, subtitulo="semanas 1–5\nNumPy, cálculo, ML clásico"),
            Marco(80, 310, 350, 300, AZUL, id="marco"),
        ),
        Paso(
            "Después vienen las redes neuronales con PyTorch, la visión por computadora, "
            "y el lenguaje, el audio y los modelos multimodales, que combinan texto e imágenes.",
            Flecha(415, 460, 455, 460),
            Caja(460, 330, 310, 260, "**Redes neuronales**", VERDE, 38, subtitulo="y PyTorch"),
            Flecha(775, 460, 815, 460),
            Caja(820, 330, 310, 260, "**Visión**", MORADO, 38, subtitulo="por computadora"),
            Flecha(1135, 460, 1175, 460),
            Caja(1180, 330, 310, 260, "**Lenguaje**", ROJO, 38, subtitulo="audio y multimodal"),
            ocultar=["marco"],
        ),
        Paso(
            "Y al final, la preparación para la competencia: problemas de años anteriores, "
            "un proyecto y un simulacro de seis horas.",
            Flecha(1495, 460, 1535, 460),
            Caja(1540, 330, 290, 260, "**Competencia**", NARANJA, 38, subtitulo="problemas pasados\nsimulacro de 6 h"),
            Texto(960, 700, "Del repaso de NumPy en octubre… a resolver problemas reales de la IOAI en julio.",
                  38, GRIS, alinear="centro"),
        ),
    ),
    # --------------------------------------------------------- cada semana
    Escena(
        "Cómo funciona cada semana",
        *[
            Paso(decir, el)
            for decir, el in zip(
                [
                    "Cada clase dura dos horas y empieza con quince minutos fijos: la tabla de posiciones de la plataforma, "
                    "el bug de la semana, y un resumen de la clase anterior que presenta un estudiante.",
                    "Fuera de clase vas a trabajar entre cuatro y cinco horas por semana: la cuota de la plataforma, "
                    "la lectura previa y los trabajos prácticos.",
                    "Tus trabajos se entregan en tu repositorio privado de GitHub. Al inicio de la clase indicada, "
                    "el profesor copia todos los repositorios. Lo que esté en el tuyo en ese momento es tu entrega.",
                    "Y una regla importante: puedes usar inteligencia artificial para entender un error o un concepto, "
                    "pero no para que escriba tus soluciones. Las reglas completas están en la Guía del estudiante.",
                ],
                vinetas(
                    150, 270,
                    [
                        "**Clase de 2 horas.** Empieza con 15 minutos fijos: tabla de posiciones, "
                        "el «bug de la semana» y un resumen hecho por un estudiante.",
                        "**4–5 horas fuera de clase:** la plataforma ioai.artix.tech, la lectura previa y los trabajos.",
                        "**Tu entrega es lo que hay en tu repositorio** al inicio de la clase: "
                        "en ese momento el profesor copia todos los repositorios.",
                        "**IA sí, para entender** errores y conceptos. **No** para que escriba tus soluciones.",
                    ],
                    tam=40, sep=170, ancho=1550, numeradas=True,
                ),
            )
        ],
    ),
    # ------------------------------------------------------- lista de inicio
    Escena(
        "Tu lista de inicio",
        Paso(
            "Antes de seguir, revisa esta lista. Crea tu cuenta en la plataforma, inicia sesión en GitHub "
            "y, si todavía no lo hiciste, envíale tu nombre de usuario al profesor.",
            *vinetas(150, 260, ["Crea tu cuenta en **ioai.artix.tech** y lee su guía.",
                                "Inicia sesión en **github.com** y envía tu usuario al profesor."],
                     tam=40, sep=120, color=VERDE),
        ),
        Paso(
            "Luego acepta la invitación a tu repositorio privado de entregas. Ojo: la invitación caduca a los siete días.",
            Vineta(150, 500, "Acepta la invitación a `entregas-<tu-usuario>`: **caduca a los 7 días**.", 40, VERDE),
        ),
        Paso(
            "En Colab, conecta tu cuenta de GitHub y marca la opción de incluir repositorios privados. "
            "Por último, ejecuta completo el notebook verificar entorno. Si algo falla, anótalo y pregunta en clase.",
            Vineta(150, 620, "En Colab: Archivo → Abrir notebook → GitHub, y marca **Incluir repositorios privados**.",
                   40, VERDE),
            Vineta(150, 740, "Ejecuta completo `verificar_entorno.ipynb` y revisa el resumen final.", 40, VERDE),
        ),
    ),
    # ------------------------------------------------------------------- Git
    Seccion("Parte 2 · Git y GitHub"),
    Escena(
        "Git: fotos de tu proyecto",
        Paso(
            "Ahora, Git. Imagina que cada vez que avanzas en un trabajo pudieras tomarle una foto, "
            "con una nota que diga qué cambiaste. Eso es Git: un programa que guarda fotos de tu proyecto "
            "a lo largo del tiempo.",
            Texto(130, 250, "**Git** guarda fotos de tu proyecto a lo largo del tiempo.", 44),
            Linea(250, 640, 1500, 640, GRIS_CLARO, 10),
            Circulo(350, 640, 60, "1", VERDE),
            Texto(350, 730, "Agregar notas\nde la semana 1", 30, alinear="centro"),
        ),
        Paso(
            "Cada foto se llama commit, y tiene autor, fecha y un mensaje. Las fotos viejas nunca se borran, "
            "así que puedes volver a cualquier versión anterior. Es como un punto de guardado en un videojuego.",
            Circulo(750, 640, 60, "2", VERDE),
            Texto(750, 730, "Agregar\nla semana 2", 30, alinear="centro"),
            Circulo(1150, 640, 60, "3", VERDE),
            Texto(1150, 730, "Corregir\nun error", 30, alinear="centro"),
            Texto(1560, 615, "**ahora**", 34, VERDE),
            Texto(130, 340, "Cada foto es un **commit**: autor, fecha y un mensaje que dice qué cambió.", 40, ancho=1650),
        ),
        Paso(
            "Por otro lado, GitHub es un sitio web que guarda una copia de esas fotos en línea. "
            "Así tu trabajo no depende de tu computadora, y el profesor lo puede ver.",
            Flecha(1620, 665, 1620, 830, MORADO, 5),
            Caja(1300, 840, 520, 120, "**GitHub**: la copia en línea", MORADO, 36),
        ),
    ),
    # ----------------------------------------------------------- vocabulario
    Escena(
        "El vocabulario mínimo",
        Paso(
            "Cinco palabras. Un repositorio es una carpeta cuyo historial controla Git. "
            "Un commit es una foto del repositorio. Y el historial, o log, es la lista de commits, "
            "del más nuevo al más viejo.",
            Tabla(150, 260, [
                ["Término", "Qué es"],
                ["repositorio (repository)", "Una carpeta cuyo historial controla Git"],
                ["commit", "Una foto del repositorio, con autor, fecha y mensaje"],
                ["historial (log)", "La lista de commits, del más nuevo al más viejo"],
            ], [560, 1060], tam=38, id="tabla"),
        ),
        Paso(
            "Un diff muestra qué líneas cambiaron entre dos commits. Y el remoto es la copia del repositorio "
            "en otro lugar. Para nosotros, el remoto es GitHub.",
            Tabla(150, 260, [
                ["Término", "Qué es"],
                ["repositorio (repository)", "Una carpeta cuyo historial controla Git"],
                ["commit", "Una foto del repositorio, con autor, fecha y mensaje"],
                ["historial (log)", "La lista de commits, del más nuevo al más viejo"],
                ["diferencia (diff)", "Qué líneas cambiaron entre dos commits"],
                ["remoto (remote)", "La copia en otro lugar: para nosotros, GitHub"],
            ], [560, 1060], tam=38),
            ocultar=["tabla"],
        ),
    ),
    # ----------------------------------------------------------- Git en vivo
    Escena(
        "Git en vivo",
        Paso(
            "Veamos Git en acción. Con git init convertimos una carpeta vacía en un repositorio.",
            Terminal(100, 240, "$ git init -b main\nInitialized empty Git repository", 28, ancho=1000, id="t"),
            Caja(1200, 260, 600, 130, "**Tu carpeta**", AZUL, 36, subtitulo="lo que ves en tu computadora"),
        ),
        Paso(
            "Creamos un archivo, notas punto md. Git lo ve, pero todavía no lo guarda: "
            "los dos signos de pregunta significan untracked, sin seguimiento.",
            Terminal(100, 240, "$ git init -b main\nInitialized empty Git repository\n"
                     "$ git status --short\n?? notas.md", 28, ancho=1000, id="t"),
            Texto(1500, 410, "`notas.md` ??", 32, alinear="centro", acento=NARANJA),
            ocultar=["t"],
        ),
        Paso(
            "Guardar una foto tiene dos pasos. Primero, add elige qué archivos entran en la foto.",
            Terminal(100, 240, "$ git init -b main\nInitialized empty Git repository\n"
                     "$ git status --short\n?? notas.md\n$ git add notas.md", 28, ancho=1000, id="t"),
            Flecha(1500, 470, 1500, 545, VERDE, etiqueta="`add`", lado="derecha"),
            Caja(1200, 550, 600, 120, "**Lista para la foto**", VERDE, 36),
            ocultar=["t"],
        ),
        Paso(
            "Después, commit toma la foto, con un mensaje. git log muestra el historial: ya tiene un commit. "
            "El código del principio es su identificador, que se llama hash. Con él puedes volver a esa versión.",
            Terminal(100, 240, "$ git init -b main\nInitialized empty Git repository\n"
                     "$ git status --short\n?? notas.md\n$ git add notas.md\n"
                     '$ git commit -m "Agregar notas de la semana 1"\n'
                     "[main (root-commit) 20218ab] Agregar notas...\n$ git log --oneline\n"
                     "20218ab Agregar notas de la semana 1", 28, ancho=1000, id="t"),
            Flecha(1500, 675, 1500, 745, MORADO, etiqueta="`commit`", lado="derecha"),
            Caja(1200, 750, 600, 190, "**Historial**", MORADO, 36, subtitulo="20218ab  Agregar notas…", id="hist"),
            ocultar=["t"],
        ),
        Paso(
            "Ahora agregamos una línea al archivo. git diff muestra qué cambió respecto de la última foto: "
            "en verde, con un signo más, la línea agregada.",
            Terminal(100, 240, "$ git diff\n--- a/notas.md\n+++ b/notas.md\n@@ -1,3 +1,4 @@\n # Mis notas\n\n"
                     " Semana 1: Git guarda fotos del proyecto.\n+Semana 2: NumPy y álgebra lineal.", 28, ancho=1000,
                     id="t"),
            ocultar=["t"],
        ),
        Paso(
            "Tomamos otra foto. Ahora el historial tiene dos commits, del más nuevo al más viejo, "
            "y podemos volver a cualquiera de los dos.",
            Terminal(100, 240, '$ git commit -am "Agregar la semana 2"\n[main 4ddc5b8] Agregar la semana 2\n'
                     "$ git log --oneline\n4ddc5b8 Agregar la semana 2\n20218ab Agregar notas de la semana 1",
                     28, ancho=1000, id="t"),
            Caja(1200, 750, 600, 190, "**Historial**", MORADO, 36,
                 subtitulo="4ddc5b8  Agregar la semana 2\n20218ab  Agregar notas…"),
            ocultar=["t", "hist"],
            pausa=0.8,
        ),
    ),
    # ------------------------------------------------------------ desde Colab
    Escena(
        "Lo mismo desde Colab",
        Paso(
            "En Colab no necesitas escribir comandos. El menú Archivo, Guardar una copia en GitHub, "
            "hace add, commit y además push, que sube el commit a GitHub. Todo en un solo paso. "
            "Cada vez que guardas así, se agrega un commit a tu repositorio de entregas.",
            Caja(130, 280, 480, 200, "**Colab**", NARANJA, 42, subtitulo="tu notebook"),
            Flecha(630, 380, 1270, 380, VERDE, 8, etiqueta="Archivo → **Guardar una copia en GitHub**", tam=32),
            Texto(950, 410, "`add` + `commit` + `push`", 34, alinear="centro", acento=VERDE),
            Caja(1290, 280, 500, 200, "**GitHub**", MORADO, 42, subtitulo="entregas-<tu-usuario>"),
        ),
        Paso(
            "Para los trabajos, la ruta siempre es el código del trabajo, una barra, y el nombre del archivo. "
            "La carpeta debe llamarse exactamente como el código, por ejemplo B1, P01 o Sprint1. "
            "Así el profesor encuentra tu entrega.",
            Texto(130, 580, "Ruta de cada trabajo:", 40),
            Texto(560, 572, "`B1/B1_ejercicios_numpy.ipynb`", 52, acento=AZUL),
            Flecha(640, 720, 640, 655, AZUL, 5),
            Texto(500, 730, "carpeta = el **código** del trabajo, exacto", 34, AZUL),
        ),
        Paso(
            "Tu turno: guarda el notebook de esta clase en la carpeta prueba de tu repositorio. "
            "Luego cambia algo y guarda otra vez, con un mensaje que diga qué cambiaste. "
            "En GitHub, abre el historial del archivo: vas a ver tu cambio en verde.",
            Caja(130, 840, 1660, 120, "**Tu turno 1:** guarda en `prueba/sem01_herramientas.ipynb`, "
                 "cambia algo, guarda otra vez y mira **History** en GitHub.", VERDE, 34),
        ),
    ),
    # ------------------------------------------------------------- objetos
    Seccion("Parte 3 · Python orientado a objetos"),
    Escena(
        "Ya usaste objetos",
        Paso(
            "Tercera parte: programación orientada a objetos. La idea es juntar en un mismo objeto los datos "
            "y las funciones que trabajan con ellos. Y ya usaste objetos: un modelo de scikit-learn es uno.",
            Codigo(130, 260, 'model = LinearRegression()   # crear el objeto\n'
                   'model.fit(X, y)              # método: cambia el estado\n'
                   'print(model.coef_)           # atributo: el estado\n'
                   'model.predict([[2.0]])       # método: usa el estado', 34, resaltar={1}, id="c"),
        ),
        Paso(
            "El método fit es una función del objeto. Al ejecutarla, el modelo aprende y cambia su estado.",
            Codigo(130, 260, 'model = LinearRegression()   # crear el objeto\n'
                   'model.fit(X, y)              # método: cambia el estado\n'
                   'print(model.coef_)           # atributo: el estado\n'
                   'model.predict([[2.0]])       # método: usa el estado', 34, resaltar={2}, id="c"),
            ocultar=["c"],
        ),
        Paso(
            "Lo que aprendió queda guardado en sus atributos, como coef, la pendiente de la recta. "
            "Y el método predict utiliza ese estado para predecir.",
            Codigo(130, 260, 'model = LinearRegression()   # crear el objeto\n'
                   'model.fit(X, y)              # método: cambia el estado\n'
                   'print(model.coef_)           # atributo: el estado\n'
                   'model.predict([[2.0]])       # método: usa el estado', 34, resaltar={3, 4}, id="c"),
            ocultar=["c"],
        ),
        Paso(
            "Esto importa porque en PyTorch, desde la semana ocho, cada modelo, cada capa y cada conjunto de datos "
            "es una clase que escribes tú. Hoy aprendemos a escribirlas.",
            Caja(130, 680, 1660, 160, "En **PyTorch**, cada modelo, cada capa y cada conjunto de datos "
                 "es **una clase que escribes tú**.", MORADO, 40),
        ),
    ),
    # --------------------------------------------------------- clase e instancia
    Escena(
        "Clase = molde · instancia = objeto",
        Paso(
            "Una clase es un molde, como un molde de galletas. Cada objeto que creas con ella es una instancia: "
            "una galleta distinta, hecha con el mismo molde.",
            Caja(1150, 250, 650, 150, "**clase** `TrainingLog`", MORADO, 40, subtitulo="el molde"),
        ),
        Paso(
            "Al crear una instancia, Python llama al método init. Ahí guardamos los datos iniciales en self. "
            "Self es la instancia misma: es lo que permite que cada objeto tenga sus propios datos.",
            Codigo(100, 250, """class TrainingLog:
    def __init__(self, name):
        self.name = name
        self.losses = []

    def add(self, loss):
        self.losses.append(loss)

    def best(self):
        return min(self.losses)""", 30, resaltar={2, 3, 4}, id="c"),
        ),
        Paso(
            "Por eso log a y log b, creados con la misma clase, tienen listas de pérdidas diferentes. "
            "Los métodos, como add y best, trabajan con los datos de la instancia que los llama.",
            Codigo(100, 250, """class TrainingLog:
    def __init__(self, name):
        self.name = name
        self.losses = []

    def add(self, loss):
        self.losses.append(loss)

    def best(self):
        return min(self.losses)""", 30, resaltar={6, 7, 9, 10}, id="c"),
            Flecha(1350, 410, 1300, 520, MORADO, 5),
            Flecha(1600, 410, 1650, 520, MORADO, 5),
            Caja(1060, 530, 400, 230, "`log_a`", AZUL, 34, subtitulo="«modelo A»\nlosses = [0.9, 0.5, 0.4, 0.45]"),
            Caja(1490, 530, 360, 230, "`log_b`", VERDE, 34, subtitulo="«modelo B»\nlosses = [1.2]"),
            Texto(1060, 800, "`log_a.best()` → 0.4", 34, acento=AZUL),
            ocultar=["c"],
        ),
    ),
    # ---------------------------------------------------- métodos especiales
    Escena(
        "Métodos especiales y herencia",
        Paso(
            "Los métodos con doble guion bajo, llamados dunder, conectan tu clase con la sintaxis de Python. "
            "Cuando escribes len de un objeto, Python llama a su método len. Si escribes el objeto con corchetes, "
            "llama a getitem. Y si llamas al objeto como si fuera una función, llama a call.",
            Tabla(100, 250, [
                ["Escribes", "Python llama a"],
                ["`TrainingLog(\"A\")`", "`__init__`"],
                ["`len(obj)`", "`__len__`"],
                ["`obj[i]`", "`__getitem__`"],
                ["`obj(x)`", "`__call__`"],
                ["`print(obj)`", "`__repr__`"],
            ], [430, 380], tam=34),
        ),
        Paso(
            "La herencia sirve para reutilizar. Si escribes class B, y entre paréntesis A, la clase B recibe todo "
            "lo de A, y puede agregar o cambiar métodos. Así PrettyTrainingLog hereda todo de TrainingLog "
            "y solo agrega len y repr.",
            Caja(1100, 250, 700, 200, "`TrainingLog`", MORADO, 38, subtitulo="name, losses, add, best"),
            Flecha(1450, 460, 1450, 580, MORADO, 6, etiqueta="hereda", lado="derecha"),
            Caja(1100, 590, 700, 200, "`PrettyTrainingLog`", AZUL, 38, subtitulo="todo lo anterior\n+ __len__ + __repr__"),
            Texto(1100, 830, "`class PrettyTrainingLog(TrainingLog):`", 30, acento=AZUL),
        ),
    ),
    # -------------------------------------------------------------- RunningMean
    Escena(
        "Tu turno 2: un promedio por lotes",
        Paso(
            "Un ejercicio para practicar. Al entrenar, procesamos los datos en lotes, o batches, "
            "y queremos la pérdida promedio de toda la época. Pero los lotes pueden tener distinto tamaño.",
            Matriz(200, 300, 1, 3, 90, AZUL, etiqueta="lote 1: 3 ejemplos, pérdida **2.0**", valores=[[2.0] * 3]),
            Matriz(800, 300, 1, 1, 90, NARANJA, etiqueta="lote 2: 1 ejemplo, pérdida **4.0**", valores=[[4.0]]),
        ),
        Paso(
            "Si un lote de tres ejemplos tiene pérdida promedio dos, y otro de un solo ejemplo tiene pérdida cuatro, "
            "el promedio no es tres. Hay que ponderar por el tamaño: tres por dos, más uno por cuatro, "
            "dividido entre cuatro ejemplos. Da dos coma cinco.",
            Texto(200, 520, "(2.0 + 4.0) / 2 = 3.0", 44, ROJO),
            Texto(700, 520, "✗  ignora el tamaño de los lotes", 36, ROJO),
            Texto(200, 610, "(3 × 2.0 + 1 × 4.0) / 4 = **2.5**", 44, VERDE),
            Texto(980, 610, "✓  cada ejemplo cuenta igual", 36, VERDE),
        ),
        Paso(
            "Tu clase RunningMean guarda dos números: el total, que es la suma de valor por tamaño, "
            "y la cantidad de ejemplos. Pausa el video y complétala en el notebook. "
            "La celda de prueba te dice si está bien.",
            Codigo(200, 720, "class RunningMean:\n"
                   "    def __init__(self): ...           # self.total = 0.0, self.count = 0\n"
                   "    def update(self, value, n=1): ... # registra un lote de n ejemplos\n"
                   "    def mean(self): ...               # promedio ponderado (0.0 si no hay datos)", 26),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------- capa lineal
    Escena(
        "Una capa lineal",
        Paso(
            "Ahora construimos una pieza real de una red neuronal: la capa lineal. Toma un ejemplo con tres números "
            "y produce dos. Cada salida es una suma ponderada de las entradas, más un sesgo. "
            "Cada línea del dibujo es un peso, y el entrenamiento ajusta esos números.",
            *red(250, 330),
            Texto(250, 760, "3 entradas", 30, GRIS, alinear="centro"),
            Texto(670, 760, "2 salidas", 30, GRIS, alinear="centro"),
            Texto(900, 330, "y₁ = w₁₁·x₁ + w₁₂·x₂ + w₁₃·x₃ + b₁", 40, VERDE),
            Texto(900, 410, "y₂ = w₂₁·x₁ + w₂₂·x₂ + w₂₃·x₃ + b₂", 40, VERDE),
            Texto(900, 510, "6 **pesos** (las líneas) + 2 **sesgos**", 36, GRIS),
        ),
        Paso(
            "Con un lote de cuatro ejemplos, uno por fila, toda la capa es una sola multiplicación de matrices: "
            "x por W transpuesta, más b. x tiene forma cuatro por tres. La salida tiene forma cuatro por dos: "
            "cuatro ejemplos, con dos números cada uno. Es la misma convención de formas que usa PyTorch.",
            Linea(900, 600, 1820, 600, GRIS_CLARO, 3),
            Matriz(900, 660, 4, 3, 44, AZUL, etiqueta="x: (4, 3)"),
            Texto(1055, 720, "@", 48, alinear="centro"),
            Matriz(1100, 680, 3, 2, 44, MORADO, etiqueta="Wᵀ: (3, 2)"),
            Texto(1225, 720, "+", 48, alinear="centro"),
            Matriz(1270, 720, 1, 2, 44, NARANJA, etiqueta="b: (2,)"),
            Texto(1395, 720, "=", 48, alinear="centro"),
            Matriz(1440, 660, 4, 2, 44, VERDE, etiqueta="y: (4, 2)"),
            Texto(1560, 690, "4 ejemplos,\n2 salidas\ncada uno", 30, GRIS),
        ),
    ),
    # ------------------------------------------------------------------ Module
    Escena(
        "Module: lo que toda capa sabe hacer",
        *[
            Paso(decir, Codigo(100, 230, """class Module:
    def __call__(self, x):
        return self.forward(x)

    def parameters(self):
        return []


class Linear(Module):
    def __init__(self, in_features, out_features, rng):
        self.weight = rng.normal(size=(out_features, in_features)) / np.sqrt(in_features)
        self.bias = np.zeros(out_features)

    def forward(self, x):
        return x @ self.weight.T + self.bias

    def parameters(self):
        return [self.weight, self.bias]""", 24, resaltar=lineas, id="c"), ocultar=["c"])
            for decir, lineas in [
                ("Primero, una clase base, Module, con lo que toda capa sabe hacer. Su método call llama a forward. "
                 "Así podemos escribir capa de x, en vez de capa punto forward de x. PyTorch hace exactamente lo mismo.",
                 {1, 2, 3}),
                ("Por defecto, parameters devuelve una lista vacía: una capa que no tiene nada que aprender.",
                 {5, 6}),
                ("Linear hereda de Module. En init crea sus parámetros: los pesos, una matriz de salidas por entradas "
                 "con números aleatorios pequeños, y el sesgo, que empieza en cero.",
                 {9, 10, 11, 12}),
                ("forward hace el cálculo: x por la transpuesta de los pesos, más el sesgo.", {14, 15}),
                ("Y parameters devuelve la lista de arreglos que el entrenamiento va a ajustar: los pesos y el sesgo.",
                 {17, 18}),
            ]
        ],
    ),
    # -------------------------------------------------------------------- ReLU
    Escena(
        "ReLU: una capa sin parámetros",
        Paso(
            "Entre capas lineales ponemos una activación. La más usada es ReLU: "
            "deja pasar los números positivos y cambia los negativos por cero.",
            Grafica(150, 280, 700, 520, lambda x: max(x, 0), -3, 3, -1, 3, AZUL, "ReLU(x) = max(x, 0)",
                    puntos=[(-2, "−2 → 0"), (1.5, "1.5 → 1.5", "abajo")]),
        ),
        Paso(
            "Como hereda de Module, ReLU ya sabe responder cuando la llamas, y su lista de parámetros está vacía: "
            "no tiene nada que aprender.",
            Codigo(1000, 280, "class ReLU(Module):\n    def forward(self, x):\n        return np.maximum(x, 0)", 30),
            Terminal(1000, 520, ">>> relu(np.array([-2.0, -0.5, 0.0, 1.5]))\n[0.  0.  0.  1.5]\n"
                     ">>> relu.parameters()\n[]", 26, ancho=800),
        ),
    ),
    # -------------------------------------------------------------- Sequential
    Escena(
        "Una red es una lista de capas",
        Paso(
            "Una red neuronal es una lista de capas que se aplican en orden. Sequential guarda las capas, y en forward "
            "pasa x por cada una: la salida de una capa es la entrada de la siguiente.",
            Texto(170, 400, "**x**\n(4, 3)", 40, alinear="centro"),
            Flecha(230, 450, 330, 450),
            Caja(340, 360, 360, 180, "`Linear(3, 4)`", AZUL, 36),
            Flecha(710, 450, 800, 450),
            Caja(810, 360, 260, 180, "`ReLU()`", NARANJA, 36),
            Flecha(1080, 450, 1170, 450),
            Caja(1180, 360, 360, 180, "`Linear(4, 2)`", AZUL, 36),
            Flecha(1550, 450, 1650, 450),
            Texto(1730, 400, "**salida**\n(4, 2)", 40, alinear="centro"),
        ),
        Paso(
            "Y lo mejor: Sequential también es un Module. Una red completa se usa igual que una sola capa: "
            "red de x. Esta es la idea central de PyTorch, y la vas a ver otra vez en la semana seis.",
            Marco(310, 310, 1260, 280, MORADO),
            Texto(330, 260, "`Sequential`: también es un `Module`", 34, acento=MORADO),
            Codigo(310, 670, "net = Sequential(Linear(3, 4, rng), ReLU(), Linear(4, 2, rng))\n"
                   "net(x).shape   # (4, 2): se usa igual que una capa", 30),
        ),
    ),
    # --------------------------------------------------- Sigmoid y parámetros
    Escena(
        "Tu turno 3: Sigmoid y contar parámetros",
        Paso(
            "Dos ejercicios más. Primero, la activación sigmoide: uno sobre uno más e a la menos x. "
            "Lleva cualquier número al intervalo entre cero y uno, y en cero vale exactamente un medio.",
            Grafica(150, 280, 700, 480, lambda x: 1 / (1 + np.exp(-x)), -6, 6, -0.1, 1.1, MORADO,
                    "σ(x) = 1 / (1 + e⁻ˣ)", puntos=[(0, "σ(0) = 0.5", "abajo")]),
        ),
        Paso(
            "Segundo, una función que cuente los parámetros de un modelo: recorre la lista de parámetros "
            "y suma el tamaño de cada arreglo. Por ejemplo, Linear de tres a cuatro tiene tres por cuatro pesos, "
            "más cuatro sesgos: dieciséis.",
            Texto(1000, 290, "`count_parameters(model)`", 38, acento=MORADO),
            Texto(1000, 380, "suma `.size` de cada arreglo de `model.parameters()`", 32, GRIS, ancho=820),
            Texto(1000, 500, "`Linear(3, 4)`:  3 × 4 pesos + 4 sesgos = **16**", 36),
            Texto(1000, 580, "`ReLU()`:  **0**", 36),
        ),
        Paso(
            "Antes de ejecutar la prueba, calcula a mano cuántos parámetros tiene la red completa. "
            "Pausa el video y compruébalo en el notebook.",
            Caja(1000, 690, 800, 150, "¿Y la red `net` completa?\nCalcúlalo a mano antes de probar.", NARANJA, 34),
            pausa=1.2,
        ),
    ),
    # ----------------------------------------------------------------- Dataset
    Escena(
        "Tu turno 4: un conjunto de datos",
        Paso(
            "En PyTorch, un conjunto de datos también es una clase, con solo dos métodos especiales: "
            "len, que dice cuántos ejemplos hay, y getitem, que devuelve el ejemplo número i, como un par x, y.",
            Matriz(300, 280, 6, 2, 72, AZUL, etiqueta="X: un ejemplo por fila", valores=np.arange(12).reshape(6, 2), tam=30),
            Matriz(560, 280, 6, 1, 72, VERDE, etiqueta="y", valores=[[v] for v in [0, 1, 0, 1, 1, 0]], tam=30),
            *[Texto(250, 300 + i * 72, str(i), 28, GRIS, alinear="centro") for i in range(6)],
        ),
        Paso(
            "Con estos datos, len de ds da seis. Y ds de dos devuelve la fila dos de X junto con y de dos. "
            "Recuerda que en Python se cuenta desde cero.",
            Marco(285, 270 + 2 * 72, 360, 92, NARANJA),
            Texto(900, 330, "`len(ds)`  →  6", 44, acento=AZUL),
            Texto(900, 440, "`ds[2]`  →  `([4, 5], 0)`", 44, acento=NARANJA),
        ),
        Paso(
            "Tu turno: completa ArrayDataset. Y si X e y tienen distinta cantidad de filas, init debe lanzar "
            "un ValueError, porque esos datos no tienen sentido.",
            Caja(900, 580, 900, 200, "**Tu turno 4:** completa `ArrayDataset`.\nSi X e y tienen distinto largo: "
                 "`ValueError`.", VERDE, 34),
            pausa=1.0,
        ),
    ),
    # -------------------------------------------------------------- todo junto
    Escena(
        "Todo junto: el esqueleto de un entrenamiento",
        Paso(
            "Juntemos todo. El conjunto de datos entrega lotes de ejemplos. La red produce predicciones. "
            "Medimos el error de cada lote, y el promedio acumulado nos da la pérdida de la época. "
            "Ese es el esqueleto de un entrenamiento.",
            Caja(100, 330, 330, 170, "`ArrayDataset`", VERDE, 32, subtitulo="lotes de 4"),
            Flecha(440, 415, 510, 415),
            Caja(520, 330, 330, 170, "`Sequential`", AZUL, 32, subtitulo="predicciones"),
            Flecha(860, 415, 930, 415),
            Caja(940, 330, 330, 170, "**pérdida**", ROJO, 32, subtitulo="error del lote"),
            Flecha(1280, 415, 1350, 415),
            Caja(1360, 330, 430, 170, "`RunningMean`", MORADO, 32, subtitulo="pérdida de la época"),
        ),
        Paso(
            "Solo falta una pieza: cambiar los parámetros para que la pérdida baje. "
            "Para eso existen el gradiente y la backpropagation, que veremos entre las semanas cuatro y siete.",
            Flecha(1105, 600, 1105, 510, NARANJA, 6),
            Flecha(685, 680, 685, 510, NARANJA, 6),
            Linea(685, 680, 1105, 680, NARANJA, 6),
            Linea(1105, 600, 1105, 680, NARANJA, 6),
            Caja(560, 720, 800, 170, "**La pieza que falta:** ajustar los parámetros", NARANJA, 36,
                 subtitulo="gradiente y backpropagation · semanas 4–7"),
        ),
    ),
    # ------------------------------------------------------------------- cierre
    Seccion("Cierre"),
    Escena(
        "Lo que vimos hoy",
        *[
            Paso(decir, vineta)
            for decir, vineta in zip(
                [
                    "Repasemos. El curso tiene cinco bloques, y tu entrega es lo que hay en tu repositorio privado "
                    "al inicio de la clase.",
                    "Git guarda fotos de tu proyecto, llamadas commits, y desde Colab guardas en GitHub con un solo menú.",
                    "En Python, una clase es un molde. Viste init, self, atributos, métodos, herencia "
                    "y métodos especiales.",
                    "Y construiste en NumPy el diseño de PyTorch en miniatura: un Module con forward y parameters, "
                    "capas que se combinan en Sequential, y un conjunto de datos que se indexa.",
                ],
                vinetas(150, 260, [
                    "**El curso:** cinco bloques; tu entrega es lo que hay en tu repositorio al inicio de la clase.",
                    "**Git:** un repositorio guarda commits, fotos con mensaje. Desde Colab: "
                    "«Guardar una copia en GitHub».",
                    "**Clases:** `__init__`, `self`, atributos, métodos, herencia y métodos especiales.",
                    "**PyTorch en miniatura:** `Module` con `forward` y `parameters`, `Sequential` "
                    "y un `Dataset` que se indexa.",
                ], tam=40, sep=150, ancho=1600),
            )
        ],
    ),
    Escena(
        "Para la semana 2",
        Paso(
            "Para la próxima clase: entrega los niveles de Python superados en la plataforma. "
            "Eso se revisa al inicio de la clase.",
            *vinetas(150, 260, ["**Entrega al inicio de la clase:** los niveles de Python superados en la plataforma."],
                     tam=40, ancho=1600, numeradas=True, color=NARANJA),
        ),
        Paso(
            "Repasa lo de hoy con dos notebooks de MÓLÓ, en inglés: uno de control de versiones con Git, "
            "y otro de programación orientada a objetos.",
            Vineta(150, 420, "**Repaso** (MÓLÓ, en inglés): Version Control with Git y "
                   "Object-Oriented Programming in Python.", 40, NARANJA, numero=2, ancho=1600),
        ),
        Paso(
            "Y haz la lectura previa de la semana dos: una introducción a NumPy y lo básico de álgebra lineal. "
            "Todos los enlaces están al final del notebook de la clase.",
            Vineta(150, 580, "**Lectura previa:** Introduction to NumPy y Linear Algebra Basics for ML.", 40, NARANJA,
                   numero=3, ancho=1600),
            Texto(150, 760, "Los enlaces están al final de `sem01_herramientas.ipynb`.", 36, GRIS),
        ),
        Paso(
            "Eso es todo por hoy. ¡Nos vemos en la semana dos!",
            Texto(960, 880, "**¡Nos vemos en la semana 2!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
