"""Video de la semana 8: Nivel_Intermedio/clases/sem08_pytorch.ipynb."""

from motor import (AZUL, GRIS, MORADO, NARANJA, ROJO, TINTA, VERDE, Barras, Caja, Circulo, Codigo, Dispersion, Escena,
                   Flecha, Formula, Grafica, Marco, Matriz, Paso, Pixeles, Seccion, Tabla, Terminal, Texto, Vineta, vinetas)
import numpy as np

META = {
    "pie": "Semana 8 · PyTorch · Nivel Intermedio",
    "voz": "es-EC-AndreaNeural",
    "velocidad": "-5%",
}

# Datos del notebook, copiados aquí porque el entorno de los videos no tiene torch.
# PERDIDA_500: la entropía cruzada de la parte 4 (todos los datos en cada paso), cada 25 pasos y el último.
# PERDIDA_TODO y PERDIDA_LOTES: las primeras 20 épocas mirando todos los datos en cada paso, y las 20 épocas con
# lotes de 32 (promedio de los lotes de cada época), parte 5.
# MNIST_*: imágenes de 28 × 28 con la intensidad llevada a 0–9, una cifra por píxel, fila por fila. Las tres primeras
# de entrenamiento (5, 0, 4) y los tres primeros errores del MLP en prueba.
PERDIDA_500 = [2.326, 1.037, 0.386, 0.25, 0.192, 0.159, 0.137, 0.121, 0.109, 0.099, 0.091, 0.084, 0.079, 0.074,
               0.069, 0.065, 0.062, 0.058, 0.056, 0.053, 0.05]
PERDIDA_TODO = [2.326, 2.299, 2.275, 2.252, 2.23, 2.205, 2.178, 2.149, 2.116, 2.08, 2.04, 1.995, 1.946, 1.892, 1.833,
                1.769, 1.701, 1.629, 1.553, 1.476]
PERDIDA_LOTES = [2.085, 1.111, 0.542, 0.355, 0.278, 0.24, 0.203, 0.172, 0.154, 0.143, 0.13, 0.128, 0.118, 0.124,
                 0.128, 0.136, 0.094, 0.086, 0.083, 0.076]
MNIST_ENTRENAMIENTO = [
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000000000000000000000000000000011145616994000000000000113569999986997200000000000289999999993332100000"
     "0000000189999976990000000000000000003649970025000000000000000000000593000000000000000000000000059700000000000000"
     "0000000000007920000000000000000000000000198640000000000000000000000003899410000000000000000000000027995100000000"
     "0000000000000001399700000000000000000000000009992000000000000000000000256997000000000000000000001589999600000000"
     "0000000000148999973000000000000000001289999730000000000000000016899997300000000000000000268999995000000000000000"
     "0000599975510000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000002696200000000000000000000002899980000000000000000000002899889200000000000000000002899973994000000"
     "0000000000069999993796000000000000000028997498239600000000000000028996034100992000000000000016987300000099600000"
     "0000000069831100000099700000000000029920000000009970000000000007970000000000997000000000003994000000000099500000"
     "0000003981000000000597000000000000398000000000598300000000000039500000002696000000000000003980000004896000000000"
     "0000003995213689862000000000000000399988999750000000000000000017999998500000000000000000000159995100000000000000"
     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000000000000000000000000000000000000002810000000002300000000000000461000000000460000000000000057100000"
     "0000860000000000000196000000000086000000000000069400000000029600000000000007920000000004960000000000001891000000"
     "0006940000000000006981000000000692000000000036993000000000069300022455998699100000000005987779998753108900000000"
     "0000466666320000049800000000000000000000000006950000000000000000000000000692000000000000000000000000069200000000"
     "0000000000000000069300000000000000000000000006930000000000000000000000000695000000000000000000000000069500000000"
     "0000000000000000039500000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
]
MNIST_ERRORES = [
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000000012221532000000000000000000358999899900000000000000015999999999990000000000000002999999986666000"
     "0000000015302888642100000000000000007930010100000000000000000006950000000000000000000000005981000000000000000000"
     "0000039810000000000000000000000005960000000000000000000000000899420000000000000000000000099999534311310000000000"
     "0000059999999999986000000000000000288999999999940000000000000000023799999996000000000000000000049952699600000000"
     "0000000000029997999500000000000000000000699999600000000000000000000016898500000000000000000000000035100000000000"
     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000000000000000000000000000000000000000000000000000000000388300000000000000000000000015897100000000000"
     "0000000000000028910000000000000000000000000393000000000000000000000000008400000000000000000001466312810000000000"
     "0000000038988987600000000000000000036510039971000000000000000028300003989820000000000000007700003871389100000000"
     "0000006400018810018810000000000000283038710000188000000000000005969810000001960000000000000077300000000039700000"
     "0000000000000000000001970000000000000000000000000039400000000000000000000000000493000000000000000000000000007800"
     "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
    ("0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"
     "0000000000000000000000000000000000000000000000000000000000000000000344797441000000000000000188899999999400000000"
     "0000000899999999999710000000000000089999999999998410000000000007997228742238993000000000000189710100017996100000"
     "0000000048961000179970000000000000000289930079973000000000000000001799539992000000000000000000003999973000000000"
     "0000000000001799910000000000000000000000059991000000000000000000000019999500000000000000000000002974781000000000"
     "0000000000018930491000000000000000000005950079100000000000000000000695469700000000000000000000069999950000000000"
     "0000000000069999600000000000000000000004996400000000000000000000000000000000000000000000000000000000000000000000"),
]
MNIST_ERRORES_INFO = [(6, 5), (9, 2), (7, 8)]  # (dijo, era)

PASOS_500 = np.array(list(range(0, 500, 25)) + [499])
EPOCAS = np.arange(1, 21)


def imagen(cifras):
    return np.array([int(c) for c in cifras]).reshape(28, 28)


def curva(xs, valores):
    return lambda t: float(np.interp(t, xs, valores))


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


# La compra de la semana 7, en la mitad derecha: (x, y, r) de cada nodo.
PRECIO, ENVIO, CANTIDAD = (1170, 330, 44), (1170, 520, 44), (1170, 730, 44)
SUMA, PROD, TOTAL = (1450, 425, 46), (1640, 620, 46), (1810, 620, 50)


def compra():
    """El grafo de la compra, con los valores hacia adelante."""
    return [
        *enlace(PRECIO, SUMA),
        *enlace(ENVIO, SUMA),
        *enlace(SUMA, PROD),
        *enlace(CANTIDAD, PROD),
        *enlace(PROD, TOTAL),
        Circulo(*PRECIO, "$3$", AZUL, 36),
        Circulo(*ENVIO, "$2$", AZUL, 36),
        Circulo(*CANTIDAD, "$4$", AZUL, 36),
        Texto(PRECIO[0], 380, "precio", 28, GRIS, alinear="centro"),
        Texto(ENVIO[0], 570, "envío", 28, GRIS, alinear="centro"),
        Texto(CANTIDAD[0], 780, "cantidad", 28, GRIS, alinear="centro"),
        Circulo(*SUMA, "$+$", MORADO, 46),
        Circulo(*PROD, r"$\times$", MORADO, 46),
        Circulo(*TOTAL, "$20$", VERDE, 36),
        Texto(SUMA[0], 480, "$5$", 30, GRIS, alinear="centro"),
        Texto(TOTAL[0], 676, "total", 28, GRIS, alinear="centro"),
    ]


COMPRA = '''
price = torch.tensor(3.0, requires_grad=True)
shipping = torch.tensor(2.0, requires_grad=True)
quantity = torch.tensor(4.0, requires_grad=True)

total = (price + shipping) * quantity
total.backward()

price.grad, shipping.grad, quantity.grad
'''

DOS_CAPAS = '''
class TwoLayerNet(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.hidden = nn.Linear(n_in, n_hidden)
        self.output = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h = torch.relu(self.hidden(x))
        return self.output(h)
'''

BUCLE = '''
scores = model(X)              # 1. ida
loss = loss_fn(scores, y)      # 2. cuánto nos equivocamos
optimizer.zero_grad()          # 3. borrar los gradientes viejos
loss.backward()                # 4. vuelta: backpropagation
optimizer.step()               # 5. un paso cuesta abajo
'''

ESCENAS = [
    # ------------------------------------------------------------------ portada
    Escena(
        None,
        Paso(
            "¡Hola! Este video resume la octava clase del nivel intermedio. Hoy llega PyTorch, la librería con la que "
            "vamos a entrenar redes el resto del año.",
            Texto(960, 250, "SEMANA 8 · NIVEL INTERMEDIO", 34, AZUL, alinear="centro"),
            Texto(960, 320, "**PyTorch**", 80, alinear="centro"),
            Texto(960, 450, "lo que ya sabes hacer, más rápido y para cualquier red", 46, GRIS, alinear="centro"),
            pausa=0.8,
        ),
        Paso(
            "Hoy cambia la herramienta, no las ideas. La capa, la ida y la vuelta ya las escribiste en las semanas uno, "
            "seis y siete. Y PyTorch trae las tres hechas.",
            Caja(130, 600, 500, 150, "**Semana 1**", AZUL, 38, subtitulo="la capa, con `forward`"),
            Caja(710, 600, 500, 150, "**Semana 6**", MORADO, 38, subtitulo="la ida"),
            Caja(1290, 600, 500, 150, "**Semana 7 y P1**", VERDE, 38, subtitulo="la vuelta"),
            Flecha(380, 760, 380, 820, GRIS, 5),
            Flecha(960, 760, 960, 820, GRIS, 5),
            Flecha(1540, 760, 1540, 820, GRIS, 5),
            Texto(380, 840, "`nn.Linear`", 38, alinear="centro"),
            Texto(960, 840, "`model(X)`", 38, alinear="centro"),
            Texto(1540, 840, "`loss.backward()`", 38, alinear="centro"),
        ),
    ),
    # ------------------------------------------------------------------ tensores
    Seccion("Parte 1 · Tensores"),
    Escena(
        "Un arreglo con dos superpoderes",
        Paso(
            "Un tensor de PyTorch, para empezar, es lo mismo que un arreglo de NumPy: una caja de números con una forma. "
            "Todo lo de la semana dos sigue valiendo.",
            Matriz(220, 280, 2, 2, celda=110, valores=[[1, 2], [3, 4]], tam=44, etiqueta="forma $(2, 2)$"),
            Codigo(620, 290, "t = torch.tensor([[1.0, 2.0],\n                  [3.0, 4.0]])\n"
                             "t.shape    # torch.Size([2, 2])\nt @ t      # igual que en NumPy", 32),
        ),
        Paso(
            "La diferencia son dos superpoderes. Primero, memoria: el tensor puede recordar con qué cuentas se calculó, y "
            "así PyTorch saca los gradientes solo.",
            Caja(130, 640, 800, 220, "**1. Memoria**", MORADO, 42, subtitulo="recuerda cómo se calculó: los gradientes salen solos"),
        ),
        Paso(
            "Segundo, pasaporte: puede viajar a la tarjeta gráfica, la GPU, y hacer allá sus cuentas, muchísimo más rápido "
            "cuando son grandes.",
            Caja(990, 640, 800, 220, "**2. Pasaporte**", NARANJA, 42, subtitulo="puede viajar a la GPU: cuentas grandes, más rápidas"),
        ),
    ),
    Escena(
        "De NumPy a PyTorch",
        Paso(
            "La traducción es casi palabra por palabra. Lo que más cambia: donde NumPy dice axis, PyTorch dice dim.",
            Tabla(130, 240, [["NumPy", "PyTorch"],
                             ["`np.array(...)`", "`torch.tensor(...)`"],
                             ["`a.sum(axis=0)`", "`t.sum(dim=0)`"],
                             ["`keepdims=True`", "`keepdim=True`"],
                             ["`rng.normal(size=(2, 3))`", "`torch.randn(2, 3)`"],
                             ["`np.exp`, `np.maximum(z, 0)`", "`torch.exp`, `torch.relu(z)`"]],
                  [560, 520], tam=32),
        ),
        Paso(
            "Por ejemplo, sumar con dim igual a cero colapsa las filas: uno, dos, tres, cuatro se vuelve cuatro, seis.",
            Matriz(1380, 270, 2, 2, celda=100, valores=[[1, 2], [3, 4]], tam=40),
            Flecha(1480, 490, 1480, 570, GRIS, 5),
            Texto(1510, 510, "`dim=0`", 30),
            Matriz(1380, 590, 1, 2, celda=100, valores=[[4, 6]], tam=40, color=VERDE),
        ),
    ),
    Escena(
        "El primer error: el tipo",
        Paso(
            "Ojo con el tipo. NumPy guarda los decimales en float sesenta y cuatro; PyTorch, en float treinta y dos, que "
            "ocupa la mitad, es mucho más rápido en la GPU y alcanza de sobra para entrenar.",
            Caja(130, 240, 800, 150, "NumPy: `float64`", GRIS, 40, subtitulo="unas 16 cifras · 8 bytes"),
            Caja(990, 240, 800, 150, "PyTorch: `float32`", AZUL, 40, subtitulo="unas 7 cifras · 4 bytes"),
        ),
        Paso(
            "Al convertir desde NumPy, el tensor conserva el float sesenta y cuatro, y PyTorch no mezcla tipos en una "
            "multiplicación. Es el primer error que vas a ver con datos reales.",
            Codigo(130, 440, "X_wrong = torch.from_numpy(X_train[:5])   # float64\n"
                             "W = torch.randn(64, 10)                    # float32\n"
                             "X_wrong @ W", 30),
            Terminal(130, 650, "RuntimeError: expected m1 and m2 to have the same dtype, but got: double != float", 28),
        ),
        Paso(
            "La regla práctica: las entradas, en float treinta y dos; las etiquetas de clase, en int sesenta y cuatro, "
            "que es lo que pide la entropía cruzada de PyTorch.",
            Caja(130, 800, 1660, 110, "Entradas en `float32` · etiquetas de clase en `int64`", VERDE, 40),
        ),
    ),
    Escena(
        "Tu turno 1: softmax en PyTorch",
        Paso(
            "Tu turno: traduce a PyTorch la softmax de la semana cuatro, con dim y keepdim. Una pista: en PyTorch, el "
            "máximo con dim devuelve los valores y sus posiciones. Pausa el video.",
            Formula(960, 250, r"\text{softmax}(z)_j = \frac{e^{\,z_j - \max z}}{\sum_k e^{\,z_k - \max z}}", 48,
                    alinear="centro"),
            Codigo(130, 450, "def softmax_rows(Z):\n    # TODO\n    raise NotImplementedError", 36),
            Texto(130, 740, "Pista: los valores del máximo son `Z.max(dim=1, keepdim=True).values`", 34),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------------ autograd
    Seccion("Parte 2 · autograd: la vuelta, automática"),
    Escena(
        "La compra de la semana 7",
        Paso(
            "Volvamos a la compra de la semana siete. Ese día recorrimos el grafo a mano. PyTorch trae ese motor incluido: "
            "se llama autograd, diferenciación automática.",
            *compra(),
        ),
        Paso(
            "Se usa en tres pasos. Primero, marcas los números respecto de los cuales quieres derivar. Desde ahí, PyTorch "
            "toma apuntes de cada cuenta en que participan.",
            Codigo(90, 250, COMPRA, 28, resaltar={1, 2, 3}, id="c1"),
        ),
        Paso(
            "Segundo, haces la cuenta: es la ida. Tercero, llamas a backward sobre el resultado, y PyTorch recorre sus "
            "apuntes hacia atrás: gradiente que llega por derivada local.",
            Codigo(90, 250, COMPRA, 28, resaltar={5, 6}, id="c2"),
            ocultar={"c1"},
        ),
        Paso(
            "Cada gradiente queda en el atributo grad de su entrada: cuatro, cuatro y cinco, los números de la pizarra.",
            Codigo(90, 250, COMPRA, 28, resaltar={8}, id="c3"),
            Terminal(90, 690, "(tensor(4.), tensor(4.), tensor(5.))", 30),
            grad(TOTAL[0], 515, "1"),
            grad(1570, 470, "4"),
            grad(1068, 310, "4"),
            grad(1068, 500, "4"),
            grad(1068, 710, "5"),
            ocultar={"c2"},
        ),
    ),
    Escena(
        "Los apuntes son un grafo",
        Paso(
            "Esos apuntes son un grafo de verdad: el total sale de una multiplicación, que recibe de una suma, que recibe "
            "de dos hojas. Una hoja es un número que creaste tú, y ahí se guarda su gradiente.",
            Terminal(130, 260, "total sale de: MulBackward0\n"
                               "   que recibe de: AddBackward0\n"
                               "      que recibe de: AccumulateGrad\n"
                               "      que recibe de: AccumulateGrad\n"
                               "   que recibe de: AccumulateGrad", 34),
            Caja(130, 660, 1660, 130, "**AccumulateGrad**: una hoja, un número que creaste tú; ahí se guarda su `.grad`",
                 MORADO, 36),
        ),
    ),
    Escena(
        "La red de papel, sin tabla",
        Paso(
            "En la semana siete, la vuelta por la red de papel nos tomó una tabla entera. Con autograd basta con escribir "
            "la ida y llamar a backward.",
            Codigo(90, 240, "W1 = torch.tensor([[1.0, 2.0], [-1.0, 1.0]], requires_grad=True)\n"
                            "b1 = torch.tensor([1.0, 0.0], requires_grad=True)\n"
                            "W2 = torch.tensor([[2.0, -1.0]], requires_grad=True)\n"
                            "b2 = torch.tensor([-0.5], requires_grad=True)\n\n"
                            "p = torch.sigmoid(W2 @ torch.relu(W1 @ x + b1) + b2)[0]\n"
                            "loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))\n"
                            "loss.backward()", 26),
        ),
        Paso(
            "Y salen los mismos números de la tabla.",
            Terminal(90, 640, "dL/dW1 = [[-0.73, 0.365], [0.0, 0.0]]\n"
                              "dL/db1 = [-0.365, 0.0]\n"
                              "dL/dW2 = [[-0.182, 0.0]]\n"
                              "dL/db2 = [-0.182]", 30),
            Caja(1220, 640, 600, 220, "**Semana 7:** una tabla\n**Hoy:** `loss.backward()`", VERDE, 36),
        ),
    ),
    Escena(
        "Los gradientes se acumulan",
        Paso(
            "Dos detalles importantes. Primero: backward no reemplaza el gradiente anterior, le suma el nuevo. Como una "
            "balanza que nunca vuelves a poner en cero.",
            Codigo(130, 240, "w = torch.tensor(1.0, requires_grad=True)\n"
                             "for i in range(3):\n"
                             "    loss = (w - 3) ** 2\n"
                             "    loss.backward()\n"
                             "    print(w.grad)", 32),
        ),
        Paso(
            "En w igual a uno, la derivada es menos cuatro. La primera vez está bien; después, menos ocho y menos doce son "
            "basura.",
            Terminal(130, 580, "después del backward 1: w.grad = -4.0\n"
                               "después del backward 2: w.grad = -8.0\n"
                               "después del backward 3: w.grad = -12.0", 32),
            Texto(1100, 600, "**bien**", 38, VERDE, VERDE),
            Texto(1100, 680, "**basura**", 38, ROJO, ROJO),
        ),
        Paso(
            "Por eso, entre un paso y el siguiente, hay que borrar los gradientes viejos.",
            Caja(130, 830, 1660, 100, "Antes de cada vuelta: **borrar los gradientes**", NARANJA, 38),
        ),
    ),
    Escena(
        "El descenso de la semana 4, sin derivar",
        Paso(
            "Segundo: el paso de descenso no es parte de la cuenta. Para que PyTorch no lo anote, se da dentro de un "
            "bloque que apaga los apuntes.",
            Codigo(90, 240, "w = torch.tensor(0.0, requires_grad=True)\n"
                            "for step in range(30):\n"
                            "    loss = (w - 3) ** 2        # 1. ida\n"
                            "    loss.backward()            # 2. vuelta: llena w.grad\n"
                            "    with torch.no_grad():\n"
                            "        w -= 0.1 * w.grad      # 3. un paso, sin anotarlo\n"
                            "    w.grad.zero_()             # 4. borrar", 28, resaltar={5, 6}),
        ),
        Paso(
            "Y con eso, el descenso de gradiente de la semana cuatro, sin escribir la derivada: los mismos números, hacia "
            "tres.",
            Grafica(1250, 280, 580, 400, lambda w: (w - 3) ** 2, -0.5, 6.5, -1, 12,
                    camino=[0, 0.6, 1.08, 1.464, 1.7712, 2.017, 2.2136, 2.3709, 2.4967, 2.5973, 2.6779]),
            Terminal(90, 640, "paso  1: w = 0.6000\npaso  2: w = 1.0800\npaso  3: w = 1.4640\npaso 30: w = 2.9963", 30),
        ),
    ),
    Escena(
        "Tu turno 2: la recta de los pingüinos",
        Paso(
            "Tu turno: la recta de los pingüinos de la semana cuatro, otra vez, pero con autograd: dos parámetros, sesenta "
            "pasos, sin derivar nada. Borra los dos gradientes en cada paso. Pausa el video.",
            Formula(960, 260, r"\text{masa} \approx w \cdot \text{aleta} + b \qquad w_0 = -0.5,\ b_0 = 2.5,\ "
                              r"\eta = 0.1,\ 60 \text{ pasos}", 44, alinear="centro"),
            Codigo(130, 430, "peng_losses = []\n# TODO: w, b y los 60 pasos", 36),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------------ nn.Module
    Seccion("Parte 3 · nn.Module: la red como piezas"),
    Escena(
        "Tu clase de la semana 1, en original",
        Paso(
            "En la semana uno escribiste una clase Linear con forward, y te dije que era el patrón de PyTorch. Aquí está "
            "el original.",
            Codigo(90, 240, "class Linear(Module):\n"
                            "    def __init__(self, in_features, out_features, rng): ...\n"
                            "    def forward(self, x):\n"
                            "        return x @ self.weight.T + self.bias", 26, titulo="Semana 1"),
            Codigo(90, 520, "layer = nn.Linear(64, 32)\n"
                            "layer.weight.shape   # (32, 64)\n"
                            "layer.bias.shape     # (32,)", 30, titulo="PyTorch"),
        ),
        Paso(
            "Sus pesos tienen forma treinta y dos por sesenta y cuatro, salidas por entradas, como nuestra W de la semana "
            "seis. Y ya vienen al azar, y marcados para el gradiente.",
            Formula(1150, 260, r"\texttt{layer}(X) = X W^{\top} + b", 48),
            Terminal(1150, 380, "¿layer(X) es X @ W.T + b? True\n¿sus pesos llevan gradiente? True", 26),
        ),
        Paso(
            "Las piezas de una red son como bloques de Lego: cada una sabe hacer su ida, y PyTorch sabe hacer la vuelta de todas.",
            Caja(1150, 600, 680, 200, "**Bloques de Lego**\ncada pieza hace su ida;\nPyTorch, la vuelta de todas", NARANJA, 34),
        ),
    ),
    Escena(
        "Piezas en fila: nn.Sequential",
        Paso(
            "Para apilar piezas en fila está Sequential. Esta es la red de los dígitos: sesenta y cuatro, treinta y dos, "
            "diez.",
            Codigo(130, 240, "model = nn.Sequential(\n    nn.Linear(64, 32),\n    nn.ReLU(),\n    nn.Linear(32, 10),\n)", 32),
            Caja(870, 280, 280, 110, "`Linear`\n64 → 32", AZUL, 30),
            Flecha(1160, 335, 1220, 335, GRIS, 5),
            Caja(1230, 280, 240, 110, "`ReLU`", NARANJA, 30),
            Flecha(1480, 335, 1540, 335, GRIS, 5),
            Caja(1550, 280, 280, 110, "`Linear`\n32 → 10", AZUL, 30),
        ),
        Paso(
            "Tiene dos mil cuatrocientos diez parámetros, los mismos de la semana siete, sin escribir una sola fórmula de "
            "la vuelta.",
            Formula(870, 470, r"64 \cdot 32 + 32 + 32 \cdot 10 + 10 = 2410", 44),
            Terminal(130, 600, "parámetros: 2410\nsalida para 5 dígitos: (5, 10)", 32),
        ),
    ),
    Escena(
        "Tu propia clase",
        Paso(
            "Cuando la red no es una fila simple, escribes una clase que hereda de Module. En init, primero super, y "
            "después cada pieza como atributo.",
            Codigo(130, 240, DOS_CAPAS, 30, resaltar={2, 3, 4, 5}, id="d1"),
        ),
        Paso(
            "Así PyTorch la registra, y sus parámetros aparecen en parameters, donde el optimizador los busca.",
            Terminal(1130, 240, "hidden.weight  (32, 64)\nhidden.bias    (32,)\noutput.weight  (10, 32)\noutput.bias    (10,)", 30),
        ),
        Paso(
            "En forward va la ida. Pero no llamas a forward directamente: llamas al modelo, y el método call llama a "
            "forward por ti, como en la semana uno.",
            Codigo(130, 240, DOS_CAPAS, 30, resaltar={7, 8, 9}, id="d2"),
            Caja(130, 740, 1660, 110, "`net(x)`  →  `__call__`  →  `forward`", AZUL, 40),
            ocultar={"d1"},
        ),
    ),
    Escena(
        "Tu turno 3: un MLP de cualquier profundidad",
        Paso(
            "Tu turno: completa la clase MLP, con una capa lineal entre cada par de tamaños y ReLU después de cada una "
            "menos la última. Una trampa: si guardas las capas en una lista común de Python, PyTorch no las ve. Pausa el "
            "video.",
            Codigo(130, 240, "class MLP(nn.Module):\n"
                             "    def __init__(self, sizes):    # por ejemplo [64, 32, 16, 10]\n"
                             "        super().__init__()\n"
                             "        # TODO\n\n"
                             "    def forward(self, x):\n"
                             "        # TODO\n"
                             "        raise NotImplementedError", 32),
            Caja(130, 740, 1660, 120, "`nn.Sequential` o `nn.ModuleList`: **nunca** una lista común de Python", ROJO, 38),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------------ bucle
    Seccion("Parte 4 · Pérdida, optimizador y el bucle de cinco líneas"),
    Escena(
        "La pérdida y el optimizador",
        Paso(
            "Faltan dos piezas. La pérdida: la entropía cruzada de PyTorch recibe los puntajes crudos de la red, en inglés "
            "logits, y las etiquetas enteras. Da lo mismo que la fórmula de la semana cuatro.",
            Codigo(130, 240, "loss_fn = nn.CrossEntropyLoss()\nloss = loss_fn(scores, y)     # puntajes crudos, etiquetas enteras", 30),
            Terminal(130, 420, "nn.CrossEntropyLoss:       2.3259\nla fórmula de la semana 4: 2.3259", 30),
        ),
        Paso(
            "Aplica la softmax por dentro. Así que no le pases probabilidades: ya las calcula ella.",
            Caja(1000, 420, 830, 140, "**No** le pases probabilidades", ROJO, 40, subtitulo="aplica la softmax por dentro"),
        ),
        Paso(
            "El optimizador recibe la lista de parámetros y hace dos cosas: borrar todos los gradientes, y dar el paso de "
            "descenso a todos a la vez.",
            Codigo(130, 640, "optimizer = torch.optim.SGD(model.parameters(), lr=0.5)\n"
                             "optimizer.zero_grad()     # borrar todos los gradientes\n"
                             "optimizer.step()          # un paso, a todos a la vez", 28),
            Formula(1260, 700, r"\theta \leftarrow \theta - \eta\, \nabla L", 52),
        ),
    ),
    Escena(
        "El bucle de cinco líneas",
        Paso(
            "Con eso, un paso de entrenamiento son cinco líneas, siempre las mismas y siempre en este orden. "
            "Las vas a escribir todo el año, para cualquier red.",
            Codigo(130, 250, BUCLE, 38),
        ),
        Paso(
            "Es como practicar tiros libres: lanzas, miras cuánto fallaste, borras la nota del tiro anterior, ves qué "
            "parte del movimiento tuvo la culpa, y la corriges un poco.",
            Caja(130, 660, 300, 150, "**1**\nlanzar", AZUL, 30),
            Caja(470, 660, 300, 150, "**2**\n¿cuánto fallé?", AZUL, 30),
            Caja(810, 660, 300, 150, "**3**\nborrar la nota anterior", NARANJA, 30),
            Caja(1150, 660, 300, 150, "**4**\n¿quién tuvo la culpa?", MORADO, 30),
            Caja(1490, 660, 300, 150, "**5**\ncorregir un poco", VERDE, 30),
        ),
        Paso(
            "Si cambias el orden, no hay ningún error: la red simplemente no aprende.",
            Caja(130, 860, 1660, 90, "Otro orden **no da error**: la red simplemente no aprende", ROJO, 36),
        ),
    ),
    Escena(
        "Los dígitos, con PyTorch",
        Paso(
            "Con esas cinco líneas, entrenamos la red de los dígitos con la receta de la semana siete: quinientos pasos, "
            "todos los ejemplos en cada paso. La pérdida baja desde logaritmo de diez.",
            Dispersion(200, 270, 620, 440, [((), ())], rx=(0, 500), ry=(0, 2.5), marcas_x=(0, 250, 500),
                       marcas_y=(0, 1, 2), etiqueta_x="paso", etiqueta_y="entropía cruzada",
                       curvas=[(lambda t: 2.303, GRIS, 3), (curva(PASOS_500, PERDIDA_500), AZUL, 6)]),
            Texto(560, 318, "adivinar: $\\ln 10 = 2.30$", 28, GRIS, alinear="centro"),
        ),
        Paso(
            "Resultado: cero coma noventa y siete, el mismo nivel que nuestra vuelta en NumPy, sin escribir la vuelta.",
            Barras(1060, 330, 760, 380, ["línea base", "semana 7, NumPy", "PyTorch"], [0.102, 0.969, 0.971],
                   [GRIS, AZUL, VERDE], vmax=1.0, formato="{:.3f}", tam=28),
        ),
    ),
    # ------------------------------------------------------------------ DataLoader
    Seccion("Parte 5 · Dataset y DataLoader: aprender de a bocados"),
    Escena(
        "Estudiar con fichas",
        Paso(
            "Hasta ahora, cada paso miraba los mil trescientos cuarenta y siete dígitos. Con millones de ejemplos, cada "
            "paso sería lentísimo.",
            Matriz(150, 380, 1, 43, celda=38, color=GRIS, id="mazo"),
            Texto(960, 300, "1347 dígitos en **cada paso**", 38, alinear="centro", id="todo"),
        ),
        Paso(
            "La salida es estudiar con fichas: barajas el mazo, tomas un puñado de treinta y dos, aprendes de él, y tomas "
            "el siguiente. Cada puñado es un lote; el mazo entero, una época.",
            Matriz(150, 380, 1, 43, celda=38, color=AZUL),
            Marco(144, 374, 50, 50, NARANJA),
            Texto(150, 450, "un **lote**: 32 dígitos", 34),
            Texto(1560, 450, "una **época**: los 43 lotes", 34, alinear="centro"),
            ocultar={"mazo", "todo"},
        ),
        Paso(
            "El gradiente de un lote es ruidoso, pero muchísimo más barato. Y en una época das cuarenta y tres pasos en "
            "vez de uno.",
            Caja(130, 600, 1660, 130, "Una época: **43 pasos** con lotes de 32 · **1 paso** con todos los datos", VERDE, 38),
        ),
    ),
    Escena(
        "Dataset y DataLoader",
        Paso(
            "PyTorch reparte el trabajo en dos piezas. Un Dataset sabe cuántos ejemplos hay y te da el ejemplo número i, "
            "con dos métodos especiales, len y getitem.",
            Codigo(90, 240, "class DigitsDataset(Dataset):\n"
                            "    def __init__(self, images, labels):\n"
                            "        self.images = images\n"
                            "        self.labels = labels\n\n"
                            "    def __len__(self):\n"
                            "        return len(self.labels)\n\n"
                            "    def __getitem__(self, i):\n"
                            "        x = torch.tensor(self.images[i], dtype=torch.float32)\n"
                            "        return x, int(self.labels[i])", 26),
        ),
        Paso(
            "Un DataLoader baraja el Dataset y lo entrega en lotes ya apilados: treinta y dos imágenes y treinta y dos "
            "etiquetas.",
            Codigo(1180, 240, "loader = DataLoader(my_set,\n    batch_size=32, shuffle=True)\n"
                              "X_batch, y_batch = next(iter(loader))", 24),
            Terminal(1180, 430, "X_batch: (32, 64)\ny_batch: (32,)  int64", 28),
        ),
        Paso(
            "Y si tus imágenes están en el disco, getitem puede leer un archivo a la vez.",
            Caja(90, 760, 1740, 110, "`__getitem__` puede leer **un archivo del disco** a la vez: así se manejan miles de imágenes",
                 MORADO, 32),
        ),
    ),
    Escena(
        "Lotes contra todos los datos",
        Paso(
            "Entrenemos la misma red con lotes de treinta y dos, durante veinte épocas. En gris, todos los datos en cada "
            "paso; en azul, los lotes.",
            Dispersion(200, 270, 680, 440, [((), ())], rx=(1, 20), ry=(0, 2.5), marcas_x=(1, 5, 10, 15, 20),
                       marcas_y=(0, 1, 2), etiqueta_x="época (pasadas por los datos)", etiqueta_y="entropía cruzada",
                       curvas=[(curva(EPOCAS, PERDIDA_TODO), GRIS, 5), (curva(EPOCAS, PERDIDA_LOTES), AZUL, 6)]),
            Texto(1010, 290, "**todos los datos por paso**", 32, GRIS, GRIS),
            Texto(1010, 345, "**lotes de 32**", 32, AZUL, AZUL),
        ),
        Paso(
            "Con los mismos datos vistos, los lotes aprenden muchísimo más: con veinte pasadas llegan casi adonde todos "
            "los datos llegan con quinientas. Por eso siempre se entrena con lotes.",
            Terminal(1010, 440, "todos los datos: 500 épocas → 0.971\nlotes de 32:      20 épocas → 0.962", 28),
            Caja(1010, 620, 820, 120, "Siempre se entrena **con lotes**", VERDE, 40),
        ),
    ),
    Escena(
        "Tu turno 4: ¿por qué no aprende?",
        Paso(
            "El último tu turno. Esta función de entrenamiento tiene exactamente dos errores, y ninguno hace fallar el "
            "código: la pérdida se queda clavada en dos coma tres. Encuéntralos y corrígelos. Pausa el video.",
            Codigo(90, 240, "def buggy_train_epoch(model, loader, loss_fn, optimizer):\n"
                            "    model.train()\n"
                            "    total = 0.0\n"
                            "    for X_batch, y_batch in loader:\n"
                            "        scores = model(X_batch)\n"
                            "        probs = torch.softmax(scores, dim=1)\n"
                            "        loss = loss_fn(probs, y_batch)\n"
                            "        loss.backward()\n"
                            "        optimizer.zero_grad()\n"
                            "        optimizer.step()\n"
                            "        total += loss.item()\n"
                            "    return total / len(loader)", 28),
            Terminal(1220, 240, "época 1: pérdida 2.304\népoca 2: pérdida 2.304\n...\népoca 5: pérdida 2.304\n"
                                "accuracy de prueba: 0.036", 28),
            Caja(1220, 560, 610, 150, "**Dos** errores", ROJO, 40, subtitulo="ninguno da error"),
            pausa=1.2,
        ),
    ),
    # ------------------------------------------------------------------ MNIST
    Seccion("Parte 6 · Un problema de verdad: MNIST"),
    Escena(
        "MNIST",
        Paso(
            "Los dígitos de sklearn eran de ocho por ocho. MNIST es el conjunto clásico de dígitos escritos a mano: "
            "setenta mil imágenes de veintiocho por veintiocho.",
            *[Pixeles(160 + 340 * i, 250, imagen(c), celda=9, vmax=9, etiqueta=str(n))
              for i, (c, n) in enumerate(zip(MNIST_ENTRENAMIENTO, (5, 0, 4)))],
            Caja(1230, 250, 600, 250, "**MNIST**\n60 000 para entrenar\n10 000 para probar", AZUL, 38),
        ),
        Paso(
            "Cada imagen se aplana en setecientos ochenta y cuatro números, y se divide entre doscientos cincuenta y cinco "
            "para que quede entre cero y uno.",
            Codigo(160, 620, "Xm_train = mnist_train.data.reshape(-1, 784).float() / 255", 30),
            Texto(160, 740, "$(60000, 28, 28)$ enteros de 0 a 255  →  $(60000, 784)$ números entre 0 y 1", 34),
        ),
    ),
    Escena(
        "Tres modelos",
        Paso(
            "Tres modelos, tres épocas cada uno. La línea base: cero coma once. Un modelo lineal, que es la regresión "
            "logística entrenada con lotes: cero coma noventa y dos.",
            Barras(200, 330, 900, 400, ["línea base", "lineal 784 → 10", "MLP 784 → 128 → 10"], [0.113, 0.919, 0.956],
                   [GRIS, AZUL, VERDE], vmax=1.0, formato="{:.3f}", tam=28),
        ),
        Paso(
            "Y un MLP con una capa oculta: cero coma noventa y seis. La capa oculta baja el error casi a la mitad, en unos "
            "segundos de CPU.",
            Caja(1230, 380, 600, 220, "error: **8.1 %** → **4.4 %**", VERDE, 42, subtitulo="casi a la mitad"),
        ),
    ),
    Escena(
        "¿En qué se equivoca?",
        Paso(
            "¿En qué se equivoca? Algunos errores son disculpables: este cinco parece un seis. Otros, no tanto.",
            *[Pixeles(160 + 340 * i, 250, imagen(c), celda=9, vmax=9, color=ROJO, etiqueta=f"dijo {p}, era {r}")
              for i, (c, (p, r)) in enumerate(zip(MNIST_ERRORES, MNIST_ERRORES_INFO))],
        ),
        Paso(
            "Un MLP no sabe qué píxel está al lado de cuál. Las redes convolucionales de la semana trece sí lo saben, y en "
            "MNIST pasan del noventa y nueve por ciento.",
            Caja(130, 660, 1660, 150, "Un MLP no sabe qué píxel está **al lado** de cuál",
                 MORADO, 38, subtitulo="redes convolucionales (semana 13): más del 99 % en MNIST"),
        ),
    ),
    # ------------------------------------------------------------------ GPU
    Seccion("Parte 7 · CPU vs GPU"),
    Escena(
        "Chefs y un estadio",
        Paso(
            "Una CPU es como unos pocos chefs expertos: rapidísimos, capaces de cualquier receta. Una GPU es un estadio "
            "con miles de ayudantes que solo saben sumar y multiplicar, pero todos a la vez.",
            Caja(130, 240, 780, 150, "**CPU**", AZUL, 44, subtitulo="pocos chefs expertos"),
            *[Circulo(310 + 140 * i, 540, 46, "", AZUL) for i in range(4)],
            Caja(1010, 240, 780, 150, "**GPU**", NARANJA, 44, subtitulo="miles de ayudantes, todos a la vez"),
            *[Circulo(1060 + 62 * (i % 12), 450 + 48 * (i // 12), 14, "", NARANJA) for i in range(60)],
        ),
        Paso(
            "Para un sándwich, gana el chef: mientras le explicas al estadio, ya terminó. Para diez mil sándwiches, gana "
            "el estadio. Y multiplicar matrices grandes son diez mil sándwiches.",
            Caja(130, 760, 780, 120, "1 sándwich: **gana la CPU**", AZUL, 38),
            Caja(1010, 760, 780, 120, "10 000 sándwiches: **gana la GPU**", NARANJA, 38),
        ),
    ),
    Escena(
        "Multiplicar matrices",
        Paso(
            "Midamos una multiplicación de dos matrices cuadradas en cada uno.",
            Tabla(130, 250, [["$n \\times n$", "CPU", "GPU", "CPU / GPU"],
                             ["64", "0.01 ms", "0.03 ms", "0.4"],
                             ["256", "0.10 ms", "0.03 ms", "3"],
                             ["1024", "5.3 ms", "0.14 ms", "39"],
                             ["2048", "48 ms", "0.98 ms", "49"]],
                  [260, 260, 260, 300], tam=34),
        ),
        Paso(
            "Con matrices chicas, la GPU pierde o empata: encargarle la cuenta cuesta más que hacerla. Con matrices "
            "grandes gana por mucho. En Colab, con la T4, cambian los números, pero no la forma.",
            Marco(122, 318, 1096, 78, ROJO),
            Marco(122, 532, 1096, 78, VERDE),
            Caja(1310, 300, 520, 260, "La GPU conviene cuando hay **mucha cuenta**", VERDE, 40),
        ),
    ),
    Escena(
        "El mismo dispositivo",
        Paso(
            "Para usar la GPU, el modelo y los datos deben estar en el mismo dispositivo: el método to es el pasaporte que "
            "los lleva.",
            Codigo(130, 240, "device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")\n"
                             "model = model.to(device)          # los parámetros viajan\n"
                             "X_batch = X_batch.to(device)      # los datos también", 28),
        ),
        Paso(
            "Si olvidas uno, da este error: el segundo que más vas a ver.",
            Terminal(130, 460, "RuntimeError: Expected all tensors to be on the same device,\n"
                               "but got mat1 is on cpu, different from other tensors on cuda:0", 28),
        ),
        Paso(
            "Por eso las funciones del notebook mandan cada lote adonde vive el modelo. En Colab, la GPU se activa desde "
            "el menú de entorno de ejecución.",
            Caja(130, 680, 1660, 150, "Colab: **Entorno de ejecución → Cambiar tipo de entorno de ejecución → GPU T4**",
                 NARANJA, 34, subtitulo="el entorno se reinicia: vuelve a ejecutar todo desde arriba"),
        ),
    ),
    Escena(
        "¿Siempre conviene?",
        Paso(
            "¿Conviene siempre? Una época de MNIST con dos redes. Con la red chica, la GPU pierde: cada lote es tan poca "
            "cuenta que el tiempo se va en encargarla y mover los datos.",
            Tabla(130, 250, [["Red", "Parámetros", "CPU", "GPU"],
                             ["chica, lotes de 64", "101 770", "0.9 s", "1.8 s"],
                             ["ancha, lotes de 256", "1 863 690", "2.9 s", "0.5 s"]],
                  [560, 360, 240, 240], tam=36),
        ),
        Paso(
            "Con la red ancha, de casi dos millones de parámetros, la GPU gana con claridad. La regla: la GPU conviene "
            "cuando hay mucha cuenta por paso, y desde la semana trece, con imágenes, va a ser indispensable.",
            Caja(130, 540, 1660, 140, "La GPU conviene cuando hay **mucha cuenta por paso**",
                 VERDE, 40, subtitulo="redes grandes · lotes grandes · imágenes"),
        ),
        Paso(
            "En el Sprint uno, en la semana once, vas a tener quince minutos de GPU.",
            Caja(130, 740, 1660, 110, "Sprint 1 (semana 11): **15 minutos de GPU**", NARANJA, 40),
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
                    "Repasemos. Un tensor es un arreglo de NumPy con memoria y pasaporte. Y backward recorre el grafo "
                    "hacia atrás: es la vuelta de la semana siete.",
                    "Los gradientes se acumulan: hay que borrarlos en cada paso. Y las piezas de PyTorch se registran "
                    "solas, si no las escondes en una lista común.",
                    "El bucle de cinco líneas, por lotes. Y la GPU conviene cuando hay mucha cuenta por paso, con el modelo y los "
            "datos en el mismo dispositivo.",
                ],
                (lambda v: [v[0:2], v[2:4], v[4:6]])(vinetas(130, 240, [
                    "**Tensor** = arreglo de NumPy + memoria + pasaporte. Entradas `float32`, etiquetas `int64`.",
                    "`loss.backward()` deja cada gradiente en `.grad`: la vuelta de la semana 7.",
                    "Los gradientes **se acumulan**: bórralos en cada paso. El paso va en `torch.no_grad()`.",
                    "`nn.Linear`, `nn.Sequential`, `nn.Module`: piezas que se **registran solas**.",
                    "Cinco líneas: ida, pérdida, `zero_grad`, `backward`, `step`. Con un `DataLoader`, por lotes.",
                    "La GPU conviene con **mucha cuenta por paso**. Modelo y datos, en el **mismo dispositivo**.",
                ], tam=36, sep=112, ancho=1650)),
            )
        ],
    ),
    Escena(
        "Un poco de historia",
        Paso(
            "PyTorch nació en dos mil dieciséis, en el laboratorio de inteligencia artificial de Facebook, hoy Meta, como "
            "heredero de Torch. Su idea clave: armar el grafo mientras se hace la cuenta, como hoy. Hoy lo usa la mayoría "
            "de la investigación en aprendizaje profundo.",
            Caja(130, 300, 480, 200, "**Torch**", GRIS, 50, subtitulo="la librería anterior"),
            Flecha(630, 400, 720, 400, GRIS, 5),
            Caja(740, 300, 480, 200, "**2016**", AZUL, 50, subtitulo="nace PyTorch"),
            Flecha(1240, 400, 1330, 400, GRIS, 5),
            Caja(1350, 300, 480, 200, "**Hoy**", VERDE, 50, subtitulo="la mayoría de la investigación"),
            Caja(130, 620, 1700, 130, "La idea clave: el grafo se arma **mientras** se hace la cuenta", MORADO, 38),
        ),
    ),
    Escena(
        "Para la semana 9",
        Paso(
            "Para la semana nueve, entrega el P dos: un MLP sobre Fashion-MNIST, que es como MNIST pero con ropa, y más "
            "difícil. Vas a cambiar una sola cosa a la vez, para medir cuánto aporta cada una.",
            Vineta(130, 240, "**P2:** MLP sobre Fashion-MNIST + tabla de ablación (una variable a la vez). "
                   "En `P2/`, al inicio de la semana 9.", 36, NARANJA, numero=1, ancho=1650),
        ),
        Paso(
            "La cuota semanal. Y la semana nueve es un seminario: tres grupos presentan funciones de pérdida, "
            "inicialización de pesos y normalización por lotes.",
            Vineta(130, 400, "**Cuota semanal** en ioai.artix.tech.", 36, NARANJA, numero=2, ancho=1650),
            Vineta(130, 490, "**Seminario:** funciones de pérdida, inicialización de pesos y batch norm.", 36, NARANJA,
                   numero=3, ancho=1650),
        ),
        Paso(
            "Si quieres repasar en inglés, el notebook enlaza el tutorial oficial de PyTorch. Desde hoy, todas tus redes "
            "van a usar estas cinco líneas. ¡Nos vemos en la semana nueve!",
            Vineta(130, 580, "**Para leer:** Learn the Basics, el tutorial oficial de PyTorch (enlace en el notebook).", 36,
                   NARANJA, numero=4, ancho=1650),
            Texto(960, 820, "**¡Nos vemos en la semana 9!**", 52, AZUL, alinear="centro"),
            pausa=1.5,
        ),
    ),
]
