"""Convierte un guion de video (un archivo .py en videos/guiones/) en un MP4 narrado.

    .venv-video/bin/python videos/motor.py videos/guiones/sem01_herramientas.py
    .venv-video/bin/python videos/motor.py videos/guiones/sem01_herramientas.py --solo-cuadros
    .venv-video/bin/python videos/motor.py videos/guiones/sem01_herramientas.py --solo-cuadros --escena 5

Un guion define META (título, pie de página, voz) y ESCENAS, una lista de Escena.
Cada Escena tiene Pasos: un Paso es una frase de narración más los elementos que
aparecen en pantalla mientras se dice (Texto, Codigo, Caja, Flecha, ...). Los
elementos se acumulan dentro de la escena hasta que un Paso los oculta por su id.

Salida en videos/salida/<nombre>/ (gitignored):
    <nombre>.mp4      el video, con subtítulos incrustados como pista opcional
    <nombre>.srt      los mismos subtítulos, para subirlos a YouTube
    cuadros/          un PNG por paso, para revisar el diseño sin escuchar el video
    hojas/            el último paso de cada escena, de a cuatro por imagen (revisión rápida)
La narración se genera con edge-tts (necesita internet) y se guarda en
videos/salida/cache_tts/, así que volver a construir solo sintetiza las frases nuevas.
Las ecuaciones (Formula y $...$ en el marcado) se compilan con LaTeX y quedan en
videos/salida/cache_latex/.

--solo-cuadros dibuja los PNG sin narración ni video: es la forma rápida de revisar
el diseño. --escena N limita todo a la escena N (desde 1).
"""

import argparse
import asyncio
import hashlib
import importlib.util
import math
import re
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.styles import get_style_by_name
from pygments.token import Token

ANCHO, ALTO = 1920, 1080
FPS = 24
MUESTREO = 48000
CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "salida"

# ---------------------------------------------------------------- tema visual

FONDO = "#FAF8F3"
TINTA = "#1F2328"
GRIS = "#6B7280"
GRIS_CLARO = "#E5E2DA"
AZUL = "#2563EB"
VERDE = "#16A34A"
NARANJA = "#EA580C"
MORADO = "#7C3AED"
ROJO = "#DC2626"
FONDO_CODIGO = "#0D1117"


def _fuente(patron):
    ruta = subprocess.run(["fc-match", "-f", "%{file}", patron], capture_output=True, text=True).stdout
    if not ruta:
        sys.exit(f"No encuentro la fuente {patron!r}: instala fonts-noto y fonts-dejavu.")
    return ruta


RUTAS_FUENTES = {
    "normal": _fuente("Noto Sans"),
    "negrita": _fuente("Noto Sans:bold"),
    "codigo": _fuente("DejaVu Sans Mono"),
    "codigo_negrita": _fuente("DejaVu Sans Mono:bold"),
    # Para los caracteres que Noto Sans no trae (→, ✓, ✗, superíndices, ...).
    "respaldo": _fuente("DejaVu Sans"),
    "respaldo_negrita": _fuente("DejaVu Sans:bold"),
}
_cache_fuentes = {}
_cmaps = {}


def _cubre(estilo, caracter):
    if estilo not in _cmaps:
        _cmaps[estilo] = set(TTFont(RUTAS_FUENTES[estilo], fontNumber=0).getBestCmap())
    return ord(caracter) in _cmaps[estilo]


def fuente(estilo, tam):
    clave = (estilo, tam)
    if clave not in _cache_fuentes:
        _cache_fuentes[clave] = ImageFont.truetype(RUTAS_FUENTES[estilo], tam)
    return _cache_fuentes[clave]


def rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))


def mezclar(c1, c2, t):
    """Color entre c1 (t=0) y c2 (t=1)."""
    a, b = rgb(c1), rgb(c2)
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def tinte(color, t=0.88):
    return mezclar(color, "#FFFFFF", t)


# ------------------------------------------------------------------- LaTeX
# Las ecuaciones se componen con LaTeX de verdad (latex + dvipng del sistema) y se pegan
# como imagen. Cada fórmula se compila una vez y queda en salida/cache_latex/.

CACHE_LATEX = SALIDA / "cache_latex"
# Colores del tema con nombre, para usar \textcolor{azul}{W} dentro de una fórmula.
_COLORES_LATEX = {"tinta": TINTA, "gris": GRIS, "azul": AZUL, "verde": VERDE, "naranja": NARANJA,
                  "morado": MORADO, "rojo": ROJO}
_PREAMBULO_LATEX = r"""\documentclass[10pt]{article}
\usepackage{amsmath,amssymb,bm,lmodern,xcolor}
\usepackage[active,tightpage,lyx]{preview}
%s
\begin{document}
\begin{preview}\color{base}$%s%s$\end{preview}
\end{document}
"""
_cache_latex = {}


def latex(tex, tam, color=TINTA, display=True):
    """Compila la fórmula tex y devuelve (imagen RGBA, profundidad bajo la línea base en px).
    tam es el tamaño de letra en px, como en Texto."""
    clave = (tex, tam, color, display)
    if clave in _cache_latex:
        return _cache_latex[clave]
    colores = dict(_COLORES_LATEX, base=color)
    definiciones = "\n".join(rf"\definecolor{{{n}}}{{HTML}}{{{c.lstrip('#')}}}" for n, c in colores.items())
    fuente_tex = _PREAMBULO_LATEX % (definiciones, r"\displaystyle " if display else "", tex)
    nombre = hashlib.sha1(fuente_tex.encode() + str(tam).encode()).hexdigest()[:16]
    png, prof = CACHE_LATEX / f"{nombre}.png", CACHE_LATEX / f"{nombre}.txt"
    if not png.exists():
        CACHE_LATEX.mkdir(parents=True, exist_ok=True)
        tmp = CACHE_LATEX / f"tmp_{nombre}"
        tmp.mkdir(exist_ok=True)
        (tmp / "f.tex").write_text(fuente_tex)
        r = subprocess.run(["latex", "-interaction=nonstopmode", "-halt-on-error", "f.tex"], cwd=tmp,
                           capture_output=True, text=True)
        if r.returncode:
            error = [l for l in r.stdout.splitlines() if l.startswith("!")]
            sys.exit(f"LaTeX no compila {tex!r}: {' '.join(error) or r.stdout[-500:]}")
        # Se compila al doble de resolución y se reduce: bordes más suaves que el antialias de dvipng.
        # 10 pt de LaTeX a D ppp miden 10·D/72.27 px; se elige D para que 1 em ≈ 1.1·tam.
        dpi = round(2 * tam * 1.1 * 72.27 / 10)
        r = subprocess.run(["dvipng", "-D", str(dpi), "-bg", "Transparent", "--depth", "-o", "f.png", "f.dvi"],
                           cwd=tmp, capture_output=True, text=True)
        profundidad = int(re.search(r"depth=(-?\d+)", r.stdout).group(1)) / 2
        img = Image.open(tmp / "f.png").convert("RGBA")
        img.resize((round(img.width / 2), round(img.height / 2)), Image.LANCZOS).save(png)
        prof.write_text(str(profundidad))
        shutil.rmtree(tmp)
    _cache_latex[clave] = Image.open(png).copy(), float(prof.read_text())
    return _cache_latex[clave]


# ------------------------------------------------- texto enriquecido (marcado)
# **negrita**, `código`, ==resaltado== (en el color de acento del elemento) y $fórmula$
# (LaTeX en línea, del color del texto).

_MARCAS = re.compile(r"(\*\*.+?\*\*|`.+?`|==.+?==|\$.+?\$)")


def _tramos(texto):
    for parte in _MARCAS.split(texto):
        if not parte:
            continue
        if parte.startswith("**"):
            yield parte[2:-2], "negrita"
        elif parte.startswith("`"):
            yield parte[1:-1], "codigo"
        elif parte.startswith("=="):
            yield parte[2:-2], "acento"
        elif parte.startswith("$"):
            yield parte[1:-1], "mate"
        else:
            yield parte, "normal"


def _piezas(texto, estilo, tam):
    """Parte el texto en [(trozo, fuente)], usando la fuente de respaldo donde falte un carácter."""
    if estilo == "codigo":
        return [(texto, fuente("codigo", round(tam * 0.88)))]
    base = "negrita" if estilo in ("negrita", "acento") else "normal"
    respaldo = "respaldo_negrita" if base == "negrita" else "respaldo"
    piezas = []
    for c in texto:
        nombre = base if _cubre(base, c) or c.isspace() else respaldo
        if piezas and piezas[-1][1] == nombre:
            piezas[-1][0] += c
        else:
            piezas.append([c, nombre])
    return [(t, fuente(n, tam)) for t, n in piezas]


def _largo(texto, estilo, tam):
    return sum(f.getlength(t) for t, f in _piezas(texto, estilo, tam))


def _maquetar(texto, tam, ancho_max):
    """Parte el texto en líneas de palabras [(palabra, estilo, ancho)] que caben en ancho_max."""
    espacio = fuente("normal", tam).getlength(" ")
    lineas = []
    for parrafo in texto.split("\n"):
        linea, x = [], 0.0
        for trozo, estilo in _tramos(parrafo):
            # El código y las fórmulas no se parten en palabras: cada uno es una unidad.
            palabras = [trozo] if estilo in ("codigo", "mate") else re.split(r"(?<= )", trozo)
            for palabra in palabras:
                if not palabra:
                    continue
                if estilo == "mate":
                    ancho = latex(palabra, tam, display=False)[0].width
                else:
                    ancho = _largo(palabra.rstrip(" "), estilo, tam)
                if estilo == "codigo":
                    ancho += tam * 0.3
                # Solo se corta donde había un espacio: un signo pegado a la palabra anterior no baja solo.
                if linea and ancho_max and x + ancho > ancho_max and linea[-1][0].endswith(" "):
                    lineas.append(linea)
                    linea, x = [], 0.0
                    palabra = palabra.lstrip(" ")
                linea.append((palabra, estilo, ancho))
                x += ancho + (espacio if palabra.endswith(" ") else 0)
        lineas.append(linea)
    return lineas, espacio


def _ancho_linea(linea, espacio):
    return sum(a + (espacio if p.endswith(" ") else 0) for p, _, a in linea) - (
        espacio if linea and linea[-1][0].endswith(" ") else 0
    )


def medir_texto(texto, tam, ancho_max=None, interlineado=1.35):
    lineas, espacio = _maquetar(texto, tam, ancho_max)
    return max((_ancho_linea(l, espacio) for l in lineas), default=0), len(lineas) * tam * interlineado


def dibujar_texto(d, x, y, texto, tam, color=TINTA, acento=AZUL, ancho_max=None, alinear="izq", interlineado=1.35):
    """Dibuja texto con marcado. alinear: 'izq' o 'centro' (x es entonces el centro)."""
    lineas, espacio = _maquetar(texto, tam, ancho_max)
    alto_linea = tam * interlineado
    for i, linea in enumerate(lineas):
        ancho = _ancho_linea(linea, espacio)
        cx = x - ancho / 2 if alinear == "centro" else x
        base = y + i * alto_linea + tam  # línea base
        for palabra, estilo, ancho_palabra in linea:
            texto_palabra = palabra.rstrip(" ")
            if estilo == "codigo":
                pad = tam * 0.15
                d.rounded_rectangle(
                    (cx, base - tam * 0.95, cx + ancho_palabra, base + tam * 0.25), radius=6, fill=tinte(acento, 0.9)
                )
                f = fuente("codigo", round(tam * 0.88))
                d.text((cx + pad, base), texto_palabra, font=f, fill=mezclar(acento, TINTA, 0.35), anchor="ls")
            elif estilo == "mate":
                formula, profundidad = latex(palabra, tam, color, display=False)
                pos = (round(cx), round(base + profundidad - formula.height))
                d._image.paste(formula, pos, formula)
            else:
                c = acento if estilo == "acento" else color
                px = cx
                for trozo, f in _piezas(texto_palabra, estilo, tam):
                    d.text((px, base), trozo, font=f, fill=c, anchor="ls")
                    px += f.getlength(trozo)
            cx += ancho_palabra + (espacio if palabra.endswith(" ") else 0)
    return len(lineas) * alto_linea


# ------------------------------------------------------------------ elementos


class Elemento:
    id = None

    def dibujar(self, img, d):
        raise NotImplementedError


class Texto(Elemento):
    """Texto con marcado. Con ancho, se parte en varias líneas."""

    def __init__(self, x, y, texto, tam=40, color=TINTA, acento=AZUL, ancho=None, alinear="izq", id=None):
        self.x, self.y, self.texto, self.tam = x, y, texto, tam
        self.color, self.acento, self.ancho, self.alinear, self.id = color, acento, ancho, alinear, id

    def dibujar(self, img, d):
        dibujar_texto(d, self.x, self.y, self.texto, self.tam, self.color, self.acento, self.ancho, self.alinear)


class Formula(Elemento):
    """Una ecuación en LaTeX (modo display, sin los $). y es el borde de arriba; alinear:
    'izq' o 'centro' (x es entonces el centro). Partes de color con \\textcolor{azul}{W}
    (colores: tinta, gris, azul, verde, naranja, morado, rojo). Para fórmulas dentro de
    una frase, usa $...$ en el marcado de Texto."""

    def __init__(self, x, y, tex, tam=48, color=TINTA, alinear="izq", id=None):
        self.x, self.y, self.tex, self.tam, self.color, self.alinear, self.id = x, y, tex, tam, color, alinear, id

    def tamano(self):
        return latex(self.tex, self.tam, self.color)[0].size

    def dibujar(self, img, d):
        formula, _ = latex(self.tex, self.tam, self.color)
        x = self.x - formula.width / 2 if self.alinear == "centro" else self.x
        img.paste(formula, (round(x), round(self.y)), formula)


class Vineta(Elemento):
    """Una viñeta: un punto de color y texto con marcado a su derecha."""

    def __init__(self, x, y, texto, tam=40, color=AZUL, ancho=1600, numero=None, id=None):
        self.x, self.y, self.texto, self.tam, self.color, self.ancho, self.numero, self.id = (
            x, y, texto, tam, color, ancho, numero, id,
        )

    def dibujar(self, img, d):
        cy = self.y + self.tam * 0.72
        if self.numero is None:
            r = self.tam * 0.18
            d.ellipse((self.x - r, cy - r, self.x + r, cy + r), fill=self.color)
        else:
            r = self.tam * 0.55
            d.ellipse((self.x - r, cy - r, self.x + r, cy + r), fill=self.color)
            d.text((self.x, cy), str(self.numero), font=fuente("negrita", round(self.tam * 0.7)), fill="white", anchor="mm")
        dibujar_texto(d, self.x + self.tam * 1.0, self.y, self.texto, self.tam, TINTA, self.color, self.ancho)


def vinetas(x, y, textos, tam=40, sep=None, color=AZUL, ancho=1600, numeradas=False):
    """Una Vineta por texto, una debajo de otra. Devuelve la lista (para repartirla entre pasos)."""
    sep = sep or tam * 2.0
    salida, yy = [], y
    for i, t in enumerate(textos):
        salida.append(Vineta(x, yy, t, tam, color, ancho, numero=i + 1 if numeradas else None))
        _, alto = medir_texto(t, tam, ancho)
        yy += max(sep, alto + tam * 0.6)
    return salida


class Caja(Elemento):
    """Rectángulo redondeado con texto centrado. relleno=None usa un tinte del color."""

    def __init__(self, x, y, w, h, texto="", color=AZUL, tam=36, relleno=None, subtitulo=None, borde=4, id=None):
        self.x, self.y, self.w, self.h, self.texto, self.color, self.tam = x, y, w, h, texto, color, tam
        self.relleno = relleno or tinte(color)
        self.subtitulo, self.borde, self.id = subtitulo, borde, id

    def dibujar(self, img, d):
        d.rounded_rectangle((self.x, self.y, self.x + self.w, self.y + self.h), radius=18,
                            fill=self.relleno, outline=self.color, width=self.borde)
        tam_sub = round(self.tam * 0.62)
        _, alto = medir_texto(self.texto, self.tam, self.w - 30)
        alto_total = alto + (tam_sub * 1.5 if self.subtitulo else 0)
        y = self.y + (self.h - alto_total) / 2
        dibujar_texto(d, self.x + self.w / 2, y, self.texto, self.tam, TINTA, self.color, self.w - 30, "centro")
        if self.subtitulo:
            dibujar_texto(d, self.x + self.w / 2, y + alto + tam_sub * 0.2, self.subtitulo, tam_sub, GRIS,
                          self.color, self.w - 30, "centro")


class Circulo(Elemento):
    """Círculo con un texto corto al centro. Un texto entre $...$ se compone con LaTeX ($x_1$)."""

    def __init__(self, cx, cy, r, texto="", color=AZUL, tam=30, relleno=None, color_texto="white", id=None):
        self.cx, self.cy, self.r, self.texto, self.color, self.tam = cx, cy, r, texto, color, tam
        self.relleno = relleno or color
        self.color_texto, self.id = color_texto, id

    def dibujar(self, img, d):
        d.ellipse((self.cx - self.r, self.cy - self.r, self.cx + self.r, self.cy + self.r),
                  fill=self.relleno, outline=self.color, width=4)
        if self.texto.startswith("$") and self.texto.endswith("$"):
            color = "#FFFFFF" if self.color_texto == "white" else self.color_texto
            formula, _ = latex(self.texto[1:-1], self.tam, color, display=False)
            img.paste(formula, (round(self.cx - formula.width / 2), round(self.cy - formula.height / 2)), formula)
        elif self.texto:
            d.text((self.cx, self.cy), self.texto, font=fuente("negrita", self.tam), fill=self.color_texto, anchor="mm")


class Flecha(Elemento):
    """Flecha de (x1, y1) a (x2, y2), con una etiqueta opcional junto a su punto medio."""

    def __init__(self, x1, y1, x2, y2, color=GRIS, grosor=6, etiqueta=None, tam=28, lado="arriba", id=None):
        self.p1, self.p2, self.color, self.grosor = (x1, y1), (x2, y2), color, grosor
        self.etiqueta, self.tam, self.lado, self.id = etiqueta, tam, lado, id

    def dibujar(self, img, d):
        (x1, y1), (x2, y2) = self.p1, self.p2
        ang = math.atan2(y2 - y1, x2 - x1)
        punta = self.grosor * 3.2
        xb, yb = x2 - punta * math.cos(ang), y2 - punta * math.sin(ang)
        d.line((x1, y1, xb, yb), fill=self.color, width=self.grosor)
        izq = (xb + punta * 0.6 * math.sin(ang), yb - punta * 0.6 * math.cos(ang))
        der = (xb - punta * 0.6 * math.sin(ang), yb + punta * 0.6 * math.cos(ang))
        d.polygon([(x2, y2), izq, der], fill=self.color)
        if self.etiqueta:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            _, alto = medir_texto(self.etiqueta, self.tam)
            if self.lado == "arriba":
                my -= alto + 8
            elif self.lado == "abajo":
                my += 10
            else:
                mx += 20 if self.lado == "derecha" else -20
                my -= alto / 2
            alinear = "centro" if self.lado in ("arriba", "abajo") else "izq"
            if self.lado == "izquierda":
                ancho, _ = medir_texto(self.etiqueta, self.tam)
                mx -= ancho
            dibujar_texto(d, mx, my, self.etiqueta, self.tam, self.color, self.color, alinear=alinear)


class Linea(Elemento):
    def __init__(self, x1, y1, x2, y2, color=GRIS_CLARO, grosor=4, id=None):
        self.p, self.color, self.grosor, self.id = (x1, y1, x2, y2), color, grosor, id

    def dibujar(self, img, d):
        d.line(self.p, fill=self.color, width=self.grosor)


class Marco(Elemento):
    """Contorno para señalar una zona de la pantalla."""

    def __init__(self, x, y, w, h, color=NARANJA, grosor=6, id=None):
        self.caja, self.color, self.grosor, self.id = (x, y, x + w, y + h), color, grosor, id

    def dibujar(self, img, d):
        d.rounded_rectangle(self.caja, radius=14, outline=self.color, width=self.grosor)


def _colores_pygments(nombre):
    estilo = get_style_by_name(nombre)
    colores = {}
    for tok, defin in estilo:
        if defin["color"]:
            colores[tok] = "#" + defin["color"]
    return colores, estilo


_COLORES, _ESTILO = _colores_pygments("github-dark")


def _color_token(tok):
    while tok not in _COLORES and tok.parent is not None:
        tok = tok.parent
    return _COLORES.get(tok, "#E6EDF3")


class Codigo(Elemento):
    """Bloque de código Python con colores. resaltar: números de línea (desde 1) que
    quedan iluminados; las demás se atenúan."""

    def __init__(self, x, y, codigo, tam=30, ancho=None, resaltar=None, titulo=None, id=None):
        self.x, self.y, self.codigo, self.tam = x, y, codigo.strip("\n"), tam
        self.ancho, self.resaltar, self.titulo, self.id = ancho, resaltar, titulo, id

    def _lineas_coloreadas(self):
        lineas = [[]]
        for tok, valor in lex(self.codigo, PythonLexer()):
            for i, trozo in enumerate(valor.split("\n")):
                if i:
                    lineas.append([])
                if trozo:
                    lineas[-1].append((trozo, _color_token(tok)))
        if lineas and not lineas[-1]:
            lineas.pop()
        return lineas

    def tamano(self):
        f = fuente("codigo", self.tam)
        n = len(self.codigo.split("\n"))
        ancho = self.ancho or max(f.getlength(l) for l in self.codigo.split("\n")) + 2 * self.tam * 1.2
        alto = n * self.tam * 1.45 + 2 * self.tam * 0.9 + (self.tam * 1.3 if self.titulo else 0)
        return ancho, alto

    def dibujar(self, img, d):
        f = fuente("codigo", self.tam)
        ancho, alto = self.tamano()
        pad = self.tam * 1.2
        d.rounded_rectangle((self.x, self.y, self.x + ancho, self.y + alto), radius=16, fill=FONDO_CODIGO)
        y = self.y + self.tam * 0.9
        if self.titulo:
            d.text((self.x + pad, y), self.titulo, font=fuente("normal", round(self.tam * 0.75)), fill="#8B949E")
            y += self.tam * 1.3
        paso = self.tam * 1.45
        for n, linea in enumerate(self._colorear(), start=1):
            activa = self.resaltar is None or n in self.resaltar
            if self.resaltar and activa:
                d.rectangle((self.x + 6, y - self.tam * 0.2, self.x + ancho - 6, y + self.tam * 1.25), fill="#1F2A3A")
            x = self.x + pad
            for trozo, color in linea:
                c = color if activa else mezclar(color, FONDO_CODIGO, 0.45)
                d.text((x, y), trozo, font=f, fill=c)
                x += f.getlength(trozo)
            y += paso

    def _colorear(self):
        return self._lineas_coloreadas()


class Terminal(Codigo):
    """Salida de consola: '$ ' en cian, '+' en verde, '-' en rojo, '@@' en morado."""

    def _colorear(self):
        lineas = []
        for linea in self.codigo.split("\n"):
            if linea.startswith("$ "):
                lineas.append([("$ ", "#7EE787"), (linea[2:], "#E6EDF3")])
                continue
            if linea.startswith("+") and not linea.startswith("+++"):
                c = "#7EE787"
            elif linea.startswith("-") and not linea.startswith("---"):
                c = "#FF7B72"
            elif linea.startswith("@@"):
                c = "#D2A8FF"
            else:
                c = "#9DA7B3"
            lineas.append([(linea, c)])
        return lineas


class Tabla(Elemento):
    """filas[0] es el encabezado. Las celdas aceptan marcado."""

    def __init__(self, x, y, filas, anchos, tam=32, color=AZUL, alto_fila=None, id=None):
        self.x, self.y, self.filas, self.anchos, self.tam, self.color, self.id = x, y, filas, anchos, tam, color, id
        self.alto_fila = alto_fila or tam * 2.1

    def dibujar(self, img, d):
        y = self.y
        total = sum(self.anchos)
        for i, fila in enumerate(self.filas):
            altos = [medir_texto(c, self.tam, w - 40)[1] for c, w in zip(fila, self.anchos)]
            h = max(self.alto_fila, max(altos) + self.tam * 0.8)
            if i == 0:
                d.rounded_rectangle((self.x, y, self.x + total, y + h), radius=10, fill=tinte(self.color, 0.8))
            x = self.x
            for celda, w, a in zip(fila, self.anchos, altos):
                dibujar_texto(d, x + 20, y + (h - a) / 2, f"**{celda}**" if i == 0 else celda, self.tam, TINTA,
                              self.color, w - 40)
                x += w
            y += h
            if i < len(self.filas) - 1:
                d.line((self.x, y, self.x + total, y), fill=GRIS_CLARO, width=2)


class Matriz(Elemento):
    """Una cuadrícula de filas × columnas celdas, para mostrar formas de arreglos."""

    def __init__(self, x, y, filas, columnas, celda=44, color=AZUL, etiqueta=None, valores=None, tam=26, id=None):
        self.x, self.y, self.f, self.c, self.celda, self.color = x, y, filas, columnas, celda, color
        self.etiqueta, self.valores, self.tam, self.id = etiqueta, valores, tam, id

    def dibujar(self, img, d):
        for i in range(self.f):
            for j in range(self.c):
                x0, y0 = self.x + j * self.celda, self.y + i * self.celda
                d.rectangle((x0, y0, x0 + self.celda, y0 + self.celda), fill=tinte(self.color, 0.75),
                            outline=self.color, width=2)
                if self.valores is not None:
                    d.text((x0 + self.celda / 2, y0 + self.celda / 2), str(self.valores[i][j]),
                           font=fuente("normal", self.tam), fill=TINTA, anchor="mm")
        if self.etiqueta:
            dibujar_texto(d, self.x + self.c * self.celda / 2, self.y + self.f * self.celda + 12, self.etiqueta,
                          self.tam + 4, TINTA, self.color, alinear="centro")


class Pixeles(Elemento):
    """Una imagen en escala de grises dibujada celda por celda: 0 es blanco y vmax es TINTA.
    numeros=True escribe el valor en cada celda (para mostrar que una imagen es un arreglo)."""

    def __init__(self, x, y, valores, celda=40, vmax=16, numeros=False, etiqueta=None, color=AZUL, tam=24, id=None):
        self.x, self.y, self.v, self.celda, self.vmax = x, y, np.asarray(valores, float), celda, vmax
        self.numeros, self.etiqueta, self.color, self.tam, self.id = numeros, etiqueta, color, tam, id

    def dibujar(self, img, d):
        filas, columnas = self.v.shape
        for i in range(filas):
            for j in range(columnas):
                t = min(max(self.v[i, j] / self.vmax, 0), 1)
                x0, y0 = self.x + j * self.celda, self.y + i * self.celda
                d.rectangle((x0, y0, x0 + self.celda, y0 + self.celda), fill=mezclar("#FFFFFF", TINTA, t))
                if self.numeros:
                    d.text((x0 + self.celda / 2, y0 + self.celda / 2), f"{self.v[i, j]:g}",
                           font=fuente("normal", self.tam), fill=TINTA if t < 0.5 else "white", anchor="mm")
        d.rectangle((self.x, self.y, self.x + columnas * self.celda, self.y + filas * self.celda),
                    outline=self.color, width=3)
        if self.etiqueta:
            dibujar_texto(d, self.x + columnas * self.celda / 2, self.y + filas * self.celda + 12, self.etiqueta,
                          self.tam + 4, TINTA, self.color, alinear="centro")


class Grafica(Elemento):
    """La gráfica de f en [xmin, xmax] dentro de la caja (x, y, w, h), con ejes en 0.
    puntos: [(x, etiqueta)] o [(x, etiqueta, "abajo")] marca f(x) con un punto.
    rectas: [(a, color)] dibuja la tangente a f en x = a (pendiente numérica).
    camino: [x0, x1, ...] los pasos de un descenso de gradiente, como puntos sobre f unidos por segmentos.
    Lo que sale del rango [ymin, ymax] no se dibuja."""

    def __init__(self, x, y, w, h, f, xmin, xmax, ymin, ymax, color=AZUL, etiqueta=None, puntos=(), rectas=(),
                 camino=(), id=None):
        self.caja = (x, y, w, h)
        self.f, self.rx, self.ry, self.color, self.etiqueta, self.puntos, self.id = (
            f, (xmin, xmax), (ymin, ymax), color, etiqueta, puntos, id,
        )
        self.rectas, self.camino = rectas, camino

    def _a_pantalla(self, xv, yv):
        x, y, w, h = self.caja
        (x0, x1), (y0, y1) = self.rx, self.ry
        return x + (xv - x0) / (x1 - x0) * w, y + h - (yv - y0) / (y1 - y0) * h

    def _curva(self, d, g, color, grosor):
        """Dibuja g en el rango x, cortada en tramos donde se sale del rango y."""
        tramo = []
        for xv in np.linspace(*self.rx, 400):
            yv = g(xv)
            if self.ry[0] <= yv <= self.ry[1]:
                tramo.append(self._a_pantalla(xv, yv))
            else:
                if len(tramo) > 1:
                    d.line(tramo, fill=color, width=grosor, joint="curve")
                tramo = []
        if len(tramo) > 1:
            d.line(tramo, fill=color, width=grosor, joint="curve")

    def dibujar(self, img, d):
        x, y, w, h = self.caja
        d.rounded_rectangle((x - 20, y - 20, x + w + 20, y + h + 20), radius=16, fill="white", outline=GRIS_CLARO, width=2)
        ex0, ey = self._a_pantalla(self.rx[0], 0)
        ex1, _ = self._a_pantalla(self.rx[1], 0)
        d.line((ex0, ey, ex1, ey), fill=GRIS, width=3)
        ejx, ey0 = self._a_pantalla(0, self.ry[0])
        _, ey1 = self._a_pantalla(0, self.ry[1])
        d.line((ejx, ey0, ejx, ey1), fill=GRIS, width=3)
        self._curva(d, self.f, self.color, 8)
        for a, color in self.rectas:
            m = (self.f(a + 1e-5) - self.f(a - 1e-5)) / 2e-5
            self._curva(d, lambda xv: self.f(a) + m * (xv - a), color, 5)
            px, py = self._a_pantalla(a, self.f(a))
            d.ellipse((px - 11, py - 11, px + 11, py + 11), fill=color)
        visibles = [self._a_pantalla(xv, self.f(xv)) if self.ry[0] <= self.f(xv) <= self.ry[1] else None
                    for xv in self.camino]
        for p, q in zip(visibles, visibles[1:]):
            if p and q:
                d.line((p, q), fill=NARANJA, width=3)
        for i, p in enumerate(visibles):
            if p:
                r = 11 if i == 0 else 8
                d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=NARANJA if i else ROJO)
        for xv, texto, *lado in self.puntos:
            px, py = self._a_pantalla(xv, self.f(xv))
            d.ellipse((px - 11, py - 11, px + 11, py + 11), fill=NARANJA)
            if texto:
                dy = 15 if lado == ["abajo"] else -55
                dibujar_texto(d, px + 22, py + dy, texto, 30, NARANJA, NARANJA)
        if self.etiqueta:
            dibujar_texto(d, x + w / 2, y + h + 30, self.etiqueta, 30, TINTA, self.color, alinear="centro")


class _Ejes(Elemento):
    """Base de Histograma y Dispersion: un panel blanco con ejes, marcas y leyenda en la caja (x, y, w, h)."""

    def _preparar(self, x, y, w, h, rx, ry, marcas_x, marcas_y, etiqueta_x, etiqueta_y, nombres, colores):
        self.caja, self.rx, self.ry = (x, y, w, h), rx, ry
        self.marcas_x, self.marcas_y, self.etiqueta_x, self.etiqueta_y = marcas_x, marcas_y, etiqueta_x, etiqueta_y
        self.nombres, self.colores = nombres, colores

    def _a_pantalla(self, xv, yv):
        x, y, w, h = self.caja
        (x0, x1), (y0, y1) = self.rx, self.ry
        return x + (xv - x0) / (x1 - x0) * w, y + h - (yv - y0) / (y1 - y0) * h

    def _fondo(self, d):
        x, y, w, h = self.caja
        d.rounded_rectangle((x - 90, y - 30, x + w + 30, y + h + 100), radius=16, fill="white",
                            outline=GRIS_CLARO, width=2)

    def _curva(self, d, g, color, grosor):
        """Dibuja g en el rango x, cortada en tramos donde se sale del rango y."""
        tramo = []
        for xv in np.linspace(*self.rx, 400):
            yv = g(xv)
            if self.ry[0] <= yv <= self.ry[1]:
                tramo.append(self._a_pantalla(xv, yv))
            else:
                if len(tramo) > 1:
                    d.line(tramo, fill=color, width=grosor, joint="curve")
                tramo = []
        if len(tramo) > 1:
            d.line(tramo, fill=color, width=grosor, joint="curve")

    def _ejes(self, img, d):
        x, y, w, h = self.caja
        d.line((x, y + h, x + w, y + h), fill=GRIS, width=3)
        d.line((x, y, x, y + h), fill=GRIS, width=3)
        f = fuente("normal", 24)
        for v in self.marcas_x:
            px, _ = self._a_pantalla(v, self.ry[0])
            d.line((px, y + h, px, y + h + 10), fill=GRIS, width=3)
            d.text((px, y + h + 14), f"{v:g}", font=f, fill=GRIS, anchor="mt")
        for v in self.marcas_y:
            _, py = self._a_pantalla(self.rx[0], v)
            d.line((x - 10, py, x, py), fill=GRIS, width=3)
            d.text((x - 16, py), f"{v:g}", font=f, fill=GRIS, anchor="rm")
        if self.etiqueta_x:
            dibujar_texto(d, x + w / 2, y + h + 48, self.etiqueta_x, 28, TINTA, alinear="centro")
        if self.etiqueta_y:
            capa = Image.new("RGBA", (round(h), 50), (0, 0, 0, 0))
            dibujar_texto(ImageDraw.Draw(capa), h / 2, 4, self.etiqueta_y, 28, TINTA, alinear="centro")
            capa = capa.rotate(90, expand=True)
            img.paste(capa, (round(x - 88), round(y)), capa)
        if self.nombres:
            largo = max(fuente("normal", 28).getlength(n) for n in self.nombres)
            lx, ly = min(x + w - 250, x + w - 50 - largo), y + 6
            for nombre, color in zip(self.nombres, self.colores):
                d.rectangle((lx, ly + 6, lx + 26, ly + 32), fill=color)
                d.text((lx + 38, ly), nombre, font=fuente("normal", 28), fill=TINTA)
                ly += 42


def _con_transparencia(img, dibujar):
    """Dibuja con colores RGBA (semitransparentes) sobre img, que es RGB."""
    capa = Image.new("RGBA", img.size, (0, 0, 0, 0))
    dibujar(ImageDraw.Draw(capa))
    img.paste(Image.alpha_composite(img.convert("RGBA"), capa).convert("RGB"))


class Histograma(_Ejes):
    """Barras de conteos sobre los intervalos definidos por bordes (len(bordes) = len(conteos) + 1).
    series: una lista de conteos por grupo; con varios grupos, las barras se superponen semitransparentes."""

    def __init__(self, x, y, w, h, bordes, series, colores=(GRIS,), nombres=None, ymax=None, marcas_x=(),
                 marcas_y=(), etiqueta_x=None, etiqueta_y=None, id=None):
        ymax = ymax or max(max(s) for s in series) * 1.1
        self._preparar(x, y, w, h, (bordes[0], bordes[-1]), (0, ymax), marcas_x, marcas_y, etiqueta_x, etiqueta_y,
                       nombres, colores)
        self.bordes, self.series, self.id = bordes, series, id

    def dibujar(self, img, d):
        self._fondo(d)
        alfa = 255 if len(self.series) == 1 else 150

        def barras(dc):
            for conteos, color in zip(self.series, self.colores):
                for a, b, n in zip(self.bordes, self.bordes[1:], conteos):
                    if n:
                        x0, y0 = self._a_pantalla(a, n)
                        x1, y1 = self._a_pantalla(b, 0)
                        dc.rectangle((x0 + 1, y0, x1 - 1, y1), fill=rgb(color) + (alfa,))

        _con_transparencia(img, barras)
        self._ejes(img, ImageDraw.Draw(img))


class Dispersion(_Ejes):
    """Un punto por par (x, y). series: [(xs, ys)] por grupo, con su color.
    tendencia=True dibuja la recta de mínimos cuadrados de cada grupo.
    cuadrantes=True marca las medias (de todos los puntos) y el signo de (x − x̄)(y − ȳ) en cada cuadrante.
    rectas: [(m, b, color)] dibuja rectas y = m·x + b elegidas a mano (por ejemplo, modelos candidatos).
    curvas: [(f, color)] o [(f, color, grosor)] dibuja y = f(x) en todo el rango x (un polinomio ajustado, una
    curva de error interpolada); lo que sale del rango y no se dibuja. Van debajo de los puntos."""

    def __init__(self, x, y, w, h, series, colores=(GRIS,), nombres=None, rx=(0, 1), ry=(0, 1), marcas_x=(),
                 marcas_y=(), etiqueta_x=None, etiqueta_y=None, tendencia=False, cuadrantes=False, radio=6,
                 rectas=(), curvas=(), id=None):
        self._preparar(x, y, w, h, rx, ry, marcas_x, marcas_y, etiqueta_x, etiqueta_y, nombres, colores)
        self.series, self.tendencia, self.cuadrantes, self.radio, self.id = series, tendencia, cuadrantes, radio, id
        self.rectas, self.curvas = rectas, curvas

    def dibujar(self, img, d):
        self._fondo(d)
        x, y, w, h = self.caja
        if self.cuadrantes:
            xs = np.concatenate([np.asarray(s[0], float) for s in self.series])
            ys = np.concatenate([np.asarray(s[1], float) for s in self.series])
            mx, my = self._a_pantalla(xs.mean(), ys.mean())
            for i in range(0, round(h), 24):
                d.line((mx, y + i, mx, y + min(i + 12, h)), fill=NARANJA, width=3)
            for i in range(0, round(w), 24):
                d.line((x + i, my, x + min(i + 12, w), my), fill=NARANJA, width=3)
            f = fuente("respaldo_negrita", 64)  # Noto Sans no trae el signo −
            for sx, sy, signo, color in [(1, -1, "+", VERDE), (-1, 1, "+", VERDE), (-1, -1, "−", ROJO),
                                         (1, 1, "−", ROJO)]:
                d.text((mx + sx * w / 4, my + sy * h / 4), signo, font=f, fill=color, anchor="mm")
        for f, color, *grosor in self.curvas:
            self._curva(d, f, color, grosor[0] if grosor else 6)
        r = self.radio

        def puntos(dc):
            for (xs, ys), color in zip(self.series, self.colores):
                for xv, yv in zip(xs, ys):
                    px, py = self._a_pantalla(xv, yv)
                    dc.ellipse((px - r, py - r, px + r, py + r), fill=rgb(color) + (190,))

        _con_transparencia(img, puntos)
        d = ImageDraw.Draw(img)
        if self.tendencia:
            for (xs, ys), color in zip(self.series, self.colores):
                xs, ys = np.asarray(xs, float), np.asarray(ys, float)
                m, b = np.polyfit(xs, ys, 1)
                tramo = (xs.min(), xs.max())
                d.line([self._a_pantalla(v, m * v + b) for v in tramo], fill=mezclar(color, TINTA, 0.3), width=7)
        for m, b, color in self.rectas:
            d.line([self._a_pantalla(v, m * v + b) for v in self.rx], fill=color, width=7)
        self._ejes(img, d)


class Contornos(_Ejes):
    """Mapa de curvas de nivel de f(a, b) (vectorizada sobre arreglos) en la caja (x, y, w, h), sobre rx × ry.
    niveles: los valores de las curvas; entre curvas, bandas más oscuras cuanto más bajo es f (el valle).
    caminos: [(puntos, color)], cada uno una lista de (a, b) unidos por segmentos; el primero se marca más grande.
    flechas: [(a, b, da, db)] en unidades de los datos (por ejemplo, el gradiente).
    marcas: [(a, b, color, texto)] un punto con su etiqueta (el mínimo, la línea base).
    igual=True usa la misma escala en los dos ejes (ajusta h para que cuadre con rx y ry)."""

    def __init__(self, x, y, w, h, f, rx, ry, niveles, color=AZUL, caminos=(), flechas=(), marcas=(), marcas_x=(),
                 marcas_y=(), etiqueta_x=None, etiqueta_y=None, igual=False, id=None):
        if igual:
            h = w * (ry[1] - ry[0]) / (rx[1] - rx[0])
        self._preparar(x, y, w, h, rx, ry, marcas_x, marcas_y, etiqueta_x, etiqueta_y, None, ())
        self.f, self.niveles, self.color = f, np.asarray(niveles, float), color
        self.caminos, self.flechas, self.marcas, self.id = caminos, flechas, marcas, id

    def _mapa(self):
        x, y, w, h = self.caja
        w, h = round(w), round(h)
        a = np.linspace(self.rx[0], self.rx[1], w)
        b = np.linspace(self.ry[1], self.ry[0], h)  # la fila 0 de la imagen es el b más grande
        A, B = np.meshgrid(a, b)
        banda = np.digitize(self.f(A, B), self.niveles)  # 0 dentro de la curva más baja
        t = 0.45 + 0.52 * banda / len(self.niveles)
        base, blanco = np.array(rgb(self.color), float), np.array([255.0, 255.0, 255.0])
        pix = base + (blanco - base) * t[..., None]
        borde = np.zeros((h, w), bool)
        borde[:, 1:] |= banda[:, 1:] != banda[:, :-1]
        borde[1:, :] |= banda[1:, :] != banda[:-1, :]
        grueso = borde.copy()
        grueso[:, 1:] |= borde[:, :-1]
        grueso[1:, :] |= borde[:-1, :]
        grueso[:, :-1] |= borde[:, 1:]
        pix[grueso] = np.array(rgb(mezclar(self.color, TINTA, 0.35)), float)
        return Image.fromarray(pix.round().astype(np.uint8), "RGB")

    def dibujar(self, img, d):
        self._fondo(d)
        x, y, w, h = self.caja
        img.paste(self._mapa(), (round(x), round(y)))
        d = ImageDraw.Draw(img)

        def dentro(p):
            return x <= p[0] <= x + w and y <= p[1] <= y + h

        for puntos, color in self.caminos:
            pts = [self._a_pantalla(a, b) for a, b in puntos]
            for p, q in zip(pts, pts[1:]):
                if dentro(p) and dentro(q):
                    d.line((p, q), fill=color, width=4)
            for i, p in enumerate(pts):
                if dentro(p):
                    r = 11 if i == 0 else 6
                    d.ellipse((p[0] - r, p[1] - r, p[0] + r, p[1] + r), fill=color, outline="white", width=2)
        for a, b, da, db in self.flechas:
            (x1, y1), (x2, y2) = self._a_pantalla(a, b), self._a_pantalla(a + da, b + db)
            if math.hypot(x2 - x1, y2 - y1) > 12:
                Flecha(x1, y1, x2, y2, ROJO, 4).dibujar(img, d)
        for a, b, color, texto in self.marcas:
            px, py = self._a_pantalla(a, b)
            d.ellipse((px - 13, py - 13, px + 13, py + 13), fill=color, outline="white", width=3)
            if texto:
                dibujar_texto(d, px + 22, py - 18, texto, 28, color, color)
        self._ejes(img, d)


class Regiones(_Ejes):
    """Las regiones de decisión de un clasificador en la caja (x, y, w, h), sobre rx × ry.
    predecir(A, B): la clase (0, 1, ...) de cada punto de las grillas A y B; la región de cada clase se pinta
    con un tinte de su color en colores. series: [(xs, ys)] los puntos de cada clase, con esos mismos colores.
    resolucion: los píxeles por celda de la grilla (más grande es más rápido y más escalonado)."""

    def __init__(self, x, y, w, h, predecir, rx, ry, colores, series=(), nombres=None, marcas_x=(), marcas_y=(),
                 etiqueta_x=None, etiqueta_y=None, radio=5, resolucion=3, id=None):
        self._preparar(x, y, w, h, rx, ry, marcas_x, marcas_y, etiqueta_x, etiqueta_y, nombres, colores)
        self.predecir, self.series, self.radio, self.resolucion, self.id = predecir, series, radio, resolucion, id

    def _mapa(self):
        _, _, w, h = self.caja
        w, h = round(w), round(h)
        nw, nh = math.ceil(w / self.resolucion), math.ceil(h / self.resolucion)
        a = np.linspace(self.rx[0], self.rx[1], nw)
        b = np.linspace(self.ry[1], self.ry[0], nh)  # la fila 0 de la imagen es el b más grande
        A, B = np.meshgrid(a, b)
        clase = np.asarray(self.predecir(A, B), int)
        paleta = np.array([rgb(tinte(c, 0.72)) for c in self.colores], np.uint8)
        return Image.fromarray(paleta[clase], "RGB").resize((w, h), Image.NEAREST)

    def dibujar(self, img, d):
        self._fondo(d)
        x, y, _, _ = self.caja
        img.paste(self._mapa(), (round(x), round(y)))
        r = self.radio

        def puntos(dc):
            for (xs, ys), color in zip(self.series, self.colores):
                for xv, yv in zip(xs, ys):
                    px, py = self._a_pantalla(xv, yv)
                    dc.ellipse((px - r, py - r, px + r, py + r), fill=rgb(color) + (220,))

        _con_transparencia(img, puntos)
        self._ejes(img, ImageDraw.Draw(img))


class Barras(Elemento):
    """Barras verticales, una por categoría, con su valor encima y su nombre debajo, en la caja (x, y, w, h).
    vmax es el valor de una barra que llena la caja. colores: uno para todas o uno por barra."""

    def __init__(self, x, y, w, h, nombres, valores, colores=AZUL, vmax=1.0, formato="{:.2f}", tam=30, id=None):
        self.caja, self.nombres, self.valores, self.vmax, self.formato, self.tam, self.id = (
            (x, y, w, h), nombres, valores, vmax, formato, tam, id,
        )
        self.colores = [colores] * len(nombres) if isinstance(colores, str) else list(colores)

    def dibujar(self, img, d):
        x, y, w, h = self.caja
        n = len(self.nombres)
        ancho = w / n
        for i, (nombre, v, color) in enumerate(zip(self.nombres, self.valores, self.colores)):
            x0 = x + i * ancho + ancho * 0.15
            x1 = x + (i + 1) * ancho - ancho * 0.15
            alto = h * min(max(v / self.vmax, 0), 1)
            d.rounded_rectangle((x0, y + h - alto, x1, y + h), radius=8, fill=color)
            dibujar_texto(d, (x0 + x1) / 2, y + h - alto - self.tam * 1.5, self.formato.format(v), self.tam, TINTA,
                          alinear="centro")
            dibujar_texto(d, (x0 + x1) / 2, y + h + 12, nombre, self.tam, TINTA, color, alinear="centro")
        d.line((x, y + h, x + w, y + h), fill=GRIS, width=3)


# ------------------------------------------------------------- escenas y pasos


class Paso:
    """decir: la narración (también el subtítulo). mostrar: elementos que aparecen.
    ocultar: ids de elementos que desaparecen. pausa: segundos de silencio al final."""

    def __init__(self, decir, *mostrar, ocultar=(), pausa=0.5):
        self.decir, self.mostrar, self.ocultar, self.pausa = decir, mostrar, set(ocultar), pausa


class Escena:
    def __init__(self, titulo, *pasos):
        self.titulo, self.pasos, self.seccion = titulo, pasos, None


class Seccion:
    """Marcador en ESCENAS: las escenas siguientes muestran este nombre sobre su título."""

    def __init__(self, nombre):
        self.nombre = nombre


# ------------------------------------------------------------------- dibujo


def dibujar_cuadro(meta, escena, elementos, progreso):
    img = Image.new("RGB", (ANCHO, ALTO), FONDO)
    d = ImageDraw.Draw(img)
    if escena.titulo:
        y = 70
        if escena.seccion:
            d.text((100, y), escena.seccion.upper(), font=fuente("negrita", 26), fill=AZUL)
            y += 42
        d.rectangle((100, y + 8, 108, y + 70), fill=AZUL)
        d.text((130, y), escena.titulo, font=fuente("negrita", 58), fill=TINTA)
    for el in elementos:
        el.dibujar(img, d)
    d.text((100, ALTO - 50), meta["pie"], font=fuente("normal", 24), fill=GRIS, anchor="lm")
    d.rectangle((0, ALTO - 8, ANCHO, ALTO), fill=GRIS_CLARO)
    d.rectangle((0, ALTO - 8, round(ANCHO * progreso), ALTO), fill=AZUL)
    return img


def recorrer(escenas):
    """Genera (n_escena, escena, n_paso, paso, elementos visibles) para cada paso."""
    seccion = None
    escenas_reales = []
    for item in escenas:
        if isinstance(item, Seccion):
            seccion = item.nombre
        else:
            item.seccion = seccion
            escenas_reales.append(item)
    for ne, escena in enumerate(escenas_reales, start=1):
        visibles = []
        for np_, paso in enumerate(escena.pasos, start=1):
            visibles = [e for e in visibles if e.id not in paso.ocultar] + list(paso.mostrar)
            yield ne, escena, np_, paso, list(visibles)


# ---------------------------------------------------------------- narración

# Cómo debe sonar cada término cuando la voz en español lo lee mal. Se aplica solo
# al audio: los subtítulos conservan la escritura original. Revisa los casos nuevos
# transcribiendo el audio (ver CLAUDE.md, "Videos de las clases").
PRONUNCIACION = {
    "GitHub": "Guit-jab",
    "Git": "Guit",
    "git": "guit",
    "commits": "cómits",
    "commit": "cómit",
    "PyTorch": "Páitorch",
    "NumPy": "Nám-pai",
    "scikit-learn": "sáikit-lern",
    "sklearn": "sáikit-lern",
    "Colab": "Cólab",
    "Linear": "Líniar",
    "ReLU": "relú",
    "Sequential": "Sícuenshal",
    "Sigmoid": "Sígmoid",
    "Dataset": "Déitaset",
    "ArrayDataset": "Arréi Déitaset",
    "RunningMean": "Ráning Min",
    "TrainingLog": "Tréining Log",
    "PrettyTrainingLog": "Príti Tréining Log",
    "Module": "Módiul",
    "forward": "fóruard",
    "notebook": "nótbuk",
    "notebooks": "nótbuks",
    "deep learning": "dip lérning",
    "machine learning": "machín lérning",
    "batch": "bach",
    "batches": "báches",
    "diff": "dif",
    "IOAI": "í, o, a, í",
    "bug": "bag",
    "ValueError": "Váliu Érror",
    "GPU": "ge pe u",
    "Python": "Páiton",
    "init": "ínit",
    "getitem": "get-áitem",
    "repr": "réper",
    "call": "col",
    "parameters": "parámeters",
    "dunder": "dánder",
    "transformers": "transfórmers",
    "Whisper": "Uísper",
    "MÓLÓ": "Mólo",
    "Artix": "Ártix",
    "History": "Jístori",
    "untracked": "an-trákt",
    "add": "ad",
    "main": "méin",
    "hash": "jash",
    "reshape": "ri-chéip",
    "None": "Non",
    "argmax": "árg-max",
    "dtype": "de-táip",
    "ndim": "en-dim",
    "sum": "som",
    "copy": "cópi",
    "broadcasting": "bródcasting",
    "NaN": "nan",
    "unknown": "ánnoun",
    "Gentoo": "Yéntu",
    "group by": "grup bái",
    "Group by": "Grup bái",
    "unstack": "an-sták",
    "np where": "ene pe, uér",
    "read csv": "rid, ce ese ve",
    "ddof": "de de o efe",
    "Simpson": "Símpson",
    "eta": "éta",
    "pipeline": "páiplain",
    "recall": "rikól",
    "accuracy": "ákiurasi",
    "kNN": "ka ene ene",
    "stratify": "strátifai",
}


def para_voz(texto):
    texto = re.sub(r"[`*=]", "", texto)
    for termino in sorted(PRONUNCIACION, key=len, reverse=True):
        texto = re.sub(rf"(?<![\w-]){re.escape(termino)}(?![\w-])", PRONUNCIACION[termino], texto)
    return texto


def sintetizar(texto, voz, velocidad, cache):
    """Devuelve el audio de la frase como int16 mono a MUESTREO Hz (con caché en disco)."""
    import edge_tts

    texto_voz = para_voz(texto)
    clave = hashlib.sha1(f"{voz}|{velocidad}|{texto_voz}".encode()).hexdigest()[:16]
    mp3 = cache / f"{clave}.mp3"
    if not mp3.exists():
        for intento in range(4):
            try:
                asyncio.run(edge_tts.Communicate(texto_voz, voz, rate=velocidad).save(str(mp3) + ".tmp"))
                Path(str(mp3) + ".tmp").rename(mp3)
                break
            except Exception as e:  # la API de edge-tts falla de vez en cuando: reintentar
                if intento == 3:
                    raise
                print(f"  edge-tts falló ({e}); reintento")
    pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(mp3), "-f", "s16le", "-ac", "1", "-ar", str(MUESTREO), "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.int16)


# ----------------------------------------------------------------- subtítulos


def _hora_srt(t):
    ms = round(t * 1000)
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _frases(texto):
    texto = re.sub(r"[`*]|==", "", texto)
    partes = re.split(r"(?<=[.!?:;])\s+", texto.strip())
    frases = []
    for p in partes:  # trozos de hasta ~90 caracteres, cortando en comas si hace falta
        while len(p) > 90 and "," in p[20:90]:
            corte = p.rfind(",", 20, 90) + 1
            frases.append(p[:corte].strip())
            p = p[corte:].strip()
        frases.append(p)
    return [f for f in frases if f]


def subtitulos(segmentos):
    """segmentos: [(inicio, duración de la voz, texto)] → contenido de un .srt."""
    bloques = []
    for inicio, dur, texto in segmentos:
        frases = _frases(texto)
        total = sum(len(f) for f in frases)
        t = inicio
        for f in frases:
            d = dur * len(f) / total
            bloques.append((t, t + d, f))
            t += d
    return "\n".join(f"{i}\n{_hora_srt(a)} --> {_hora_srt(b)}\n{f}\n" for i, (a, b, f) in enumerate(bloques, 1))


# -------------------------------------------------------------------- armado


def hojas_de_contacto(pasos, imagenes, carpeta):
    """El último paso de cada escena, reducido, de a cuatro por imagen."""
    shutil.rmtree(carpeta, ignore_errors=True)
    carpeta.mkdir(parents=True)
    ultimos = [(f"e{p[0]:02d}_p{p[2]:02d}", img) for i, (p, img) in enumerate(zip(pasos, imagenes))
               if i == len(pasos) - 1 or pasos[i + 1][0] != p[0]]
    for k in range(0, len(ultimos), 4):
        hoja = Image.new("RGB", (ANCHO, ALTO), "black")
        d = ImageDraw.Draw(hoja)
        for j, (nombre, img) in enumerate(ultimos[k : k + 4]):
            x, y = (j % 2) * 964, (j // 2) * 544
            hoja.paste(img.resize((956, 536)), (x, y))
            d.text((x + 8, y + 6), nombre, font=fuente("negrita", 22), fill=ROJO)
        hoja.save(carpeta / f"hoja_{k // 4 + 1:02d}.png")


def cargar_guion(ruta):
    sys.modules.setdefault("motor", sys.modules[__name__])  # el guion importa estas mismas clases
    spec = importlib.util.spec_from_file_location("guion", ruta)
    modulo = importlib.util.module_from_spec(spec)
    sys.modules["guion"] = modulo
    spec.loader.exec_module(modulo)
    return modulo.META, modulo.ESCENAS


def construir(ruta, solo_cuadros=False, solo_escena=None):
    meta, escenas = cargar_guion(ruta)
    nombre = Path(ruta).stem
    destino = SALIDA / nombre
    cuadros = destino / "cuadros"
    shutil.rmtree(cuadros, ignore_errors=True)
    cuadros.mkdir(parents=True)
    cache = SALIDA / "cache_tts"
    cache.mkdir(parents=True, exist_ok=True)

    pasos = [p for p in recorrer(escenas) if solo_escena is None or p[0] == solo_escena]
    total = len(pasos)

    # 1. Narración de cada paso (y duraciones).
    voz, velocidad = meta.get("voz", "es-EC-AndreaNeural"), meta.get("velocidad", "+0%")
    entrada = 0.25  # silencio antes de cada frase, mientras aparece lo nuevo
    audios = []
    for i, (ne, escena, np_, paso, _) in enumerate(pasos):
        if solo_cuadros:
            audios.append(None)
            continue
        print(f"  voz {i + 1}/{total} (escena {ne}, paso {np_})", end="\r")
        audios.append(sintetizar(paso.decir, voz, velocidad, cache))
    print()

    # 2. Un PNG por paso, con la barra de progreso según el tiempo acumulado.
    durs = []
    for audio, (_, _, _, paso, _) in zip(audios, pasos):
        voz_s = 0 if audio is None else len(audio) / MUESTREO
        durs.append(entrada + voz_s + paso.pausa)
    acumulado = np.cumsum([0] + durs)
    imagenes = []
    for i, (ne, escena, np_, paso, visibles) in enumerate(pasos):
        img = dibujar_cuadro(meta, escena, visibles, acumulado[i + 1] / acumulado[-1] if acumulado[-1] else 1)
        img.save(cuadros / f"e{ne:02d}_p{np_:02d}.png")
        imagenes.append(img)
    hojas_de_contacto(pasos, imagenes, destino / "hojas")
    print(f"  {total} cuadros en {cuadros}")
    if solo_cuadros:
        return

    # 3. Pista de audio, alineada a cuadros enteros para que no se desincronice.
    pista, segs, cuadros_por_paso, t = [], [], [], 0.0
    for audio, dur, (_, _, _, paso, _) in zip(audios, durs, pasos):
        n_cuadros = round(dur * FPS)
        n_muestras = n_cuadros * MUESTREO // FPS
        bloque = np.zeros(n_muestras, np.int16)
        ini = round(entrada * MUESTREO)
        bloque[ini : ini + len(audio)] = audio[: n_muestras - ini]
        pista.append(bloque)
        segs.append((t + entrada, len(audio) / MUESTREO, paso.decir))
        cuadros_por_paso.append(n_cuadros)
        t += n_muestras / MUESTREO
    wav = destino / "narracion.wav"
    with wave.open(str(wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(MUESTREO)
        w.writeframes(np.concatenate(pista).tobytes())
    srt = destino / f"{nombre}.srt"
    srt.write_text(subtitulos(segs), encoding="utf-8")

    # 4. Video: cada paso es su imagen fija; al cambiar de imagen hay un fundido corto.
    mp4 = destino / f"{nombre}.mp4"
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{ANCHO}x{ALTO}",
         "-r", str(FPS), "-i", "-", "-i", str(wav), "-i", str(srt),
         "-map", "0:v", "-map", "1:a", "-map", "2:s",
         "-c:v", "libx264", "-preset", "veryfast", "-tune", "stillimage", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "128k", "-c:s", "mov_text", "-metadata:s:s:0", "language=spa",
         "-movflags", "+faststart", str(mp4)],
        stdin=subprocess.PIPE,
    )
    anterior = None
    for i, (img, n) in enumerate(zip(imagenes, cuadros_por_paso)):
        print(f"  video {i + 1}/{total}", end="\r")
        cambio_escena = i == 0 or pasos[i][0] != pasos[i - 1][0]
        fundido = min(n, round((0.5 if cambio_escena else 0.3) * FPS)) if anterior is not None else 0
        for k in range(fundido):
            ffmpeg.stdin.write(Image.blend(anterior, img, (k + 1) / (fundido + 1)).tobytes())
        fijo = img.tobytes()
        for _ in range(n - fundido):
            ffmpeg.stdin.write(fijo)
        anterior = img
    ffmpeg.stdin.close()
    if ffmpeg.wait() != 0:
        sys.exit("ffmpeg falló")
    wav.unlink()
    minutos, segundos = divmod(round(t), 60)
    print(f"\n  listo: {mp4} ({minutos}:{segundos:02d})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("guion", help="archivo de videos/guiones/")
    parser.add_argument("--solo-cuadros", action="store_true", help="solo los PNG, sin narración ni video")
    parser.add_argument("--escena", type=int, help="construir solo esta escena (desde 1)")
    args = parser.parse_args()
    construir(args.guion, args.solo_cuadros, args.escena)
