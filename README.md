## Entrenamiento
# Nivel Intermedio: Cronograma del Curso de Machine Learning
### 7 de octubre de 2026 → 30 de junio de 2027 (IOAI 2027: Singapur, 4–10 de julio)
Plataforma: [ioai.artix.tech](https://ioai.artix.tech/) · se desarrolla en paralelo con el Nivel Inicial (octubre–diciembre)
 
---

## Antes de empezar: GitHub

El curso usa dos lugares en GitHub:

- **Este repositorio** ([github.com/melvinpqbsc/Alto_Rendimiento_IA](https://github.com/melvinpqbsc/Alto_Rendimiento_IA)), público: cronograma, guía del estudiante, notebooks de clase y enunciados de los trabajos. Lo lees y abres sus notebooks en Colab, pero no guardas nada en él.
- **Tu repositorio de entregas**, `Alto-Rendimiento-IA/entregas-<tu-usuario>`, en la [organización del curso](https://github.com/Alto-Rendimiento-IA): es **privado** (solo lo ven tú y el profesor) y lo usas todo el año. Ahí guardas y entregas tus trabajos.

*Git* es el programa que guarda la historia de cambios de los archivos; *GitHub* es el sitio donde viven los repositorios en línea. Para este curso no necesitas instalar nada: todo se hace desde el navegador con GitHub y Google Colab.

### Organización del repositorio

```
Alto_Rendimiento_IA/
├── README.md                  ← este cronograma
├── Guia_del_estudiante.md     ← reglas del curso
├── requirements.txt           ← versiones de las librerías (ver Entorno de trabajo)
├── verificar_entorno.ipynb    ← se ejecuta en la primera clase
├── Nivel_Inicial/
│   ├── clases/                ← notebooks de cada clase
│   └── tareas/                ← enunciados de los trabajos
├── Nivel_Intermedio/
│   ├── clases/
│   └── tareas/
└── herramientas/              ← scripts del profesor
```

Trabaja solo en la carpeta de tu nivel.

### Configuración (una sola vez, antes de la primera clase)

1. **Crea una cuenta** en [github.com](https://github.com/). Elige un nombre de usuario que puedas mostrar. Envíaselo al profesor.
2. **Acepta la invitación a tu repositorio de entregas.** Cuando el profesor lo crea, te llega una invitación a `Alto-Rendimiento-IA/entregas-<tu-usuario>` por correo y en [github.com/notifications](https://github.com/notifications). Acéptala antes de 7 días (después caduca y hay que pedir otra).
3. **Conecta Colab con GitHub**: en [Colab](https://colab.research.google.com/), *Archivo → Abrir notebook → GitHub*, marca **Incluir repositorios privados** y acepta la autorización.

### Clases

Cada notebook de clase tiene arriba un botón **Abrir en Colab**. También puedes abrirlos con *Archivo → Abrir notebook → GitHub* y el repositorio `melvinpqbsc/Alto_Rendimiento_IA`. Si quieres conservar tus notas, *Archivo → Guardar una copia en Drive*.

### Trabajos prácticos

1. **Abre el enunciado** con su botón **Abrir en Colab**, en la carpeta `tareas/` de tu nivel.
2. **Guárdalo en tu repositorio de entregas apenas empieces**: *Archivo → Guardar una copia en GitHub*, repositorio `Alto-Rendimiento-IA/entregas-<tu-usuario>`, rama `main`, y como ruta la carpeta del trabajo más el nombre del archivo: `P01/P01_autograd_desde_cero.ipynb`. Escribe un mensaje que diga qué hiciste ("ejercicio 3 resuelto").
3. **Sigue trabajando desde tu copia**: de ahí en adelante ábrela con *Archivo → Abrir notebook → GitHub* (con **Incluir repositorios privados** marcado) y guarda con *Guardar una copia en GitHub* en la misma ruta. Cada guardado es un *commit*: guarda seguido.
4. **Entrega.** No hay que enviar nada: al inicio de la clase el profesor copia todos los repositorios de entregas, y lo que esté en el tuyo en ese momento es tu entrega. Lo que subas después no cuenta.
5. **Revisión entre compañeros.** Guarda también una copia en Drive (*Archivo → Guardar una copia en Drive*) y compártela con tu compañero revisor con permiso de comentar. Tu repositorio sigue siendo privado.

Los sprints se entregan igual, en su carpeta (`Sprint1/`…). Para el proyecto final cada pareja recibe un repositorio privado compartido, `Alto-Rendimiento-IA/proyecto-<pareja>`.

¿Quieres aprender más? El curso gratuito [Introduction to GitHub](https://github.com/skills/introduction-to-github) toma menos de una hora. Más adelante (proyecto final, en parejas) vale la pena aprender a usar `git` desde la terminal.

## Entorno de trabajo

El curso usa **Python 3.13** y las mismas versiones de librerías que Google Colab (lista completa en [`requirements.txt`](requirements.txt)).

**Opción recomendada: Google Colab.** No hay que instalar nada. Para usar GPU: *Entorno de ejecución → Cambiar tipo de entorno de ejecución → T4 GPU*. Si un notebook necesita una librería que Colab no trae, la instala en su primera celda.

**Opción local (si tienes una computadora propia y quieres trabajar sin conexión).** Instala [uv](https://docs.astral.sh/uv/getting-started/installation/) y, dentro de la carpeta del repositorio, ejecuta:

```bash
uv venv --python 3.13                 # crea el entorno en .venv/ (descarga Python 3.13 si hace falta)
uv pip install -r requirements.txt    # instala las librerías del curso (unos 4 GB, por PyTorch)
uv run jupyter lab                    # abre Jupyter en el navegador
```

En Windows con GPU NVIDIA, instala antes PyTorch con soporte CUDA siguiendo [pytorch.org/get-started](https://pytorch.org/get-started/locally/) (versión 2.11.0); sin GPU, los comandos de arriba alcanzan.

**En los dos casos**, en la primera clase ejecuta [`verificar_entorno.ipynb`](verificar_entorno.ipynb) ([abrir en Colab](https://colab.research.google.com/github/melvinpqbsc/Alto_Rendimiento_IA/blob/main/verificar_entorno.ipynb)): revisa las versiones, la GPU y entrena un modelo mínimo.

---
 
## 0. Supuestos
 
| Ítem | Supuesto |
|---|---|
| Encuentros | 1 por semana, 2 h, los **miércoles**, en línea (curso nacional) |
| Tiempo de trabajo fuera de clase | 4–5 h/semana (más que en el nivel inicial) |
| Requisito de entrada | Escribe funciones y bucles en Python sin ayuda; sabe qué es una derivada y una matriz |
| Calendario | Primer período 7 oct – 16 dic · receso de fin de año (incluye Navidad y Año Nuevo) · 3 feb – 30 jun |
| Sesiones | **33 encuentros** (ningún feriado nacional de Ecuador del período cae en miércoles: la ley los traslada a lunes o viernes) |
 
Como el curso es nacional y en línea, solo cuentan los feriados nacionales. Si cambia el día de clase, hay que revisar cuáles caen ese día y mover las filas.
 
**Tipos de sesión:**
**[A]** clase expositiva del profesor (≈16 en total) · **[B]** laboratorio, los estudiantes avanzan en la plataforma · **[C]** seminario de estudiantes (20 min + notebook ejecutable) · **[D]** sprint / simulacro, conjunto de prueba oculto, tabla de posiciones pública.
 
**Apertura fija de 15 minutos cada semana:** tabla de posiciones en pantalla → "bug de la semana" presentado por un estudiante → resumen de una diapositiva de la semana anterior, a cargo de un estudiante rotativo.
 
---
 
## 1. Visión general
 
| Bloque | Fechas | Sesiones | Tema |
|---|---|---|---|
| **1. Puente** | 7 oct – 4 nov | 5 | Repaso: Python/NumPy, datos con pandas, matemática para deep learning, ML clásico en síntesis |
| **2. Redes neuronales y PyTorch** | 11 nov – 16 dic | 6 | Perceptrón → backpropagation → PyTorch → entrenar bien |
| **Receso de fin de año** | 17 dic – 2 feb | asincrónico | ML clásico / datos tabulares + Kaggle |
| **3. Visión por computadora** | 3 feb – 24 mar | 8 | CNN, transfer learning, detección/segmentación, modelos generativos |
| **4. Lenguaje, audio, multimodal** | 31 mar – 19 may | 8 | Embeddings, atención, transformers, LLM, CLIP, Whisper |
| **5. Preparación para la competencia** | 26 may – 30 jun | 6 | Repaso de ML clásico, problemas de ediciones anteriores, simulacro de 6 horas, cierre |
 
---
 
## 2. Semana por semana
 
### Bloque 1: Puente (repaso acelerado del contenido del nivel inicial)
 
Objetivo: en 5 sesiones (herramientas + 4 de repaso), asegurar que todos tengan exactamente las herramientas que el deep learning da por sabidas. Quienes ya dominan el contenido vencen al "jefe" (*boss*) de esos niveles en la plataforma y actúan como monitores.
 
**Clase invertida** (*flipped classroom*): antes de cada clase se leen 2 o 3 notebooks de [MÓLÓ](https://mi.versenyportal.hu/en/molo#notebooks), el programa de entrenamiento de la Olimpiada Húngara de IA (ruta *Competitor*, en inglés, licencia CC BY-NC-SA). Ábrelos en Colab y ejecuta las celdas; no se entregan. La clase no repite la lectura: parte de ella, resuelve dudas y llega hasta lo que el deep learning necesita.
 
| Sem | Fecha | Sesión | Tipo | Entrega al inicio de la clase |
|---|---|---|---|---|
| 1 | **7 oct** | **Inicio: las herramientas del curso.** Qué es la IOAI y cómo funciona el curso. Cuentas (plataforma y GitHub, ver [Antes de empezar: GitHub](#antes-de-empezar-github)), invitación al repositorio privado de entregas, configuración de Colab y `verificar_entorno.ipynb`. **Git y GitHub en la práctica**: repositorio, *commit*, historial; guardar desde Colab un primer notebook en una carpeta de prueba del repositorio de entregas y ver el *commit* en GitHub. **Python orientado a objetos** (*object-oriented programming*): clases, `__init__`, atributos, métodos, herencia y `__call__`; construir una clase `Linear` con un método `forward`, el mismo patrón de `nn.Module` y `Dataset` desde la semana 8. | [A] | — |
| 2 | 14 oct | **NumPy para tensores + álgebra lineal**: formas (*shapes*), indexación, *broadcasting*, vectorización, `reshape`/`axis` (el modelo mental que PyTorch reutiliza en todo); producto punto, matriz × vector como "una capa", matriz × matriz como un lote (*batch*) que pasa por esa capa. | [A] | Niveles de Python superados en la plataforma |
| 3 | 21 oct | **Datos reales: pandas, gráficos y estadística descriptiva.** Laboratorio: cargar un CSV, tipos de columnas, datos faltantes, `groupby`, histogramas y diagramas de dispersión; media, desviación estándar, correlación. Datos estructurados vs no estructurados: una imagen y un texto también terminan siendo arreglos. | [B] | **B1: Ejercicios de NumPy** (vectorizar 6 funciones con bucles, reportar la aceleración) |
| 4 | 28 oct | **Cálculo y probabilidad para DL**: derivadas parciales, gradiente, regla de la cadena; la pérdida como función de los parámetros; descenso de gradiente a mano; probabilidad → softmax → logaritmo → entropía cruzada. | [A] | Cuota semanal |
| 5 | 4 nov | **ML clásico en síntesis**: aprendizaje supervisado, entrenamiento/validación/prueba, sobreajuste y sesgo-varianza, validación cruzada, *accuracy* vs precisión/recall/F1, `fit/predict` de sklearn (regresión lineal y logística, kNN, árbol de decisión). Al final se publica el **Sprint 0** (asincrónico, se entrega en la semana 6): limpiar un CSV desordenado, ajustar dos modelos de sklearn, reportar métricas contra una línea base. | [A] | **B2: Descenso de gradiente a mano** (1-D y 2-D, gráfico de contornos, romperlo con una mala tasa de aprendizaje) |
 
**Lectura previa (MÓLÓ).** Se lee durante la semana anterior a la clase indicada.
 
| Para la sem | Notebooks |
|---|---|
| Repaso de la sem 1 | [Version Control with Git](https://colab.research.google.com/drive/1zHl10Ig9yOJuDcGovfO3mukjnuS0pn0_), [Object-Oriented Programming in Python](https://colab.research.google.com/drive/1W7DHMCFkxaOyusVImjvWKan-gIxJB_VO). Como consulta: [Google Colab Notebook introduction](https://colab.research.google.com/drive/1EnMf1IDa22wiiCC1xTLeiyhEs0IJ_DT8). Si todavía te cuestan las funciones y los bucles: [Python Basics](https://youtu.be/plWVZk1uqCs) (video, ruta *Explorer*) |
| 2 | [Introduction to NumPy](https://colab.research.google.com/drive/1iKRRivG4blcvVzXLT1fNqfYh8kMuFL-m), [Linear Algebra Basics for ML](https://colab.research.google.com/drive/1E4R9ZwvEH_6cw2N03eaHt6oXqLqVq2Oc) |
| 3 | [Advanced NumPy](https://colab.research.google.com/drive/1HWvN658jm7TuzNucOSTmFO95zzYfRs_1) (indexación avanzada y *broadcasting*, ayuda con B1), [Data Analysis with Pandas](https://colab.research.google.com/drive/1JSY4kZpt-H5IUkEKturC-38-u9g4jIer), [Descriptive statistics](https://colab.research.google.com/drive/12-WDEoD-CApbtgtVaACc60Z-EFVmOiju) |
| 4 | [Calculus basics for ML](https://colab.research.google.com/drive/1yc25B_ODIvC1TtdjAJllyaSfXXCOzgSB), [Functions and Parameters](https://colab.research.google.com/drive/1LvEAjw4XcV_k0QZ-QMkNTyaIS9y1ntPv), [Probability and Distributions](https://colab.research.google.com/drive/1ssiR39Q8H0W7LTLSKR8VkayxqX7wpzWM) |
| 5 | [Supervised Learning](https://colab.research.google.com/drive/140FnxeOBXsbFYKNvJP8WwQ7XH45LNZ1W), [Inference, Training, Test and Evaluation](https://colab.research.google.com/drive/1H0AbbfDlyfmJDrIDcXVOqgC6OO87VO7g), [Generalisation, Overfitting and Bias](https://colab.research.google.com/drive/1pzug6uPG9Ee_2dxRdzvymyzrpoUyfOGD) |
| 6 (consulta para el Sprint 0) | [Regression](https://colab.research.google.com/drive/1oAHb8vKEVqUfL-RVua3OHkDhNAPiABHA), [Decision Trees](https://colab.research.google.com/drive/1JCPwXKXioucTJigNkxCnuhKXL-h0gw-m), [Non-Parametric Learning](https://colab.research.google.com/drive/1d7jCTYiRb9EMrYS79wQQzDUSjmFZy5vQ) (kNN) |
 
**Punto de control del bloque (se entrega en la semana 7):** **B3 — Regresión lineal y logística desde cero** en NumPy, comparadas con sklearn sobre la misma partición. Es la puerta de entrada: contiene todas las ideas sobre las que se construye el Bloque 2.
 
### Bloque 2: Redes neuronales y PyTorch
 
| Sem | Fecha | Sesión | Tipo | Entrega |
|---|---|---|---|---|
| 6 | 11 nov | **Perceptrón → MLP.** Propagación hacia adelante en papel; funciones de activación (ReLU, sigmoide, tanh) y por qué importa la no linealidad. | [A] | Sprint 0 |
| 7 | 18 nov | **Backpropagation.** La clase que no se debe delegar. Regla de la cadena sobre un grafo computacional pequeño, despacio, en la pizarra. | [A] | **B3** |
| 8 | 25 nov | **PyTorch**: tensores, `autograd`, `nn.Module`, `DataLoader`, el bucle de entrenamiento de 5 líneas, CPU vs GPU. | [A] | **P1: Motor de diferenciación automática desde cero** (estilo micrograd, entrenar un MLP sobre "two moons") |
| 9 | 2 dic | **Funciones de pérdida, inicialización, batch norm.** Seminarios: MSE vs MAE vs entropía cruzada; inicialización de pesos; batch norm. | [C] | **P2: MLP sobre Fashion-MNIST + tabla de ablación** (una variable a la vez) |
| 10 | 9 dic | **Entrenar bien**: SGD, mini-batch, momentum, Adam/AdamW, tasas de aprendizaje y *schedules*; dropout, weight decay, early stopping. | [A] | **P3: Carrera de optimizadores** (SGD / momentum / Adam / AdamW, un gráfico, un párrafo) |
| 11 | 16 dic | **Sprint 1**: el mejor modelo que se pueda entrenar en 15 minutos de GPU, evaluado con un conjunto de prueba oculto. Luego, retrospectiva + presentación del plan del receso. | [D] | Sprint 1 |
 
### Receso de fin de año (17 dic – 2 feb): asincrónico, con seguimiento público
 
Sin clases entre el Sprint 1 y el reinicio; Navidad (25 dic) y Año Nuevo (1 ene) caen en este receso.
 
El programa de la IOAI sigue exigiendo ML clásico (árboles, gradient boosting, SVM, PCA, k-means, t-SNE/UMAP), y los problemas tabulares de competencia suelen ganarse con esas técnicas. El receso es el lugar para eso. Cada estudiante elige un nivel:
 
- **Bronce**: mantener la racha en la plataforma; superar los niveles de ML clásico. Material de apoyo: los notebooks de *Classical ML Methods*, *Unsupervised Learning* y *Kaggle Competitions* de [MÓLÓ](https://mi.versenyportal.hu/en/molo#notebooks).
- **Plata**: una competencia Kaggle Playground, ≥10 envíos, bitácora de 1 página. Debe probar gradient boosting (XGBoost/LightGBM).
- **Oro**: Plata + **S1 — Informe no supervisado**: PCA, k-means y UMAP sobre un conjunto de datos real (por ejemplo, indicadores socioeconómicos por municipio o región), ponerle nombre a los clústeres y argumentar si significan algo.
Un mensaje por semana, no más.
 
### Bloque 3: Visión por computadora
 
| Sem | Fecha | Sesión | Tipo | Entrega |
|---|---|---|---|---|
| 12 | 3 feb | **Reinicio.** Muestra de los trabajos del receso. Repaso de 30 min del bucle de entrenamiento (también es la puerta de entrada para los egresados del nivel inicial que se suman ahora; ver §4). | [C] | Bitácora del receso |
| 13 | 10 feb | **Convolución**: qué es un filtro; construir uno en NumPy y luego con `nn.Conv2d`; *padding*, *stride*, *pooling* (máximo/promedio). | [A] | Cuota semanal |
| 14 | 17 feb | **Arquitecturas CNN**: de LeNet a ResNet, conexiones residuales. Laboratorio. | [B] | **P4: CNN desde cero sobre CIFAR-10** |
| 15 | 24 feb | **Transfer learning y fine-tuning**: ResNet preentrenada, congelar capas vs ajuste completo. La habilidad de competencia más útil del año. | [A] | Cuota semanal |
| 16 | 3 mar | **Aumentación de datos y datos reales.** Seminario: aumentación (volteos, recortes, ruido) y qué aporta. | [C] | **P5: CNN desde cero vs ResNet ajustada**, mismo presupuesto de 15 min de GPU |
| 17 | 10 mar | **Detección y segmentación**: YOLO (práctico), U-Net. Seminarios. | [C] | **P6: Tu propio dataset de 200 imágenes**, 4 clases, partición correcta, clasificador + informe de aumentación |
| 18 | 17 mar | **Panorama generativo y autosupervisado**: autoencoders, GAN, difusión, aprendizaje autosupervisado. Seminarios, nivel práctico. | [C] | **P7: Autoencoder sobre MNIST** + gráfico del espacio latente |
| 19 | 24 mar | **Sprint 2 (visión)**: clasificación de imágenes en un dataset nuevo, conjunto de prueba oculto. | [D] | Sprint 2 |
 
### Bloque 4: Lenguaje, audio, multimodal
 
| Sem | Fecha | Sesión | Tipo | Entrega |
|---|---|---|---|---|
| 20 | 31 mar | **Embeddings y tokenización**: texto (BPE, vocabulario), parches de imagen, audio. *Padding* de secuencias de longitud variable. | [A] | Informe del Sprint 2 |
| 21 | 7 abr | **Atención**, paso a paso en la pizarra: *queries*, *keys*, *values*, softmax, enmascaramiento. | [A] | Cuota semanal |
| 22 | 14 abr | **Transformers y BERT**: bloques codificadores, codificación posicional; ajustar un BERT en español (p. ej. BETO). | [B] | **P8: Atención desde cero** en PyTorch, verificada contra `nn.MultiheadAttention` |
| 23 | 21 abr | **Visión-texto**: ViT, CLIP, clasificación *zero-shot*. Seminarios. | [C] | **P9: Análisis de sentimiento en español**: BERT en español vs línea base TF-IDF + regresión logística |
| 24 | 28 abr | **Modelado de lenguaje**: construir un GPT pequeño (el camino de nanoGPT de Karpathy), muestreo, temperatura. | [A] | **P10: CLIP zero-shot sobre fotos propias + galería de errores** · **Propuesta de proyecto** (1 página, en parejas) |
| 25 | 5 may | **LLM preentrenados** (de código abierto y vía API), *prompting*, construcción de un conjunto de evaluación; **fine-tuning eficiente en parámetros (LoRA)**. | [A] | **P11: GPT pequeño** entrenado sobre un corpus de texto en español |
| 26 | 12 may | **Audio**: Whisper, HuBERT; modelos codificador-decodificador. Seminarios. | [C] | **P12: Benchmark de prompts**: conjunto de evaluación de 30 preguntas, 3 estrategias de *prompting*, *accuracy* con barras de error |
| 27 | 19 may | **Sprint 3 (multimodal)**: tarea de texto o imagen+texto, conjunto de prueba oculto. | [D] | Sprint 3 |
 
### Bloque 5: Preparación para la competencia
 
| Sem | Fecha | Sesión | Tipo | Entrega |
|---|---|---|---|---|
| 28 | 26 may | **Día tabular y clásico**: dos seminarios de 15 min (kNN, árboles de decisión) y luego gradient boosting, clustering, PCA/t-SNE/UMAP, ingeniería de variables, datos faltantes. Ejercicio cronometrado al final. | [A] | **P13: Whisper con distintos acentos del español** (WER por región) |
| 29 | 2 jun | **Técnica de competencia con problemas de ediciones anteriores de la IOAI**: leer el problema, línea base en 20 minutos, después iterar; manejo del tiempo; interpretar métricas. | [B] | Problema de edición anterior n.º 1 |
| 30 | 9 jun | **Técnica de competencia II**: un segundo problema de edición anterior, cronometrado (2 h), y luego revisión cruzada de soluciones; errores típicos (fuga de datos, métrica mal leída, formato de envío equivocado). | [B] | Problema de edición anterior n.º 2 |
| 31 | 16 jun | **Presentaciones de proyectos** (en parejas, 8 min cada una). | [C] | **Proyecto** |
| 32 | 23 jun | **Simulacro completo, 6 horas** (los días de competencia de la IOAI duran 6 h en máquinas con GPU). Idealmente un sábado. Problemas de ediciones anteriores, solo en inglés, sin ayuda. | [D] | — |
| 33 | 30 jun | **Análisis del simulacro + cierre.** Repaso de las soluciones del simulacro; qué sigue (universidad, investigación, próximas competencias). | [A] | Reflexión sobre el simulacro |
 
---
 
## 3. Cobertura del programa (programa IOAI 2026 → semana)
 
| Área del programa | Dónde |
|---|---|
| Python, NumPy, pandas, Matplotlib, sklearn | Sem 1–5, plataforma, lectura previa MÓLÓ |
| PyTorch, tensores, CPU/GPU | Sem 8 en adelante |
| Regresión lineal/logística, kNN, árboles, métricas, validación cruzada, sobreajuste | Sem 5, Sprint 0, B3, receso, sem 28 |
| Ensambles, SVM, L1/L2 | Receso, sem 28 |
| PCA, k-means, t-SNE/UMAP, DBSCAN | Receso (S1), sem 28 |
| Perceptrón, backprop, activaciones, pérdidas, MLP | Sem 6–9 |
| SGD/Adam, tasa de aprendizaje, dropout, weight decay, inicialización, batch norm | Sem 9–10 |
| Capas convolucionales, pooling, clasificación, codificadores preentrenados, aumentación | Sem 13–16 |
| Detección, segmentación | Sem 17 |
| Autoencoders, GAN, difusión, autosupervisado | Sem 18 |
| Embeddings, atención, transformers, BERT | Sem 20–23 |
| CLIP, visión-texto | Sem 23 |
| Modelado de lenguaje, LLM preentrenados, fine-tuning (completo + PEFT) | Sem 24–25 |
| Codificador-decodificador, audio (Whisper, HuBERT) | Sem 26 |
 
---
 
## 4. Cómo conviven los dos niveles
 
- **Cada estudiante elige su nivel**; no hay test de ubicación. Si durante el Bloque Puente alguien ve que el ritmo no le sirve, puede cambiarse de nivel. El B3 (semana 7, mediados de noviembre) es un buen momento para decidirlo, en los dos sentidos.
- **Los egresados del nivel inicial se suman en febrero.** La semana 12 está pensada como su punto de entrada. En enero reciben un paquete de nivelación de 2 semanas: B3 + P2 (ablación del MLP) + los videos de backpropagation y PyTorch (3Blue1Brown, Karpathy). Si se suman muchos, agregar una sesión extra de nivelación [B] en la semana 13.
- **Los estudiantes intermedios como monitores del nivel inicial**, una o dos veces por mes. Explicarle NumPy a un principiante es el mejor repaso posible, y le quita carga al profesor.
- **Sprints compartidos.** El Sprint 1 (16 dic) puede tener una variante para el nivel inicial (solo sklearn) sobre el mismo dataset y en la misma tabla de posiciones. Les da a los principiantes un objetivo concreto.
---
 
## 5. Evaluación
 
| Componente | Peso |
|---|---|
| Progreso en la plataforma (niveles superados) | 20% |
| Trabajos prácticos (B1–B3, P1–P13) | 35% |
| Seminarios presentados | 15% |
| Sprints + simulacro | 10% |
| Proyecto | 20% |
 
Cada notebook se evalúa con cuatro criterios, de 0 a 2 cada uno: **se ejecuta de principio a fin / es correcto / está justificado / está bien comunicado**, y termina con un párrafo en lenguaje sencillo sobre qué significan los números. Antes de que lo corrija el profesor, lo revisa un compañero.
 
---
 
## 6. Notas prácticas
 
- **Recursos de cómputo**: Colab gratuito alcanza hasta abril, aproximadamente. Desde el Bloque 4 (GPT pequeño, LoRA, Whisper) conviene prever las horas gratuitas de GPU de los notebooks de Kaggle o una máquina de laboratorio de la universidad; organizarlo en marzo.
- **El simulacro de 6 horas (sem 32)** necesita un sábado y GPU para cada estudiante durante 6 horas (Colab o Kaggle). Organizarlo en mayo.
- **Idioma**: las clases y los materiales van en español, pero la plataforma, la documentación y la competencia están en inglés. No traducir la plataforma: esa fricción también es entrenamiento.
- **Es esperable una deserción** del 30–50% hacia febrero; ningún trabajo práctico debe depender de uno de un bloque anterior.
