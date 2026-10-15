# CLAUDE.md

Course repository for training high-school students toward IOAI 2027 (Singapore, 4–10 July 2027). Two levels, one weekly 2 h class each, all work done in Google Colab notebooks. This **public** repo holds the schedule, class notebooks and homework statements. Each student has one private repo, `Alto-Rendimiento-IA/entregas-<usuario>`, for the whole year: they open a statement from here in Colab and save it there under `<codigo>/`. The teacher's scripts in `herramientas/` create those repos and snapshot them at each deadline; that snapshot is the submission (student workflow: README.md, section "Antes de empezar: GitHub"; teacher workflow: `herramientas/README.md`).

## Sources of truth

- `README.md`: Intermediate-level schedule. Week numbers, session types, topics, deliverable codes, grading, and the IOAI syllabus coverage table (§3). Read the relevant row before writing anything.
- `Guia_del_estudiante.md`: student-facing rules (AI-use policy, rubric, deliverable rules). Notebooks must follow them.
- Nivel Inicial has no schedule yet. Before writing any of its notebooks, ask the user for the topic list and the entry level. What is known: it runs Oct–Dec 2026, it uses sklearn but no PyTorch (README §4), and its graduates join the intermediate level in February.
- `Guia_estudiante.docx` is the Word version of the guide, kept in sync with the `.md` by editing its XML (docx skill). If a date changes, update README.md, Guia_del_estudiante.md and the .docx together. Notebooks give the week number only, never a calendar date, so they stay correct when the calendar changes.

## Environment

- `requirements.txt` mirrors the **Colab runtime** (Python 3.13), so local runs behave like the students' Colab runs. To refresh it, take the versions from `https://raw.githubusercontent.com/googlecolab/backend-info/main/pip-freeze.txt`. Keep its section headers (`# --- ...`), because `verificar_entorno.ipynb` parses them.
- The last section lists packages Colab lacks. A notebook that uses one installs it in its first cell with the pinned version (`%pip install -q ultralytics==<pin>`). Colab's preinstalled packages are never reinstalled from a notebook: that is slow and can break its CUDA build.
- Local env: `.venv/` (gitignored, built with `uv venv --python 3.13 && uv pip install -r requirements.txt`, uv lives at `~/.local/bin/uv`). This machine has an RTX 5090, so GPU notebooks can be verified here. Don't add `.python-version`: the user's pyenv reads it and breaks `python3` in this folder.

## Layout and naming

```
Nivel_<Inicial|Intermedio>/
  clases/semNN_<tema>.ipynb        # NN = README week number, two digits: sem08_pytorch.ipynb
  tareas/<codigo>_<tema>.ipynb     # codes from README: B0…B3, P01…P13, S1, Sprint0…Sprint3
  soluciones/                      # reference solutions, gitignored, local only
  README.md                        # index: add a row for every notebook you create
```

`herramientas/` holds the teacher's stdlib-only scripts (emails to students, repo creation, deadline snapshots). Email templates live in `herramientas/plantillas/`. Every `herramientas/*.csv` except `*.ejemplo.csv`, and the snapshots, are gitignored because they hold personal data. They must never be committed, and neither must any other file with student data (form exports, grades).

File names are snake_case ASCII: no accents, no `ñ`, no spaces, because they end up in Colab and GitHub URLs.

## Writing a notebook

1. **Locate the row.** Find the week/code in README.md: its topic is the scope, its type ([A]/[B]/[C]/[D]) sets the shape (below), and the §3 table lists which syllabus items it must cover.
2. **Check the prerequisites.** List what earlier weeks taught and use only those tools (for example, no PyTorch before week 8, no `nn.Conv2d` before week 13). Each homework must also be **self-contained**: a student who skipped every earlier homework can still complete it (README §6, expected 30–50% dropout).
3. **Build it with `nbformat`** from a Python script in the scratchpad. Hand-edited notebook JSON breaks.
4. **Verify.** Run `.venv/bin/jupyter nbconvert --to notebook --execute` on class notebooks and on every homework's reference solution. The homework statement itself fails its checks by design. Done means it runs top to bottom with no errors on a fresh kernel. If it needs a GPU or Colab-only features and you can't run it locally, tell the user which cells you could not run.
5. **Commit with outputs cleared**, then add the row to the level README.

## Every notebook

- **Spanish** for all prose, comments, and plot labels. Code identifiers are in English, following PyTorch/sklearn conventions. The first time a technical term appears, give it in English in italics with a Spanish gloss (*broadcasting*, *accuracy*), as README does. Datasets, APIs, and competition problems stay in English: that friction is part of the training.
- **Header cell** with: title, level, week number, session type, 3–4 learning objectives, prerequisites, estimated time, compute needs (CPU / GPU T4), and an "Abrir en Colab" badge pointing to `https://colab.research.google.com/github/melvinpqbsc/Alto_Rendimiento_IA/blob/main/<path>`.
- **Fresh-runtime reproducible**: the first code cell holds the `%pip install -q` lines (see Environment), all imports, and a fixed seed. Data comes from torchvision / sklearn / Hugging Face datasets or a stable public URL, with no Drive mounting and no local paths. Size datasets and training so they fit in free Colab: minutes, not hours, unless the README row says otherwise (for example, the 15 min GPU budgets).
- **Baseline first**: every modelling result is compared against a simple baseline (majority class, linear model, TF-IDF + logistic regression, …).

## Classes (`clases/`) by session type

- **[A] lecture**: teacher-driven. Short explanation cells alternate with live demos. Build ideas from scratch in NumPy before showing the library call. End sections with short "Tu turno" exercises.
- **[B] lab**: mostly exercises in increasing difficulty, each with a check cell so students can see their own progress while the teacher circulates.
- **[C] seminar**: the students present, so write a scaffold for the presenter: topic outline, the key figure/demo to build, 3 references, and 3 discussion questions for the audience.
- **[D] sprint**: problem statement, data loading, a working baseline, the submission format, and a local validation split. Delivered like homework, in the student's `entregas` repo under `SprintN/`. The hidden test set lives only in `test_oculto/` (gitignored). Any public commit of it leaks the answers.

## Class videos (`videos/`)

Each class notebook gets a narrated explainer video (Spanish, Ecuadorian voice, ~10–15 min) for students who missed the class or want to review it. A video is a **script** (`videos/guiones/<notebook stem>.py`) rendered by `videos/motor.py`: slides drawn with Pillow, narration from edge-tts (needs internet), and an MP4 with soft subtitles plus an `.srt`, all written to `videos/salida/<stem>/` (gitignored). The env is `.venv-video/` (setup in `videos/requirements.txt`), separate from `.venv` because `.venv` mirrors Colab. `videos/guiones/sem01_herramientas.py` is the model: copy its structure and visual vocabulary. The element API (`Texto`, `Codigo`, `Terminal`, `Caja`, `Flecha`, `Tabla`, `Matriz`, `Grafica`, ...) is documented in `motor.py`.

1. **Read the notebook end to end**, plus its README row. The video covers the same parts in the same order, with the same week numbers, names, and examples (run the notebook's code to get real outputs for `Terminal`).
2. **Write the script.** A `Seccion` marker per notebook part, an `Escena` per idea, a `Paso` per sentence or two of narration with the elements it reveals. Aim for 50–70 steps.
   - **Make it as intuitive as possible.** The video is for a 15-year-old who may have missed the class. Every idea gets three things, in this order:
     1. An everyday analogy: commit = a photo of the project, class = a cookie mold, loop vs NumPy = a cashier scanning item by item vs the whole cart at once, reshape = the same eggs in a different carton, mask = a colander, slice = a window rather than a photocopy.
     2. A concrete example with small, real numbers.
     3. Only then, the definition and the code.
   - **Show operations, don't describe them.** Animate them step by step across `Paso`s:
     - A `Matriz` whose axis collapses for a reduction.
     - A `Tabla` of shapes for broadcasting, with the failing row framed by a `Marco`.
     - `Pixeles` cells moving for `reshape`.
     - A row times a column for a dot product.
     - Arrows into a neuron for a layer.
     - Arrows or `Caja` diagrams for data flow and pipelines.
   - Prefer real data over abstract boxes. For example, draw an actual digit with `Pixeles`, and show the notebook's real timings and accuracies in `Terminal`.
   - A text-only slide is a last resort; use one only where an idea has no shape.
   - Keep one idea per scene.
   - Narration is spoken Spanish in *tú*: short sentences, numbers and symbols said in words ("x por W transpuesta"), no markup, no hashes or long identifiers read aloud. On screen goes the real code and terms.
   - "Tu turno" exercises: explain the idea and say "pausa el video"; show only the statement's skeleton. Never show a solution or the answer a check cell verifies (e.g. the parameter count of `net`).
   - **Data in the script.** `.venv-video` has no sklearn or torch. If a scene needs dataset values (images, means), compute them once with `.venv`, then embed them in the script as compact literals. `sem02_numpy.py` does this with its digit images.
   - **New visuals.** If no element fits, add one to `motor.py` next to the others (`Pixeles` was added this way), with a docstring like the others.
3. **Check the layout**: `.venv-video/bin/python videos/motor.py <script> --solo-cuadros`, then look at every image in `videos/salida/<stem>/hojas/` (last step of each scene) and at `cuadros/` for steps that replace elements. Done when nothing overlaps, no text runs off its box, no glyph renders as a box, and nothing touches the footer.
4. **Build and check the voice**: run `motor.py <script>` without flags, then `.venv/bin/python videos/revisar_voz.py <script>`. Whisper is noisy, so look for a technical term mangled across several steps; fix it in `PRONUNCIACION` (motor.py, audio only) or by rephrasing. Sentence-initial English words are read worst: put a Spanish word before them ("Por otro lado, GitHub..."). Rebuilding only synthesizes changed sentences (cached). `PRONUNCIACION` is shared by every script, so a change there re-voices older videos the next time they are rebuilt. If edge-tts fails with a 403 handshake error, Microsoft changed its endpoint: upgrade `edge-tts` in `.venv-video` and pin the new version in `videos/requirements.txt`.
5. **Deliver**: commit the script (and any engine changes); the MP4 stays out of git. Give the user the MP4 path and its length; they upload it (e.g. YouTube, with the `.srt`).

## Homework (`tareas/`)

- The notebook is a **statement**, never a solution. It holds context and a goal, numbered tasks, `# TODO` cells, and `assert`-based check cells wherever correctness is checkable (shapes, values against a library reference such as micrograd vs `torch.autograd`, attention vs `nn.MultiheadAttention`).
- Reference solutions go to `<nivel>/soluciones/` (gitignored). The repo is public, so a solution that gets committed has already leaked.
- **Delivery.** The homework header tells students to save their copy right away to `Alto-Rendimiento-IA/entregas-<usuario>` at `<codigo>/<file name>` (e.g. `P01/P01_autograd_desde_cero.ipynb`) and to keep working from that copy. The snapshot script checks for a notebook in the `<codigo>/` folder, so the folder name must be exactly the README code.
- Peer review happens outside the private repos: students share a Drive copy of the notebook with their reviewer. Write the rubric cell so a reviewer can fill it in as a comment.
- It ends with three fixed cells: **Conclusión** (the student's plain-language paragraph: what the numbers mean, what they'd try next; no paragraph, no grade), **Uso de IA** (one line: what they used AI for, if anything), and the **rubric**: runs / correct / justified / well communicated, 0–2 each, with space for the student's self-score.
