"""Arma la tabla de posiciones de un sprint a partir de una recolección.

    python herramientas/calificar_sprint.py herramientas/recolecciones/<fecha_hora>_Sprint0 \\
        Nivel_Intermedio/test_oculto/Sprint0_etiquetas.csv

Lee <usuario>/<trabajo>/predicciones.csv de cada estudiante (columnas id,Churn con 0 o 1),
lo compara con las etiquetas ocultas y escribe tabla_posiciones.csv en la carpeta de la
recolección. La métrica es F1 de la clase 1; también se reportan accuracy, precisión y recall.
"""

import argparse
import csv
from pathlib import Path


def leer_columnas(ruta, columna):
    """Lee un CSV id,<columna> y devuelve {id: valor}. Lanza ValueError si el formato no sirve."""
    with open(ruta, encoding="utf-8-sig", newline="") as f:
        lector = csv.DictReader(f)
        if not lector.fieldnames or "id" not in lector.fieldnames or columna not in lector.fieldnames:
            raise ValueError(f"faltan las columnas id,{columna}")
        valores = {}
        for fila in lector:
            id_, valor = fila["id"].strip(), fila[columna].strip()
            if valor.endswith(".0"):  # 1.0 en vez de 1: lo aceptamos
                valor = valor[:-2]
            if valor not in ("0", "1"):
                raise ValueError(f"valor {valor!r} en id {id_}: debe ser 0 o 1")
            if id_ in valores:
                raise ValueError(f"id {id_} repetido")
            valores[id_] = int(valor)
    return valores


def metricas(verdad, prediccion):
    vp = sum(1 for i, y in verdad.items() if y == 1 and prediccion[i] == 1)
    fp = sum(1 for i, y in verdad.items() if y == 0 and prediccion[i] == 1)
    fn = sum(1 for i, y in verdad.items() if y == 1 and prediccion[i] == 0)
    aciertos = sum(1 for i, y in verdad.items() if prediccion[i] == y)
    precision = vp / (vp + fp) if vp + fp else 0.0
    recall = vp / (vp + fn) if vp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"f1": f1, "accuracy": aciertos / len(verdad), "precision": precision, "recall": recall}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("recoleccion", help="carpeta creada por recolectar_entregas.py")
    parser.add_argument("etiquetas", help="CSV oculto id,<columna> (en test_oculto/, nunca en el repositorio)")
    parser.add_argument("--trabajo", default="Sprint0", help="carpeta del sprint en cada repositorio (Sprint0 por defecto)")
    parser.add_argument("--columna", default="Churn", help="columna de la etiqueta (Churn por defecto)")
    args = parser.parse_args()

    raiz = Path(args.recoleccion)
    verdad = leer_columnas(args.etiquetas, args.columna)
    tabla, problemas = [], []
    for carpeta in sorted(p for p in raiz.iterdir() if p.is_dir()):
        ruta = carpeta / args.trabajo / "predicciones.csv"
        if not ruta.exists():
            problemas.append((carpeta.name, f"no hay {args.trabajo}/predicciones.csv"))
            continue
        try:
            prediccion = leer_columnas(ruta, args.columna)
        except (ValueError, UnicodeDecodeError, csv.Error) as err:
            problemas.append((carpeta.name, str(err)))
            continue
        if prediccion.keys() != verdad.keys():
            faltan, sobran = len(verdad.keys() - prediccion.keys()), len(prediccion.keys() - verdad.keys())
            problemas.append((carpeta.name, f"ids distintos a los de test.csv ({faltan} faltan, {sobran} sobran)"))
            continue
        tabla.append({"usuario": carpeta.name, **metricas(verdad, prediccion)})

    tabla.sort(key=lambda fila: -fila["f1"])
    print(f"{'#':>3}  {'usuario':<24}{'F1':>7}{'accuracy':>10}{'precisión':>11}{'recall':>8}")
    for puesto, fila in enumerate(tabla, 1):
        fila["puesto"] = puesto
        print(f"{puesto:>3}  {fila['usuario']:<24}{fila['f1']:>7.3f}{fila['accuracy']:>10.3f}"
              f"{fila['precision']:>11.3f}{fila['recall']:>8.3f}")
    for usuario, motivo in problemas:
        print(f"  -  {usuario:<24}sin puntaje: {motivo}")

    with open(raiz / "tabla_posiciones.csv", "w", encoding="utf-8", newline="") as f:
        escritor = csv.DictWriter(f, fieldnames=["puesto", "usuario", "f1", "accuracy", "precision", "recall", "problema"])
        escritor.writeheader()
        for fila in tabla:
            escritor.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()})
        for usuario, motivo in problemas:
            escritor.writerow({"usuario": usuario, "problema": motivo})
    print(f"\nTabla guardada en {raiz / 'tabla_posiciones.csv'}")


if __name__ == "__main__":
    main()
