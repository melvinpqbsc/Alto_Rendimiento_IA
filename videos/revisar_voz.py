"""Transcribe con Whisper la narración de un guion ya construido, para encontrar
términos que la voz pronuncia mal.

    .venv/bin/python videos/revisar_voz.py videos/guiones/sem01_herramientas.py

Usa el entorno principal (.venv), que tiene torch y transformers. Imprime, por cada
paso, lo que dice el guion y lo que entendió Whisper. Whisper también se equivoca:
busca los términos técnicos que salen deformados en varios pasos (Git → "Hit") y
corrígelos en PRONUNCIACION de motor.py o reformulando la frase.
"""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import motor  # noqa: E402  (necesita pillow, pygments y fonttools, que también están en .venv)

MODELO = "openai/whisper-small"


def cargar(mp3):
    pcm = subprocess.run(["ffmpeg", "-v", "quiet", "-i", str(mp3), "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(pcm, np.float32)


def main(ruta):
    meta, escenas = motor.cargar_guion(ruta)
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    proc = WhisperProcessor.from_pretrained(MODELO)
    modelo = WhisperForConditionalGeneration.from_pretrained(MODELO).to(dispositivo).eval()
    voz, velocidad = meta.get("voz", "es-EC-AndreaNeural"), meta.get("velocidad", "+0%")
    for ne, _, np_, paso, _ in motor.recorrer(escenas):
        texto_voz = motor.para_voz(paso.decir)
        clave = hashlib.sha1(f"{voz}|{velocidad}|{texto_voz}".encode()).hexdigest()[:16]
        mp3 = motor.SALIDA / "cache_tts" / f"{clave}.mp3"
        if not mp3.exists():
            print(f"e{ne:02d}_p{np_:02d}: sin audio; construye el video primero")
            continue
        audio, partes = cargar(mp3), []
        for i in range(0, len(audio), 16000 * 28):  # Whisper escucha ventanas de 30 s
            entrada = proc(audio[i : i + 16000 * 28], sampling_rate=16000, return_tensors="pt").input_features
            ids = modelo.generate(entrada.to(dispositivo), language="es", task="transcribe")
            partes.append(proc.batch_decode(ids, skip_special_tokens=True)[0].strip())
        print(f"e{ne:02d}_p{np_:02d}  guion:   {paso.decir}\n         whisper: {' '.join(partes)}\n")


if __name__ == "__main__":
    main(sys.argv[1])
