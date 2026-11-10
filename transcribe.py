import glob
import json
import os
import subprocess
import sys
import tempfile
import wave

import numpy as np
from faster_whisper import WhisperModel

CHUNK_SECONDS = 600  # split long audio into 10 minute pieces


def convert_to_wav(src, dst):
    # any format -> 16kHz mono wav (what whisper expects)
    subprocess.run(
        ["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", dst],
        check=True, capture_output=True,
    )


def split_audio(wav, folder):
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav, "-f", "segment",
         "-segment_time", str(CHUNK_SECONDS), "-c", "copy",
         os.path.join(folder, "chunk_%03d.wav")],
        check=True, capture_output=True,
    )
    return sorted(glob.glob(os.path.join(folder, "chunk_*.wav")))


def load_wav(path):
    # read the 16kHz mono wav and convert to float samples for whisper
    with wave.open(path) as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(path):
    model = WhisperModel("base", compute_type="int8")
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "audio.wav")
        convert_to_wav(path, wav)

        for i, chunk in enumerate(split_audio(wav, tmp)):
            print(f"Transcribing chunk {i + 1}...", file=sys.stderr)
            offset = i * CHUNK_SECONDS  # shift timestamps so they are global
            segments, info = model.transcribe(load_wav(chunk), vad_filter=True)
            for seg in segments:
                results.append({
                    "start": round(seg.start + offset, 2),
                    "end": round(seg.end + offset, 2),
                    "text": seg.text.strip(),
                })

    return results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python transcribe.py <audio file>")
        sys.exit(1)

    segments = transcribe(sys.argv[1])
    output = {
        "text": " ".join(s["text"] for s in segments),
        "segments": segments,
    }
    print(json.dumps(output, indent=2))