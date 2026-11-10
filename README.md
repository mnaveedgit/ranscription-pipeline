# Audio Transcription Pipeline

A simple script that converts an audio file into text with a timestamp for every segment.

## Setup

1. Install Python 3.9 or newer.
2. Install ffmpeg (https://ffmpeg.org/download.html) and make sure it is on your PATH.
3. Create a virtual environment and install the dependencies:

```
py -m venv venv
venv\Scripts\activate
py -m pip install -r requirements.txt
```

## Usage

```
py transcribe.py sample.ogg
```

The first run downloads the Whisper `base` model, so it takes a bit longer.

Output (JSON):

```json
{
  "text": "Hello and welcome to the meeting. Today we will review the roadmap.",
  "segments": [
    {"start": 0.0, "end": 3.2, "text": "Hello and welcome to the meeting."},
    {"start": 3.2, "end": 7.8, "text": "Today we will review the roadmap."}
  ]
}
```

## How it works

1. **Convert**: ffmpeg converts the input (OGG, MP3, WAV, M4A, FLAC, even video) to 16 kHz mono WAV.
2. **Split**: the WAV is cut into 10-minute chunks so long files use little memory.
3. **Load**: each chunk is read directly as audio samples.
4. **Transcribe**: the samples go through faster-whisper with voice activity detection.
5. **Merge**: each chunk's timestamps are shifted by its offset, so they are correct for the full recording.
6. **Output**: JSON with the full text and per-segment start/end times.

## Design decisions

- **faster-whisper**: open source, runs locally (no API cost, audio stays private), faster and lighter than the original Whisper. The `base` model with int8 runs on a normal CPU; a bigger model or GPU can be used for more accuracy.
- **ffmpeg normalization**: one canonical input format means no per-format code and consistent results.
- **Reading the WAV directly**: the script already controls the conversion, so it loads the samples itself instead of relying on another decoder inside the library.
- **VAD filter**: skips silence, which saves time and reduces hallucinated text.
- **Chunking**: keeps memory low, makes retries cheap, and allows parallel processing later.
