from typing import BinaryIO, Union, Annotated
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import StreamingResponse, JSONResponse
import numpy as np
import ffmpeg
from transformers import pipeline
import torch

SAMPLE_RATE = 16000

app = FastAPI(title="ASR Server", description="ASR Server using HuggingFace Transformers", version="0.1.0")


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/asr")
async def asr(
        audio_file: UploadFile = File(...),
        encode: bool = Query(default=True, description="Encode audio first through ffmpeg"),
        task: Union[str, None] = Query(default="transcribe", enum=["transcribe", "translate"]),
        initial_prompt: Union[str, None] = Query(default=None),
        word_timestamps: bool = Query(default=False, description="Word level timestamps"),
        output: Union[str, None] = Query(default="txt", enum=["txt", "vtt", "srt", "tsv", "json"])
):
    result = transcribe(load_audio(audio_file.file, encode)) #, task, initial_prompt, word_timestamps, output)
    return JSONResponse(
        content=result['text'],
        headers={
            'Content-Disposition': f'attachment; filename="{audio_file.filename}.{output}"'
        })


def transcribe(
        audio_file: BinaryIO,
):
    pipe = pipeline("automatic-speech-recognition",
                    model='openai/whisper-large-v3',
                    device='cpu',
                    torch_dtype=torch.float32)
    result = pipe(audio_file, chunk_length_s=30, batch_size=32, return_timestamps=True)
    print('ASR Server transcribe 3', result)
    return result


def load_audio(file: BinaryIO, encode=True, sr: int = SAMPLE_RATE):
    """
    Open an audio file object and read as mono waveform, resampling as necessary.
    Modified from https://github.com/openai/whisper/blob/main/whisper/audio.py to accept a file object
    Parameters
    ----------
    file: BinaryIO
        The audio file like object
    encode: Boolean
        If true, encode audio stream to WAV before sending to whisper
    sr: int
        The sample rate to resample the audio if necessary
    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    if encode:
        try:
            # This launches a subprocess to decode audio while down-mixing and resampling as necessary.
            # Requires the ffmpeg CLI and `ffmpeg-python` package to be installed.
            out, _ = (
                ffmpeg.input("pipe:", threads=0)
                .output("-", format="s16le", acodec="pcm_s16le", ac=1, ar=sr)
                .run(cmd="ffmpeg", capture_stdout=True, capture_stderr=True, input=file.read())
            )
        except ffmpeg.Error as e:
            raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e
    else:
        out = file.read()

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
