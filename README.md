# Audio Speech Recognition Server

Work with [Obsidian Transcription Plugin](https://github.com/djmango/obsidian-transcription) as https://github.com/ahmetoner/whisper-asr-webservice did.

## Run

``` 
poetry install
python -m gunicorn -k uvicorn.workers.UvicornWorker --bind 127.0.0.1:9000 --workers 1 --timeout 0 --reload main:app
```

## License

MIT

## Acknowledgments

- This tool is powered by Hugging Face's ASR models, primarily Whisper by OpenAI.
- Inspired from https://github.com/ochen1/insanely-fast-whisper-cli
