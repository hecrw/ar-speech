# Fine-tuned Models

Drop your three fine-tuned model folders from Google Drive in here. The app
auto-detects them; if any folder is missing, it falls back to the public
HuggingFace base model so the app still runs.

## Expected layout

```
models/
├── whisper-turbo-arabic/         ← from Drive: whisper-turbo-arabic/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── preprocessor_config.json
│   └── ...
│
├── mt5-arabic-summarization/     ← from Drive: mt5-arabic-summarization/
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   ├── spiece.model
│   └── ...
│
└── nomic-embed-arabic-search/    ← from Drive: nomic-embed-arabic-search/
    ├── config.json
    ├── modules.json
    ├── model.safetensors
    └── ...
```

## How to populate

The training notebooks in `notebooks/` save each model to:

| Notebook | Drive path |
|----------|------------|
| `train_whisper_arabic.ipynb` | `MyDrive/whisper-turbo-arabic/` |
| `train_silma_summarization.ipynb` | `MyDrive/mt5-arabic-summarization/` |
| `train_nomic_embed_arabic.ipynb` | `MyDrive/nomic-embed-arabic-search/` |

Download each folder from Drive and put it inside this `models/` directory
with **the same folder name**. The folder must contain `config.json` at its
root for the app to recognize it as a fine-tuned model.

## Verify the wire-up

After placing the folders, run:

```bash
python -c "import config; print('Whisper:    ', config.WHISPER_MODEL_ID); print('Summarizer:', config.SUMMARIZER_MODEL_ID); print('Embedder:  ', config.EMBEDDING_MODEL_ID)"
```

If a folder is recognized, you'll see a path like `models/whisper-turbo-arabic`.
If not, you'll see the HuggingFace ID fallback (e.g. `openai/whisper-large-v3-turbo`).

## Then launch

```bash
python app.py
```

Open <http://localhost:7860> in your browser.
