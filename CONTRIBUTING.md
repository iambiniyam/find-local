# Contributing

Thanks for taking a look. Small, focused pull requests are easiest to review.

## Development setup

```bash
uv venv
source .venv/bin/activate
uv pip install -e ".[dev,models]"
pytest -q
```

The test suite uses the deterministic hash embedder, so it does not download the real models. To try the full pipeline, install the models extra and run `find-local index examples`.

## Before opening a pull request

- Run the test suite.
- Add a test for any indexing or search change.
- Update the README if a command, flag, or output changes.

## Model downloads

The default models are downloaded from Hugging Face. If the download stalls, set `HF_HUB_DISABLE_XET=1` and use the mirror with `HF_ENDPOINT=https://hf-mirror.com`.

## Bug reports

Include the command you ran, the Python version, the operating system, and the size of the index. Do not attach private documents. A small folder of fake files is enough to reproduce most bugs.
