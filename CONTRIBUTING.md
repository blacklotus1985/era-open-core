# Contributing to ERA Open-Core

## Ground rules
- Scrivi test per ogni feature (`tests/`), preferibilmente senza internet o con marker `@pytest.mark.internet`.
- Segui Black/PEP8. Esegui `pre-commit` se disponibile.
- Documenta nuove metriche in `docs/` e aggiungi esempi in `examples/`.

## Dev setup
```
python -m venv .venv && source .venv/bin/activate
pip install -e .[openai,anthropic]
pip install pytest
```

## Test
```
pytest -q -m "not internet"
pytest -q -m internet  # opzionale
```
