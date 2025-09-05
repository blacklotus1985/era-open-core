# ERA (Ethical Rating of Algorithms): Concept Drift & Lineage Auditing — Open-Core

**Goal**: misurare e documentare come cambia il *significato* dei concetti nei modelli AI tra versioni (drift concettuale), con:
- **CSI (Concept Shift Index)** = embedding drift + lexical probability shift (bounded, interpretable).
- **Lineage graph** = genealogia di modelli (foundational / fine-tune / architectural siblings).
- **Protocollo black-box** = probe neutrali, candidate lexicon per concetto, controlli statistici, report AI Act-ready.

Questo repo è pensato per essere **riproducibile**, **testato**, e **pronto per la comunità** (open-core).

## 🚀 Quickstart
```bash
# 1) Setup
python -m venv .venv && source .venv/bin/activate
pip install -e .

# (opzionale) backend vendor
pip install "era-open-core[openai]" "era-open-core[anthropic]"
export OPENAI_API_KEY=...  # se usi OpenAI

# 2) Esegui una demo
python -m era.cli run --config examples/configs/demo_hf.yaml

# 3) Genera un report AI Act-ready (PDF)
python -m era.cli report --config examples/configs/demo_hf.yaml --out results/AIAct_Report.pdf
```

## 📦 Struttura
```
src/era/           # libreria
examples/          # config, notebook, script di run
docs/              # metodologia, paper summary
tests/             # unit & smoke tests
results/           # output (gitignored)
figures/           # figure generate
```

## 📐 Metodologia (sintesi)
- **Probe**: ≥3 parafrasi neutre per concetto.
- **Embedding drift**: 1 - coseno sulle medie degli embedding delle risposte (encoder fisso).
- **Probability shift**: media delle log-prob sull’ultimo passo per un *candidate lexicon* del concetto → exp+renorm → KL simmetrico + Wasserstein.
- **CSI**: combinazione normalizzata con pesi per dominio; allerta se CSI ≥ soglia.
- **Controlli**: sentinelle stabili, placebo (M vs M), bootstrap CI, FDR, change-point su serie CSI.

Vedi `docs/methodology.md` per i dettagli e formule.

## 🧪 Reproducibilità
- Seed e impostazioni deterministiche per le **misure** (temperature≤0.2).
- Config YAML → stessa esperienza su modelli diversi.
- Test unitari: `pytest -q` (tagga `-m "not internet"` su CI).

## 📊 Esempi inclusi
- HF baseline vs candidate (distilgpt2 → gpt2) su domini: HR, Healthcare, Banking, PA.
- Lineage graph example + CSI per concetto (figure salvate in `figures/`).

## 🧭 Come citare
Usa `CITATION.cff` o:
```
@misc{ERA_open_core_2025,
  title = {ERA: Concept Drift & Lineage Auditing (Open-Core)},
  year = {2025},
  author = {Your Name},
  howpublished = {GitHub repository},
  url = {https://github.com/blacklotus1985/era-open-core}
}
```

## 🛡️ Licenza
Apache-2.0. Vedi `LICENSE`.

## 💬 Community
Apri issue / PR. Vedi `CONTRIBUTING.md` e `CODE_OF_CONDUCT.md`.
