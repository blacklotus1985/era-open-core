import os, yaml
import numpy as np
import pandas as pd
from .embeddings import Embedder
from .probers.hf import HFProber
try:
    from .probers.openai import OpenAIProber
except Exception:
    OpenAIProber = None
try:
    from .probers.anthropic import AnthropicProber
except Exception:
    AnthropicProber = None
from .metrics import cosine_distance, symmetric_kl, w1
from .csi import combine_csi, CSIWeights
from .concept_library import ConceptLibrary
from .lineage import build_lineage, save_graph
import matplotlib.pyplot as plt

def make_prober(spec: dict):
    backend = spec.get('backend','hf')
    if backend == 'hf':
        return HFProber(spec['name'], spec.get('revision'))
    if backend == 'openai':
        if OpenAIProber is None: raise ImportError("Install openai extra")
        return OpenAIProber(spec['name'], spec.get('base_url'), spec.get('api_key'))
    if backend == 'anthropic':
        if AnthropicProber is None: raise ImportError("Install anthropic extra")
        return AnthropicProber(spec['name'], spec.get('api_key'))
    raise ValueError(f"Unsupported backend {backend}")

def run_pipeline(config_path: str):
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    reports = cfg.get('reports_dir','results')
    os.makedirs(reports, exist_ok=True)

    emb = Embedder(cfg['embedding_model'])
    models = cfg['generator_models']
    base = next(m for m in models if m.get('role') == 'baseline')
    cand = next(m for m in models if m.get('role') != 'baseline')
    pb = make_prober(base); pc = make_prober(cand)

    lib = ConceptLibrary(cfg['concepts_file'])
    with open(cfg['probes_file'], 'r') as f:
        probes = yaml.safe_load(f)

    rows = []
    for domain, cname, centry in lib.iter_concepts():
        candidates = centry['candidates']
        domain_prompts = [p['prompt'] for p in probes.get('global', [])]
        domain_prompts += [p['prompt'] for p in probes.get(domain, [])]

        out_b, out_c = [], []
        for p in domain_prompts:
            pr = p.format(concept=cname)
            out_b.append(pb.generate(pr, max_new_tokens=40))
            out_c.append(pc.generate(pr, max_new_tokens=40))
        E_b = emb.encode(out_b); E_c = emb.encode(out_c)
        d_emb = cosine_distance(E_b, E_c)

        scores_b, scores_c = [], []
        for p in domain_prompts:
            pr = p.format(concept=cname)
            lb = pb.candidate_logprobs(pr, candidates)
            lc = pc.candidate_logprobs(pr, candidates)
            scores_b.append([lb[x] for x in candidates])
            scores_c.append([lc[x] for x in candidates])
        sb = np.exp(np.array(scores_b)).mean(axis=0); sb = sb/sb.sum() if sb.sum() else sb
        sc = np.exp(np.array(scores_c)).mean(axis=0); sc = sc/sc.sum() if sc.sum() else sc
        skl = symmetric_kl(sb, sc) if sb.sum() and sc.sum() else 0.0
        w = w1(sb, sc) if sb.sum() and sc.sum() else 0.0
        d_prob = 0.5*skl + 0.5*w

        wts = CSIWeights(**cfg.get('csi_weights', {'embedding':0.5, 'prob':0.5}))
        csi = combine_csi(d_emb, d_prob, wts)
        alert = csi >= cfg.get('thresholds',{}).get('csi_alert', 0.35)

        rows.append({
            "domain": domain, "concept": cname,
            "embedding_distance": float(d_emb),
            "prob_shift": float(d_prob),
            "CSI": float(csi),
            "alert": bool(alert),
            "baseline_model": base['name'],
            "candidate_model": cand['name']
        })

    df = pd.DataFrame(rows)
    out_csv = os.path.join(reports, 'ERA_report.csv')
    df.to_csv(out_csv, index=False)

    # Simple figures
    for domain in df['domain'].unique():
        d = df[df['domain']==domain]
        plt.figure(figsize=(8,4))
        plt.bar(d['concept'], d['CSI'])
        plt.title(f"CSI per concetto – {domain}")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(reports, f"CSI_{domain}.png"))
        plt.close()

    G = build_lineage([{"name": m['name'], "role": m.get('role',"candidate"), "parent": m.get('parent')} for m in models])
    save_graph(G, os.path.join(reports, "lineage_graph.png"))

    return out_csv
