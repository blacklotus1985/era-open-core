import argparse, os, yaml
from .pipeline import run_pipeline
from .report import ai_act_report

def main():
    p = argparse.ArgumentParser("era")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run"); r.add_argument("--config", required=True)
    rep = sub.add_parser("report"); rep.add_argument("--config", required=True); rep.add_argument("--out", required=False)

    args = p.parse_args()
    if args.cmd == "run":
        run_pipeline(args.config)
    elif args.cmd == "report":
        with open(args.config,'r') as f:
            cfg = yaml.safe_load(f)
        csv_path = os.path.join(cfg.get("reports_dir","results"), "ERA_report.csv")
        out = args.out or os.path.join(cfg.get("reports_dir","results"), "AIAct_Report.pdf")
        models = cfg['generator_models']
        base = next(m for m in models if m.get('role')=='baseline')
        cand = next(m for m in models if m.get('role')!='baseline')
        meta = {"baseline_model": base['name'], "candidate_model": cand['name'], "csi_alert": cfg.get('thresholds',{}).get('csi_alert')}
        ai_act_report(csv_path, out, meta)
        print("Report:", out)
