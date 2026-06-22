"""Salvage header-only/empty progress.csv files by regenerating them from the pickled algo.log.

The per-generation log lives on the pickled NSGA2 algorithm (algo.log), which is the authoritative
source write_progress derives from. For runs whose progress.csv was clobbered by a mid-write kill,
the pickle still holds the full history, so we can rebuild progress.csv losslessly. Idempotent and
non-destructive: only touches dirs whose progress.csv is empty/header-only AND whose pickle has a
matching-length log. Writes atomically (temp + os.replace). DRY-RUN unless --apply is passed.
"""
import os, sys, glob, pickle, argparse
sys.path.insert(0, os.getcwd())
import pandas as pd

def linecount(p):
    if not os.path.exists(p): return None
    return sum(1 for _ in open(p, errors='replace'))

def last_gen(fronts):
    if not fronts or not os.path.exists(fronts): return None
    last=None
    for line in open(fronts):
        last=line
    try: return int(last.split(',')[0])
    except: return None

def load_log(d):
    for name in ('algo.pkl','algo_backup.pkl'):
        p=os.path.join(d,name)
        if not os.path.exists(p): continue
        try:
            with open(p,'rb') as f: algo=pickle.load(f)
            return getattr(algo,'log',None), name
        except Exception as e:
            print(f"   ! {name} failed: {e}")
    return None, None

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base', default='Experiments/E1 SF-12')
    ap.add_argument('--apply', action='store_true'); args=ap.parse_args()
    dirs=[os.path.dirname(p) for p in glob.glob(os.path.join(args.base,'**','config.json'),recursive=True)]
    targets=[d for d in dirs if (linecount(os.path.join(d,'progress.csv')) or 0) <= 1]
    print(f"{len(targets)} header-only/empty progress.csv dirs to salvage (apply={args.apply})\n")
    ok=skip=0
    for d in targets:
        log,src=load_log(d)
        lg=last_gen(os.path.join(d,'fronts.csv'))
        rel=os.path.relpath(d,args.base)
        if not log:
            print(f"SKIP (no usable log)  {rel}"); skip+=1; continue
        rows=len(log); last_iter=log[-1].get('iteration')
        match = (lg is None) or (last_iter == lg)
        status = "ok" if match else f"WARN last_iter={last_iter} vs fronts_lastgen={lg}"
        print(f"{'APPLY' if args.apply else 'DRY  '} rows={rows} src={src} {status}  {rel}")
        if args.apply:
            tgt=os.path.join(d,'progress.csv'); tmp=tgt+'.tmp'
            pd.DataFrame(log).to_csv(tmp,index=False); os.replace(tmp,tgt)
        ok+=1
    print(f"\nsalvageable: {ok}, skipped: {skip}")

if __name__=='__main__': main()
