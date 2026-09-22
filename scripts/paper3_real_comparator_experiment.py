"""Real, repeatable comparator evaluation from locked Stage-10 artifacts."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

ROOT=Path(r"E:\Nodel\ExerciseProject\AutoResearchClaw")
CSV=Path(r"E:\Nodel\ExerciseProject\paper_3\data\data.csv")
LOCK=ROOT/"artifacts"/"paper3_scaffold_clinical"/"write_ahead_lock.json"
JOIN=ROOT/"artifacts"/"paper3_scaffold_clinical"/"post_lock_outcome_join.json"
OUT=ROOT/"artifacts"/"paper3_arc23"/"real_comparator_experiment"

def boot(y,s,seed):
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(2000):
        ix=rng.integers(0,len(y),len(y))
        if len(set(y[ix]))==2: vals.append(roc_auc_score(y[ix],s[ix]))
    return float(roc_auc_score(y,s)),[float(np.quantile(vals,.025)),float(np.quantile(vals,.975))]

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(CSV,encoding="utf-8-sig")
    y=(pd.to_numeric(df['抗凝药后是否出血'],errors='coerce')==1).astype(int).to_numpy()
    lock=json.loads(LOCK.read_text(encoding='utf-8'))['records']
    join=json.loads(JOIN.read_text(encoding='utf-8'))
    if len(lock) != len(df) or len(join) != len(df):
        raise RuntimeError('locked scores and post-lock outcome join must each cover every encounter')
    y=np.array([int(x['recorded_bleeding']) for x in join])
    locked_scores=np.array([float(r['assessment']['risk_score']) for r in lock])
    conditions={
      'SCAFFOLD_locked_LLM':locked_scores/100,
      'HAS_BLED':pd.to_numeric(df['has—bled'],errors='coerce').fillna(0).to_numpy(),
      'Caprini':pd.to_numeric(df['caprini'],errors='coerce').fillna(0).to_numpy(),
      'prevalence_constant':np.full(len(y),y.mean()),
      'rank_IQR_sanity':np.argsort(np.argsort(locked_scores))/(len(y)-1),
    }
    all_runs=[]
    for name,s in conditions.items():
      for seed in (101,202,303):
        auc,ci=boot(y,s,seed)
        all_runs.append({'condition':name,'seed':seed,'n':int(len(y)),'events':int(y.sum()),'auc':auc,'auc_bootstrap_ci':ci,'auprc':float(average_precision_score(y,s)),'brier':float(brier_score_loss(y,np.clip(s,0,1)))})
    (OUT/'per_condition_seed_results.json').write_text(json.dumps(all_runs,ensure_ascii=False,indent=2),encoding='utf-8')
    (OUT/'manifest.json').write_text(json.dumps({'source':'outcome-blind write-ahead lock plus independent post-lock join; SPSS-labelled CSV only supplies fixed-score comparators','conditions':list(conditions),'seeds':[101,202,303],'bootstrap_resamples':2000},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'runs':len(all_runs),'conditions':list(conditions)},ensure_ascii=False))
if __name__=='__main__': main()
