import json,csv,pathlib,collections,math,hashlib
import numpy as np
import pandas as pd
OUT=pathlib.Path(__file__).parent
rows=json.loads(OUT.joinpath('parsed.json').read_text())
def writecsv(name,rs):
    if not rs:return
    keys=list(dict.fromkeys(k for r in rs for k in r))
    with OUT.joinpath(name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader()
        for r in rs:w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
seen={}; good=[]; excluded=[]
for r in rows:
    if r['status']!='complete':
        excluded.append({k:r.get(k) for k in ('source_file','source_row','run_id','status','version','attack_name','error')});continue
    if r['run_id'] in seen:
        old=seen[r['run_id']]
        assert all(old[k]==r[k] for k in ('metrics','attack_trials','methodology'))
        assert {k:v for k,v in old['config'].items() if k!='firestore_collection'}=={k:v for k,v in r['config'].items() if k!='firestore_collection'}
        excluded.append(dict(source_file=r['source_file'],source_row=r['source_row'],run_id=r['run_id'],status='duplicate',version=r['version'],attack_name=r['attack_name'],error='Identical metrics, trials and methodology; configuration differs only in Firestore collection. Retained first source.'));continue
    seen[r['run_id']]=r;good.append(r)
writecsv('excluded_records.csv',excluded)
writecsv('completed_runs.csv',[{k:v for k,v in r.items() if k!='raw_cells'} for r in good])
flat=[];trials=[];rag=[];ragtrials=[];issues=[]
def calculate(ts,sign=1):
    assert all(type(t['truth_member']) is bool and type(t['pred_member']) is bool and math.isfinite(t['score']) for t in ts)
    tp=sum(t['truth_member'] and t['pred_member'] for t in ts);fn=sum(t['truth_member'] and not t['pred_member'] for t in ts)
    fp=sum(not t['truth_member'] and t['pred_member'] for t in ts);tn=sum(not t['truth_member'] and not t['pred_member'] for t in ts)
    p=np.array([sign*t['score'] for t in ts if t['truth_member']]);n=np.array([sign*t['score'] for t in ts if not t['truth_member']])
    assert len(p) and len(n)
    auc=float(((p[:,None]>n).sum()+.5*(p[:,None]==n).sum())/(len(p)*len(n)))
    return dict(n=len(ts),members=len(p),nonmembers=len(n),tp=tp,fn=fn,fp=fp,tn=tn,tpr=tp/len(p),fpr=fp/len(n),balanced_accuracy=.5*(tp/len(p)+tn/len(n)),advantage=tp/len(p)-fp/len(n),auc=auc,constant_prediction=len({t['pred_member'] for t in ts})==1,unique_scores=len(set(t['score'] for t in ts)))
for r in good:
    c=r['config'];m=calculate(r['attack_trials'],-1 if r['attack_name']=='loss' else 1)
    ident={k:r[k] for k in ('run_id','source_file','source_row','version','attack_name')}
    f={**ident,**{k:c.get(k) for k in ('model_id','dataset_name','seed','num_clients','clients_per_round','federated_rounds','local_epochs','max_length','threshold','gradient_threshold','ldp_mechanism')},'rag_enabled':bool(r['pipeline'] and r['pipeline'].get('rag')),**m}
    for k,out in [('num_trials','n'),('accuracy','balanced_accuracy'),('tpr','tpr'),('roc_auc','auc'),('roc_auc_from_scores','auc'),('roc_auc_loss_inverted','auc')]:
        if k in r['metrics'] and r['metrics'][k] is not None and not math.isclose(r['metrics'][k],m[out],abs_tol=1e-10):issues.append((r['run_id'],k,r['metrics'][k],m[out]))
    assert len(r['attack_trials'])==c['attack_trials']
    flat.append(f)
    for t in r['attack_trials']:trials.append({**ident,**t})
    for p in r['pipeline_trials'] or []:
        for condition,x in p['rag_conditions'].items():
            cm=calculate(x['membership_trials'])
            for key,out in [('num_trials','n'),('accuracy','balanced_accuracy'),('tpr','tpr'),('roc_auc','auc')]:
                if key in x['metrics']:assert math.isclose(x['metrics'][key],cm[out],abs_tol=1e-10)
            rr={**ident,'fl_trial_id':p['trial_id'],'condition':condition,'rag_attack':x['attack'],'membership_target':x['membership_target'],'defense':r['pipeline']['defense']['mechanism'],**cm,**x['utility'],'recognized':sum(t['answer_recognized'] for t in x['membership_trials']),'hidden_document_queries':x['hidden_document_queries']}
            rag.append(rr)
            for t in x['membership_trials']:ragtrials.append({**ident,'fl_trial_id':p['trial_id'],'condition':condition,**t})
writecsv('run_metrics.csv',flat);writecsv('training_trials.csv',trials);writecsv('rag_conditions.csv',rag);writecsv('rag_trials.csv',ragtrials)
df=pd.DataFrame(flat)
summary=[]
for (v,a),g in df.groupby(['version','attack_name']):
    summary.append(dict(version=v,attack_name=a,runs=len(g),trials=int(g.n.sum()),mean_balanced_accuracy=g.balanced_accuracy.mean(),median_balanced_accuracy=g.balanced_accuracy.median(),mean_auc=g.auc.mean(),min_auc=g.auc.min(),max_auc=g.auc.max(),above_chance_runs=int((g.balanced_accuracy>.5+1e-9).sum()),auc_above_chance_runs=int((g.auc>.5+1e-9).sum()),constant_prediction_runs=int(g.constant_prediction.sum()),mean_tpr=g.tpr.mean(),mean_fpr=g.fpr.mean()))
sdf=pd.DataFrame(summary).sort_values(['version','mean_balanced_accuracy','mean_auc'],ascending=[True,False,False]);writecsv('attack_summary.csv',sdf.to_dict('records'))
print('COUNTS',len(rows),len(good),collections.Counter(r['status'] for r in excluded),'TRIALS',len(trials),'RAG',len(rag),len(ragtrials))
print('MISMATCHES',issues)
print(sdf.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
print('100 TRIAL RUNS');print(df[df.n==100].to_string(index=False))
print('CORRECTED');print(df[df.version=='theory_v2'].to_string(index=False))
print('AMIA UNIQUE SCORES')
for r in good:
    if r['attack_name']=='amia':print(r['version'],r['run_id'],len({t['score'] for t in r['attack_trials']}),r['config']['dataset_name'],r['config']['seed'])
base=dict(model_id='distilgpt2',dataset_name='squad',seed=7,num_clients=2,clients_per_round=2,federated_rounds=1,local_epochs=1,max_length=64)
print('CONTROLLED ONE FACTOR')
for factor in ('num_clients','local_epochs','federated_rounds','max_length','dataset_name','seed'):
    g=df[(df.version=='legacy') & (df.n==6)]
    for k,v in base.items():
        if k==factor or (factor=='num_clients' and k=='clients_per_round'):continue
        g=g[g[k]==v]
    print(factor);print(g.pivot_table(index='attack_name',columns=factor,values=['balanced_accuracy','auc']).round(3).to_string())
print('FAILED MODELS',collections.Counter((r['config']['model_id'],r['version']) for r in rows if r['status']=='failed'))
print('RAG SUMMARY',pd.DataFrame(rag).groupby('condition')[['balanced_accuracy','auc','tpr','fpr','exact_match','token_f1','recognized']].agg(['mean','sum']).to_string())
manifest={ 'input_files':[{ 'file':f.name,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(pathlib.Path('/tmp/codex-remote-attachments/01a09782-7523-79d3-b99a-ca1051d9e8aa/E212F4A6-54B6-4A7D-B35C-DDC7184C6C4C').glob('*.csv'))], 'input_rows':len(rows),'unique_completed_runs':len(good),'exclusions':dict(collections.Counter(r['status'] for r in excluded)),'metric_mismatches':issues,'training_trial_rows':len(trials),'rag_condition_rows':len(rag),'rag_trial_rows':len(ragtrials)}
OUT.joinpath('validation.json').write_text(json.dumps(manifest,indent=2))
assert not issues
