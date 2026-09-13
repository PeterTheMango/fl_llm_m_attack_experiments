import csv, json, pathlib, collections
ROOT=pathlib.Path('/tmp/codex-remote-attachments/01a09782-7523-79d3-b99a-ca1051d9e8aa/E212F4A6-54B6-4A7D-B35C-DDC7184C6C4C')
OUT=pathlib.Path(__file__).parent
rows=[]
for f in sorted(ROOT.glob('*.csv')):
    raw=list(csv.reader(f.open(encoding='utf-8-sig')))
    for i,cells in enumerate(raw[1:],2):
        r={'source_file':f.name,'source_row':i,'field_count':len(cells),'run_id':cells[0]}
        for k in ('config','metrics','methodology','attack_trials','federated_history','artifacts','pipeline','pipeline_trials','training_privacy_metrics','certificate','training_provenance','privacy_accounting','probe_training_loss'):r[k]=None
        r['raw_cells']=cells
        if cells[0]=='monitor_state':
            r.update(status='metadata',attack_name=None,version=None,error='');rows.append(r);continue
        statuses=[v for v in cells if v in ('complete','failed','running','pending')]
        assert len(statuses)==1,(i,statuses)
        r['status']=statuses[0]
        for s in cells[1:]:
            try: v=json.loads(s)
            except (json.JSONDecodeError,TypeError):continue
            key=None
            if isinstance(v,dict):
                if 'model_id' in v and 'num_clients' in v:key='config'
                elif 'tpr' in v and 'tnr' in v:key='metrics'
                elif 'paper_attack' in v:key='methodology'
                elif 'defense' in v and 'rag' in v:key='pipeline'
                elif 'artifact_dir' in v or 'probe_path' in v:key='artifacts'
                elif 'low_fpr_resolution' in v:key='training_privacy_metrics'
                elif 'certified_attack' in v:key='certificate'
                elif 'target_token_sha256' in v:key='training_provenance'
                elif 'accountant' in v and 'epsilon' in v:key='privacy_accounting'
            elif isinstance(v,list) and v and isinstance(v[0],dict):
                if 'truth_member' in v[0] and 'pred_member' in v[0]:key='attack_trials'
                elif 'rag_conditions' in v[0]:key='pipeline_trials'
                elif any(k in v[0] for k in ('rounds','history','selected_clients')):key='federated_history'
            elif isinstance(v,list) and v and all(isinstance(x,(float,int)) for x in v):key='probe_training_loss'
            if key:
                assert r[key] is None,(i,key)
                r[key]=v
        assert r['config'] is not None,i
        r['attack_name']=r['config'].get('attack_name') or ('loss' if 'threshold_quantile' in r['config'] else 'amia' if 'probe_epochs' in r['config'] else None)
        assert r['attack_name'],i
        r['version']='theory_v2' if r['run_id'].startswith('theory_v2') else 'pipeline_v1' if r['run_id'].startswith('pipeline_v1') else 'legacy'
        r['error']=next((s for s in cells if s.startswith(('Got unsupported','[Errno','An error','Traceback'))),'')
        r['updated_at_unix']=next((int(s) for s in reversed(cells) if s.isdigit() and len(s)==10),None)
        if r['status']=='complete':assert r['attack_trials'] and r['metrics'],i
        rows.append(r)
OUT.joinpath('parsed.json').write_text(json.dumps(rows,indent=2))
print('STATUS',collections.Counter(r['status'] for r in rows))
print('VERSION STATUS',collections.Counter((r['version'],r['status']) for r in rows))
print('DUP IDS',len(rows)-len({r['run_id'] for r in rows}))
print('ERRORS',collections.Counter(r['error'] for r in rows if r['status']=='failed'))
good=[r for r in rows if r['status']=='complete']
for k in ('model_id','dataset_name','attack_trials','seed','num_clients','clients_per_round','federated_rounds','local_epochs','max_length'):
    print(k,collections.Counter(r['config'].get(k) for r in good))
for r in good:
    print(r['version'],r['attack_name'],r['run_id'],r['config']['attack_trials'],r['metrics'].get('adv'),r['metrics'].get('roc_auc',r['metrics'].get('roc_auc_from_scores',r['metrics'].get('roc_auc_loss_inverted'))))
