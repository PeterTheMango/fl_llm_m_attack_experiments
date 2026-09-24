"""Reproduce the raw smoke audit without fitting a detector or attack threshold."""
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
INPUT = ROOT / 'outputs/guard_v2_smoke_received_20260924/outputs/guard-v2-smoke-remote'
batch = INPUT / 'results/20260922-173328-2d75dea95957'
manifest = json.loads((batch / 'manifest.json').read_bytes())
previous = json.loads((batch / 'smoke-analysis-20260924-045023.json').read_bytes())
queue_hash = sha256((INPUT / 'experiments.yaml').read_bytes()).hexdigest()
assert all(e['source_sha256'] == queue_hash for e in manifest['entries'])
assert all(sha256((batch / Path(i['path']).name).read_bytes()).hexdigest() == i['sha256']
           for i in previous['inputs'])


def metrics(rows):
    available = [r for r in rows if r.get('score') is not None and r.get('pred_member') is not None]
    pos = [r for r in available if r['truth_member']]
    neg = [r for r in available if not r['truth_member']]
    if not pos or not neg:
        return {'balanced_accuracy': None, 'auc': None, 'available': len(available)}
    auc = sum((p['score'] > n['score']) + .5 * (p['score'] == n['score'])
              for p in pos for n in neg) / (len(pos) * len(neg))
    return {'balanced_accuracy': .5 * (sum(r['pred_member'] for r in pos) / len(pos)
                                       + sum(not r['pred_member'] for r in neg) / len(neg)),
            'auc': auc, 'available': len(available),
            'tpr': sum(r['pred_member'] for r in pos) / len(pos),
            'fpr': sum(r['pred_member'] for r in neg) / len(neg)}


runs = []
for entry in manifest['entries']:
    path = batch / entry['result_file']
    data = json.loads(path.read_bytes())
    assert data['status'] == entry['status'] == 'complete'
    assert data['run_id'] == entry['run_id']
    checked = metrics(data['attack_trials'])
    assert checked['balanced_accuracy'] == data['metrics']['adv']
    assert checked['auc'] == data['metrics']['roc_auc']
    grouped = defaultdict(list)
    for event in data.get('guard_events', []):
        grouped[event['scope']].append(event)
    traces = []
    for scope, events in grouped.items():
        features = [e['features'] for e in events if e.get('features') is not None]
        traces.append({'scope': scope, 'requests': len(events),
                       'accepted': sum(e['decision'] == 'accepted' for e in events),
                       'mean_seconds': statistics.mean(e['seconds'] for e in events),
                       'bytes_per_request': sorted({e['request_bytes'] for e in events}),
                       'features': {k: {'min': min(f[k] for f in features), 'max': max(f[k] for f in features)}
                                    for k in features[0]} if features else {},
                       'mean_stages_seconds': {k: statistics.mean(e['stages_seconds'][k] for e in events)
                                               for k in events[0]['stages_seconds']}})
    worlds = data['federated_history'] if data['attack_name'] == 'reference' else [{'rounds': data['federated_history']}]
    training = []
    for world in worlds:
        clients = defaultdict(list)
        for r in world['rounds']:
            for c in r['client_outcomes']:
                clients[c['client_id']].append(c['train_loss'])
        training.append({'round_mean_losses': [statistics.mean(c['train_loss'] for c in r['client_outcomes'])
                                               for r in world['rounds']],
                         'client_losses': clients,
                         'all_clients_decrease_each_round': all(all(b < a for a, b in zip(v, v[1:])) for v in clients.values())})
    rag = []
    for e in data['pipeline_evaluations']:
        conditions = {}
        for k, r in e['rag_conditions'].items():
            m = metrics(r['membership_trials'])
            assert m['balanced_accuracy'] == r['metrics']['adv']
            assert m['auc'] == r['metrics']['roc_auc']
            conditions[k] = {'metrics': m, 'utility': r['utility'], 'utility_queries': len(r['utility_trials'])}
        rag.append({'trial_id': e['trial_id'], 'conditions': conditions,
                    'no_retrieval_utility': e['no_retrieval_utility'], 'overlap': e['membership_overlap_cells']})
    runs.append({'file': path.name, 'sha256': sha256(path.read_bytes()).hexdigest(),
                 'run_id': data['run_id'], 'attack': data['attack_name'], 'variant': data.get('attack_variant'),
                 'condition': data['pipeline']['condition'], 'recomputed_metrics': checked,
                 'target_hashes': sorted({r.get('target_sha256') or r['candidate_sha256'] for r in data['attack_trials']}),
                 'seconds': data['computation_seconds'], 'training': training, 'traces': traces, 'rag': rag})

report = {'checks': {'queue_hash_matches_manifest': True, 'all_raw_hashes_match_prior_analysis': True,
                     'all_attack_and_rag_metrics_recomputed': True, 'run_count': len(runs)},
          'total_computation_seconds': sum(r['seconds'] for r in runs),
          'all_client_loss_sequences_decrease': all(t['all_clients_decrease_each_round'] for r in runs for t in r['training']),
          'independent_target_hashes': sorted({h for r in runs for h in r['target_hashes']}),
          'runs': runs,
          'feature_comparison_limit': 'Training features reference initialization; causal observation features reference the trained checkpoint. Ranges are diagnostic, not a detector evaluation.'}
(OUT / 'raw-summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
print(json.dumps({k: v for k, v in report.items() if k != 'runs'}, indent=2))
