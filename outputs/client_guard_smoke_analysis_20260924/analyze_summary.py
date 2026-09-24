"""Summarize the supplied analysis JSON; does not revalidate absent raw results."""
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
raw = (ROOT / 'smoke-analysis.received.json').read_bytes()
data = json.loads(raw)
provenance = {r['run_id']: r for r in data['provenance']}
runs = {r['run_id']: r for r in data['study_v2']['runs']}
conditions = data['conditions']
assert len(conditions) == len({r['run_id'] for r in conditions}) == 11
assert set(runs) == set(provenance) == {r['run_id'] for r in conditions}
assert len(data['inputs']) == 11
rows = []
for c in conditions:
    r, p = runs[c['run_id']], provenance[c['run_id']]
    guard = c.get('guard')
    if guard:
        assert sum(v['requests'] for v in guard['per_client'].values()) == guard['training_requests']
        assert sum(v['rejected'] for v in guard['per_client'].values()) == guard['training_rejected']
    rows.append({
        'run_id': c['run_id'], 'attack': c['attack'], 'variant': p['config'].get('attack_variant'),
        'condition': c['condition'], 'status': c['status'], 'balanced_accuracy': c['balanced_accuracy'],
        'observation_auroc': c['membership_auc'], 'world_mean_auroc': r['raw_privacy']['roc_auc'],
        'independent_targets': r['raw_privacy']['independent_groups'],
        'observations': p['config']['attack_trials'], 'release_coverage': c['release_coverage'],
        'no_context_f1': c['no_retrieval_f1'], 'absolute_f1_floor_pass': c['utility_valid'],
        'training_requests': guard['training_requests'] if guard else None,
        'training_rejected': guard['training_rejected'] if guard else None,
        'availability': r['availability'], 'paired_runtime': r['paired_runtime'],
        'guard_profile_seconds': r['guard_profile_seconds'],
        'rag': {k: {**{key: v[key] for key in ('utility','recognition_rate','refusal_rate')},
                    'auroc': v['privacy']['roc_auc'], 'document_groups': v['privacy']['independent_groups']}
                for k,v in r['rag'].items()},
        'overlap': r['overlap']})
report = {
    'source_sha256': sha256(raw).hexdigest(),
    'source_scope': 'User-supplied derived analysis, not raw result JSONs or remote manifest',
    'checks': {'eleven_unique_runs_joined': True, 'per_client_guard_totals_consistent': True},
    'raw_metrics_independently_recomputed': False,
    'reported_source_fingerprints': sorted({p['implementation_fingerprint'] for p in provenance.values()}),
    'reported_raw_input_hashes': data['inputs'], 'runs': rows,
    'reported_benign_requests': sum(r['training_requests'] or 0 for r in rows),
    'reported_benign_rejections': sum(r['training_rejected'] or 0 for r in rows),
    'reported_guarded_completed_rounds': sum(r['availability']['completed_rounds'] or 0 for r in rows),
    'next_stage': 'Inspect saved traces and optimize measured guard costs before broad collection',
    'gates': {'round_overhead': 'descriptively above 10% target in every guarded arm',
              'no_context_absolute_utility': 'below provisional 0.10 F1 floor in all arms',
              'relative_utility': 'reported F1 and EM differences are zero for all matched slices',
              'benign_rejection': 'zero observed; insufficient independent evidence for 1% population gate'},
    'limitations': ['One shared target across AMIA variants and Reference pair',
                    'No classifier evaluated', 'No fitted held-out joint transcript attacker result supplied',
                    'Raw scores, request features, per-round losses, absolute computation times and token/context audits are absent',
                    'RAG document-bootstrap results condition on the tested corpus/checkpoints']}
(ROOT / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print('Joined 11 runs; 96 reported benign requests, 24 reported guarded rounds; raw result audit still pending.')
