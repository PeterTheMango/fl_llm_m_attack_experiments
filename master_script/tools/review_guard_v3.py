"""Reproduce frozen guard-v3 results without fitting or modifying its detector."""
import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
import tarfile
import subprocess


def digest(raw):
    return sha256(raw).hexdigest()


def review(downloads, output):
    downloads, output = Path(downloads), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs = {}
    names = ['progress.json', 'splits.complete.json', 'guard-workers-20260926-060354.json',
             '0000-theory_v2_7d9f9493f7c3_pipeline_v1_49612f2d54dbc42393c1344f.json']
    raw = (downloads / 'detector-review.tar.gz').read_bytes()
    inputs['detector-review.tar.gz'] = digest(raw)
    (output / 'detector-review.tar.gz').write_bytes(raw)
    with tarfile.open(downloads / 'detector-review.tar.gz') as archive:
        for name in ['dataset.json', 'dataset.json.provenance.json', 'detector.json', 'detector.json.report.json']:
            raw = archive.extractfile(name).read()
            inputs[name] = digest(raw)
            (output / name).write_bytes(raw)
    for name in names:
        raw = (downloads / name).read_bytes(); inputs[name] = digest(raw)
        (output / name).write_bytes(raw)
    read = lambda name: json.loads((output / name).read_bytes())
    rows, provenance, detector, old_report = map(read, ['dataset.json', 'dataset.json.provenance.json', 'detector.json', 'detector.json.report.json'])
    splits, progress, workers, source = map(read, ['splits.complete.json', 'progress.json', names[2], names[3]])
    assert inputs['dataset.json'] == provenance['dataset_sha256'] == old_report['dataset_sha256']
    assert inputs['detector.json'] == old_report['detector_sha256']
    assert digest(json.dumps(splits, sort_keys=True).encode()) == provenance['manifest_sha256']
    assert progress['active_job'] is None and len(progress['completed']) == 24
    assert {r['job'] for r in progress['completed']} == set(range(24))
    assert {(r['result'], r['sha256']) for r in progress['completed']} == {(r['path'], r['sha256']) for r in splits['sources']}
    assert set(progress['targets'].values()) == set(splits['groups']) and len(set(progress['targets'].values())) == 8
    assert len(rows) == len(provenance['provenance']) == 424
    source_hashes = {s['sha256'] for s in splits['sources']}
    for row, trace in zip(rows, provenance['provenance'], strict=True):
        assert row['group'] == trace['target_sha256']
        assert splits['groups'][row['group']] == row['split']
        assert trace['source_sha256'] in source_hashes
        assert row['group'] in detector['split_groups'][row['split']]
        assert set(row['features']) == set(detector['features'])
        assert row['malicious'] == (row['variant'] != 'legitimate_training')
        assert not (row['split'] != 'test' and row['variant'] == 'causal_gradient_alignment')
    # Independent scalar implementation; no repository scoring/metric helpers.
    def score(row):
        z = detector['intercept'] + sum((row['features'][k]-m)/s*w for k,m,s,w in
            zip(detector['features'],detector['mean'],detector['scale'],detector['weights'],strict=True))
        return 1/(1+math.exp(-max(-700,min(700,z))))
    train = [r for r in rows if r['split'] == 'train']
    for k, mean, scale in zip(detector['features'], detector['mean'], detector['scale'], strict=True):
        values = [r['features'][k] for r in train]
        calculated = sum(values)/len(values)
        sd = math.sqrt(sum((x-calculated)**2 for x in values)/len(values))
        assert math.isclose(calculated, mean, rel_tol=1e-12)
        assert math.isclose(sd, scale, rel_tol=1e-12)
    val_benign = [r for r in rows if r['split'] == 'validation' and not r['malicious']]
    assert math.isclose(max(map(score,val_benign)),detector['threshold'],rel_tol=1e-12)
    baseline = max(r['features']['relative_delta'] for r in val_benign)
    test = [r for r in rows if r['split'] == 'test']
    decisions = [score(r)>detector['threshold'] for r in test]
    assert decisions == [r['features']['relative_delta']>baseline for r in test]
    positive = [score(r) for r in test if r['malicious']]
    negative = [score(r) for r in test if not r['malicious']]
    auc = sum((p>n)+.5*(p==n) for p in positive for n in negative)/len(positive)/len(negative)
    variants = {}
    for variant in sorted({r['variant'] for r in test}):
        subset = [r for r in test if r['variant'] == variant]
        variants[variant] = {'requests':len(subset), 'rejected':sum(score(r)>detector['threshold'] for r in subset),
            'score_range':[min(map(score,subset)),max(map(score,subset))],
            'feature_ranges':{k:[min(r['features'][k] for r in subset),max(r['features'][k] for r in subset)] for k in detector['features']}}
    available_hash = inputs[names[3]]
    assert available_hash in source_hashes
    matched = 0
    for row, trace in zip(rows,provenance['provenance'],strict=True):
        if trace['source_sha256'] != available_hash:
            continue
        event = source['guard_events'][trace['event_index']]
        assert row['features'] == event['features']
        assert row['malicious'] == event['scope'].startswith('observation:')
        assert trace['client_id'] == event['client_id'] and trace['round'] == event['round_id']
        matched += 1
    utility = source['pipeline_evaluations']
    base = 'e2b812ad8807d97543ee54bea2483ebe1019f08b'
    historical = sha256()
    historical_sources = {}
    paths = subprocess.check_output(['git','ls-tree','-r','--name-only',base,'master_script/core'],text=True).splitlines()
    for name in sorted(p for p in paths if p.endswith('.py')):
        raw = subprocess.check_output(['git','show',base+':'+name])
        historical.update(str(Path(name).relative_to('master_script/core')).encode());historical.update(raw)
        historical_sources[name] = digest(raw)
    assert historical.hexdigest()[:12] == provenance['implementation_fingerprint']
    report = {'schema':'guard_v3_negative_review_v1','inputs_sha256':inputs,
        'historical_code':{'commit':base,'implementation_fingerprint':historical.hexdigest()[:12],
                           'source_sha256':historical_sources},
        'counts':dict(Counter(r['split']+':'+r['variant'] for r in rows)),
        'independent_groups':{k:len(v) for k,v in detector['split_groups'].items()},
        'test':{'roc_auc':auc,'attack_recall':sum(decisions[i] for i,r in enumerate(test) if r['malicious'])/len(positive),
            'per_variant':variants,'decisions_equal_relative_delta_baseline':True},
        'threshold':detector['threshold'],'baseline_threshold':baseline,
        'checks':{'artifact_hashes':True,'manifest_progress_provenance':True,'train_only_mean_scale':True,
            'threshold_equals_validation_benign_maximum':True,'available_source_rows_reproduced':matched},
        'limitations':['23 raw result sources and original launch/configs are absent locally; their contents and prospective timing cannot be independently re-audited.',
            'Retired weights are unavailable; historical features cannot be recomputed from parameters.',
            'Request-origin labels are not membership labels. Two test targets do not certify population false rejection.',
            'Collection completion/retirement for other jobs is supported by progress metadata, not direct remote disk inspection.'],
        'historical_worker_median_seconds':workers['median_wall_seconds'],
        'historical_worker_hash_feature_equivalence':workers['features_and_hashes_identical'],
        'first_job_mean_training_guard_seconds':sum(e['seconds'] for e in source['guard_events'] if e['accounting_scope']=='training')/12,
        'first_job_utility_f1':{'no_retrieval':sum(e['no_retrieval_utility']['token_f1'] for e in utility)/len(utility),
            **{k:sum(e['rag_conditions'][k]['utility']['token_f1'] for e in utility)/len(utility) for k in ['public_ordinary','private_ordinary']}},
        'source_audit_sha256':{str(p):digest(p.read_bytes()) for p in sorted(Path('master_script/core').rglob('guard*.py'))}}
    assert math.isclose(auc,old_report['test']['roc_auc'])
    (output/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('downloads');p.add_argument('output')
    args=p.parse_args();report=review(args.downloads,args.output)
    print(json.dumps(report['test'],indent=2))


if __name__ == '__main__':
    main()
