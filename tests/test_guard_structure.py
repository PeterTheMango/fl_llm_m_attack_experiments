"""Math, fail-closed versioning and prospective collection for public features."""
from dataclasses import replace
from hashlib import sha256
import json
import numpy as np
import pytest
from master_script.core.guard_features import (FEATURE_NAMES, FEATURE_SCHEMA, STRUCTURE_SCHEMA,
    STRUCTURE_NAMES, parameter_features, feature_vector)
from master_script.core.guard_detector import fit_detector, detector_score, load_detector
from master_script.core.guard_runtime import prepare_guard, read_guard_events, parse_guard
from master_script.tools import collect_guard_traces
from tests.test_guard_integration import settings
from tests.test_guard_detector import traces


def geometry(v, p):
    return parameter_features(v, p, STRUCTURE_SCHEMA)


def test_dense_oracle_and_legacy_projection():
    rng = np.random.default_rng(45)
    prior = [rng.normal(size=n) for n in [7, 91, 65539]]
    value = [p+rng.normal(size=p.size)*.03 for p in prior]
    got = geometry(value, prior)
    assert {k:got[k] for k in FEATURE_NAMES} == parameter_features(value, prior)
    cos = [np.dot(v-p,p)/np.linalg.norm(v-p)/np.linalg.norm(p) for v,p in zip(value,prior)]
    concentration = [np.max(np.abs(v-p))/np.linalg.norm(v-p) for v,p in zip(value,prior)]
    deltas = [np.linalg.norm(v-p)/np.linalg.norm(p) for v,p in zip(value,prior)]
    expected = [np.median(deltas),np.quantile(deltas,.9),np.mean(cos),min(cos),max(cos),np.mean(concentration),max(concentration)]
    np.testing.assert_allclose([got[k] for k in STRUCTURE_NAMES[3:]], expected, rtol=1e-12, atol=1e-14)


def test_equal_v1_features_can_hide_opposite_update_directions():
    # Synthetic constructions prove information gain, not attack detection.
    p = [np.array([1., 1.])]
    positive = [np.array([2., 2.])]
    negative = [np.array([0., 0.])]
    a,b = geometry(positive,p),geometry(negative,p)
    assert a['relative_delta'] == b['relative_delta'] == 1
    assert a['mean_update_reference_cosine'] == pytest.approx(1)
    assert b['mean_update_reference_cosine'] == pytest.approx(-1)
    # Stronger exact v1 collision: reflect a fixed-length request on the circle
    # around p with equal max/L2 ratio by choosing different radii along p.
    positive = [np.array([3.,3.])]; negative = [np.array([-1.,-1.])]
    a,b = geometry(positive,p),geometry(negative,p)
    np.testing.assert_allclose([a[k] for k in FEATURE_NAMES],[b[k] for k in FEATURE_NAMES])
    assert a['mean_update_reference_cosine'] == pytest.approx(1)
    assert b['mean_update_reference_cosine'] == pytest.approx(-1)


def test_sparse_perturbation_and_benign_rescaling():
    p=[np.ones(64)]
    benign=geometry([p[0]*1.01],p)
    sparse=p[0].copy();sparse[5]+=.08
    attack=geometry([sparse],p)
    assert benign['relative_delta'] == pytest.approx(attack['relative_delta'])
    assert benign['maximum_update_concentration'] == pytest.approx(1/8)
    assert attack['maximum_update_concentration'] == 1
    assert geometry(p,p)['mean_update_concentration'] == 0
    assert geometry(p,p)['mean_update_reference_cosine'] == 0


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
def test_zero_and_extreme_finite_inputs(dtype):
    cases = [(np.zeros(3,dtype=dtype), np.zeros(3,dtype=dtype))]
    if dtype != np.int64:
        high=np.finfo(dtype).max; tiny=np.nextafter(dtype(0),dtype(1))
        cases += [(np.array([high,-high],dtype=dtype),np.array([-high,high],dtype=dtype)),
                  (np.array([tiny,0],dtype=dtype),np.zeros(2,dtype=dtype))]
    else:
        cases += [(np.array([2**62,-2**62],dtype=dtype),np.array([-2**62,2**62],dtype=dtype))]
    for v,p in cases:
        f=geometry([v],[p]);assert np.isfinite(feature_vector(f,STRUCTURE_SCHEMA)).all()
        assert -1 <= f['minimum_update_reference_cosine'] <= f['maximum_update_reference_cosine'] <= 1
        assert 0 <= f['maximum_update_concentration'] <= 1
    if dtype != np.int64:
        assert geometry([cases[1][0]],[cases[1][1]])['mean_update_reference_cosine'] == pytest.approx(-1)
        assert geometry([cases[2][0]],[cases[2][1]])['maximum_update_concentration'] == 1


def test_strided_chunked_and_malformed_inputs():
    p=np.arange(140000,dtype=float).reshape(400,350).T
    v=(p+1)[:,::2];p=p[:,::2]
    assert geometry([v],[p]) == geometry([v.copy()],[p.copy()])
    for bad in [np.full(v.shape,np.nan),np.full(v.shape,np.inf),np.ones(5),np.array(['x'])]:
        with pytest.raises(ValueError):geometry([bad],[p])
    with pytest.raises(ValueError):geometry([v],[])
    with pytest.raises(ValueError):feature_vector({**geometry([v],[p]),'target_id':0},STRUCTURE_SCHEMA)
    with pytest.raises(ValueError):feature_vector(geometry([v],[p]))
    with pytest.raises(ValueError):parameter_features([v],[p],'unknown')


@pytest.mark.parametrize('workers',[1,2,4])
def test_runtime_features_reference_binding_tampering_and_accounting(tmp_path,workers):
    rng=np.random.default_rng(12);p=[rng.normal(size=65539).astype('float32') for _ in range(4)]
    v=[x+np.float32(.001) for x in p]
    guard=replace(settings(tmp_path,'shadow'),feature_schema=STRUCTURE_SCHEMA,validation_workers=workers)
    runtime=prepare_guard(guard,'new',p,rounds=3)
    snapshot=runtime.authorize(v,0,1)
    event=read_guard_events(tmp_path/'local')[-1]
    assert event['features']==geometry(v,p)
    assert event['feature_schema']==STRUCTURE_SCHEMA
    assert event['reference_sha256']==sha256(''.join(runtime.reference_digests).encode()).hexdigest()
    v[0][:]=0
    assert not np.array_equal(v[0],snapshot[0])
    assert runtime.authorize(snapshot,0,1) is None
    assert read_guard_events(tmp_path/'local')[-1]['reservation_count']==1
    np.savez(runtime.reference_path,*[np.zeros_like(x) for x in p])
    with pytest.raises(ValueError,match='integrity'):runtime.authorize(snapshot,0,2)
    from master_script.core.release_ledger import ReleaseLedger
    assert ReleaseLedger(tmp_path/'local/release-ledger.sqlite').count('new:0')==1


def test_artifact_versions_are_explicit_and_final_data_cannot_tune(tmp_path):
    rows=traces()
    for row in rows:
        row['features'].update(dict.fromkeys(STRUCTURE_NAMES[3:],float(row['malicious'])))
    artifact=fit_detector(rows,schema=STRUCTURE_SCHEMA)
    assert artifact['schema']==STRUCTURE_SCHEMA
    for row in rows:
        if row['split']=='test':row['features']=dict.fromkeys(STRUCTURE_NAMES,123.)
    assert artifact==fit_detector(rows,schema=STRUCTURE_SCHEMA)
    with pytest.raises(ValueError):fit_detector(rows)
    path=tmp_path/'v2.json';raw=json.dumps(artifact).encode();path.write_bytes(raw)
    assert load_detector(path,sha256(raw).hexdigest())==artifact
    with pytest.raises(ValueError):detector_score(artifact,traces()[0]['features'])
    opts=settings(tmp_path)
    config={'mode':'classifier','policy_file':opts.policy_file,'release_budget':2,
            'detector_file':str(path),'detector_sha256':sha256(raw).hexdigest()}
    with pytest.raises(ValueError,match='schema mismatch'):parse_guard(config,'<config>')
    assert parse_guard({**config,'feature_schema':STRUCTURE_SCHEMA},'<config>').feature_schema==STRUCTURE_SCHEMA
    with pytest.raises(ValueError,match='schema'):parse_guard({**config,'feature_schema':'unknown'},'<config>')


def exclusion(tmp_path):
    path=tmp_path/'old.json'
    path.write_text(json.dumps({'schema':'guard_splits_v2','groups':{'historical':'test'}}))
    return path


def test_pilot_requires_exclusions_reserves_final_and_resolves_without_gpu(tmp_path,monkeypatch):
    from master_script.core import datasets
    with pytest.raises(ValueError,match='exclusions'):
        collect_guard_traces.prepare(tmp_path/'bad',feature_schema=STRUCTURE_SCHEMA)
    out=tmp_path/'pilot';old=exclusion(tmp_path)
    launch=collect_guard_traces.prepare(out,targets=3,seed=4000,feature_schema=STRUCTURE_SCHEMA,
        exclude_manifests=[old],reserve_final_targets=2)
    assert len(launch['jobs'])==9 and len(launch['reserved_final'])==2
    assert 'historical' in launch['excluded_targets']
    monkeypatch.setattr(datasets,'target_record_for',lambda config,fallback:f'target-{config.seed}')
    monkeypatch.setattr(collect_guard_traces.subprocess,'run',lambda *a,**kw:pytest.fail('GPU launched'))
    collect_guard_traces.execute(out/'launch.json',resolve_only=True)
    state=json.loads((out/'progress.json').read_bytes())
    assert len(state['targets'])==3 and len(state['reserved_final_targets'])==2
    assert not set(state['targets'].values()) & set(state['reserved_final_targets'].values())
    assert state['completed']==[] and state['active_job'] is None
    collect_guard_traces.execute(out/'launch.json',resolve_only=True)
    monkeypatch.setattr(datasets,'target_record_for',lambda config,fallback:'collision')
    with pytest.raises(ValueError,match='same target'):
        collect_guard_traces.execute(out/'launch.json',resolve_only=True)


def test_final_reservation_collision_is_rejected(tmp_path,monkeypatch):
    from master_script.core import datasets
    out=tmp_path/'pilot'
    collect_guard_traces.prepare(out,targets=3,seed=4000,feature_schema=STRUCTURE_SCHEMA,
        exclude_manifests=[exclusion(tmp_path)],reserve_final_targets=2)
    monkeypatch.setattr(datasets,'target_record_for',lambda c,f:f'target-{c.seed if c.seed<5000 else 4000}')
    with pytest.raises(ValueError,match='Reserved final cohort overlaps'):
        collect_guard_traces.execute(out/'launch.json',resolve_only=True)


def test_dataset_schema_and_reserved_roles_are_fail_closed(tmp_path):
    from master_script.tools.assemble_guard_dataset import assemble
    from master_script.tools.build_guard_dataset import build_dataset
    event={'scope':'training:4000:True','features':geometry([np.ones(4)],[np.ones(4)]),
           'feature_schema':STRUCTURE_SCHEMA,'client_id':0,'round_id':1}
    result={'implementation_fingerprint':'fixture','attack_trials':[{'seed':4000,'target_sha256':'new'}],
            'guard_events':[event],'config':{}}
    source=tmp_path/'source.json';raw=json.dumps(result).encode();source.write_bytes(raw)
    manifest={'schema':'guard_splits_v2','sources':[{'path':'source.json','sha256':sha256(raw).hexdigest()}],
              'groups':{'new':'train'},'held_out_variants':['causal_gradient_alignment']}
    with pytest.raises(ValueError,match='schemas'):assemble(manifest,tmp_path)
    rows,metadata=assemble(manifest,tmp_path,STRUCTURE_SCHEMA)
    assert len(rows)==1 and metadata['feature_schema']==STRUCTURE_SCHEMA
    manifest['groups']['new']='final'
    rows,metadata=assemble(manifest,tmp_path,STRUCTURE_SCHEMA)
    assert rows==[] and metadata['excluded']['final']==1
    labels=[{'event_index':0,'group':'new','split':'train','malicious':False,'variant':'benign'}]
    assert build_dataset([event],labels,STRUCTURE_SCHEMA)[0]['features']==event['features']
    with pytest.raises(ValueError,match='schemas'):build_dataset([event],labels)


@pytest.mark.parametrize('scale',[1.,1e-35,1e35])
def test_float32_single_pass_matches_scaled_geometry(scale):
    from master_script.core.guard_features import _update_geometry
    rng=np.random.default_rng(88)
    p=(rng.uniform(-1,1,size=65539)*scale).astype('float32')
    v=(p.astype('float64')+rng.uniform(-.01,.01,size=p.size)*scale).astype('float32')
    f=geometry([v],[p]);expected=_update_geometry(v,p)
    np.testing.assert_allclose([f['mean_update_reference_cosine'],f['maximum_update_concentration']],expected,rtol=1e-10,atol=1e-12)


def test_pilot_pins_orchestration_and_frozen_cohort(tmp_path,monkeypatch):
    from master_script.core import datasets
    out=tmp_path/'pilot'
    collect_guard_traces.prepare(out,targets=3,seed=4000,feature_schema=STRUCTURE_SCHEMA,
        exclude_manifests=[exclusion(tmp_path)],reserve_final_targets=2)
    original=collect_guard_traces.collection_tool_digest
    monkeypatch.setattr(collect_guard_traces,'collection_tool_digest',lambda:'changed')
    with pytest.raises(ValueError,match='orchestration'):
        collect_guard_traces.execute(out/'launch.json',resolve_only=True)
    monkeypatch.setattr(collect_guard_traces,'collection_tool_digest',original)
    monkeypatch.setattr(datasets,'target_record_for',lambda c,f:f'target-{c.seed}')
    collect_guard_traces.execute(out/'launch.json',resolve_only=True)
    cohort=out/'cohort.json';data=json.loads(cohort.read_bytes())
    assert list(data['groups'].values()).count('final')==2
    data['groups'][next(iter(data['groups']))]='final';cohort.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='Frozen cohort changed'):
        collect_guard_traces.execute(out/'launch.json',resolve_only=True)
