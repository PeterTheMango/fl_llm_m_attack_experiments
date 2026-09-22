from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from master_script.core.guard_runtime import prepare_guard, read_guard_events, read_training_rounds
from master_script.core.guard_features import parameter_features
from master_script.core.guard_statistics import (collapse_worlds, fit_attack_calibration, apply_attack_calibration,
    privacy_report, fit_transcript_calibration, apply_transcript_calibration)
from tests.test_guard_integration import settings


def test_reference_tampering_fails_before_release(tmp_path):
    runtime = prepare_guard(settings(tmp_path), "train", [np.ones(2)], rounds=2)
    np.savez(runtime.reference_path, np.zeros(2))
    with pytest.raises(ValueError, match="integrity"):
        runtime.authorize([np.ones(2)], 0, 1)
    from master_script.core.release_ledger import ReleaseLedger
    assert ReleaseLedger(tmp_path / "local/release-ledger.sqlite").count("train:0") == 0


def test_authorized_snapshot_is_owned_and_later_round_delta_visible(tmp_path):
    runtime = prepare_guard(settings(tmp_path), "train", [np.ones(2)], rounds=2)
    incoming = [np.ones(2)]
    checked = runtime.authorize(incoming, 0, 1)
    incoming[0][:] = np.nan
    assert np.isfinite(checked[0]).all()
    assert runtime.authorize([np.ones(2)*100], 0, 2) is not None
    events = read_guard_events(tmp_path / "local")
    assert events[-1]["features"]["relative_delta"] == pytest.approx(99.)
    assert all(v >= 0 for v in events[-1]["stages_seconds"].values())


def test_reference_arrays_read_once_per_validation(monkeypatch, tmp_path):
    runtime = prepare_guard(settings(tmp_path), "train", [np.ones(2),np.ones(3)], rounds=1)
    original = np.lib.npyio.NpzFile.__getitem__; reads=[]
    def traced(self, key):
        reads.append(key); return original(self, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", traced)
    assert runtime.check([np.ones(2),np.ones(3)], 0, 1)
    assert reads == ["arr_0", "arr_1"]


def test_feature_math_preserved_at_extreme_scale():
    for scale in (1e-15, 1., 1e150):
        x=np.array([1.,2.,3.])*scale; y=np.array([3.,1.,2.])*scale
        f=parameter_features([x], [y])
        assert all(np.isfinite(list(f.values())))
        if scale >= 1:
            assert f["relative_delta"] == pytest.approx(np.sqrt(6/14))
            assert f["parameter_concentration"] == pytest.approx(3/np.sqrt(14))


def validation_rows(prefix="val", count=40):
    return [{"group": f"{prefix}-{i}", "truth_member": member,
             "score": .1 if member else .9, "pred_member": False, "decision": "accepted",
             "response_seconds": .02 if member else .01}
            for i in range(count) for member in (True,False)]


def test_reversed_direction_only_learned_from_validation():
    val = validation_rows(); frozen=fit_attack_calibration(val, target_fpr=.05)
    assert frozen["direction"] == -1
    final=validation_rows("final")
    applied=apply_attack_calibration(final,frozen)
    assert privacy_report(applied,draws=20)["roc_auc"] == 1
    for row in final: row["truth_member"] = not row["truth_member"]
    assert frozen["direction"] == -1
    assert privacy_report(apply_attack_calibration(final,frozen),draws=20)["roc_auc"] == 0
    with pytest.raises(ValueError,match="overlap"):
        apply_attack_calibration(val,frozen)
    with pytest.raises(ValueError,match="binding"):
        apply_attack_calibration(final,frozen,binding={"revision":"changed"})


def test_repeats_do_not_supply_low_fpr_resolution():
    rows=validation_rows(count=1)*100
    assert len(collapse_worlds(rows)) == 2
    with pytest.raises(ValueError,match="count"):
        fit_attack_calibration(rows,target_fpr=.01)
    report=privacy_report(collapse_worlds(rows),draws=10)
    assert report["independent_groups"] == 1 and report["interval_95"] is None
    assert report["low_fpr_resolution"] == 1


def test_joint_transcript_attack_includes_rejections_without_fabricated_gradient():
    val=validation_rows(count=100)
    for r in val:
        r.update(score=None,pred_member=None,decision="rejected")
    frozen=fit_transcript_calibration(val,target_fpr=.05)
    test=deepcopy(val)
    for r in test: r["group"]="test-"+r["group"]
    observed=apply_transcript_calibration(test,frozen)
    assert privacy_report(observed,draws=20)["roc_auc"] == 1
    assert privacy_report(collapse_worlds(test),draws=20)["roc_auc"] is None
    assert all(r["score"] is None for r in test)


def test_rag_parser_handles_refusal_echoes_and_ambiguity():
    from master_script.core.rag import parse_membership_answer, response_metrics, continuation_question
    for text in ("I cannot confirm. Yes or no is private.", "unrelated output", "Yes/No", "The passage contains yes."):
        assert parse_membership_answer(text)["pred_member"] is None
    assert parse_membership_answer("Answer: No.")["pred_member"] is False
    rows=[{"truth_member":b,"score":None,"pred_member":None} for b in (True,False)]
    m=response_metrics(rows)
    assert m["roc_auc"] is None and m["accuracy"] is None and m["attempted_queries"] == 2
    query,suffix=continuation_question("one two three four five six")
    assert suffix not in query


def test_raw_rag_answers_stay_local_with_restrictive_permissions(tmp_path):
    from master_script.core.rag import audit_answer
    audit_answer(SimpleNamespace(runtime_audit_directory=str(tmp_path)), {"answer":"private research text"})
    path=tmp_path/"rag-answers.jsonl"
    assert path.stat().st_mode & 0o777 == 0o600
    assert json.loads(path.read_text())["answer"] == "private research text"


def test_cpu_multiround_training_abort_and_restart(tmp_path):
    from master_script.tools.guard_local_smoke import run
    output=tmp_path/"smoke";report=run(output)
    assert report["convergence_improved"]
    rounds=read_training_rounds(output/"client-owned")
    assert [r["status"] for r in rounds] == ["completed"]*4+["policy_aborted"]
    reasons=[e["reason"] for e in report["guard_events"]]
    assert reasons[-2:] == ["duplicate_request","budget_exhausted"]


def test_stages_cover_both_attacks_and_pins(tmp_path):
    from master_script.tools.build_guard_stage import build_stage, ROOT
    from master_script.core.yaml_config import load_config_doc
    from tests.test_guard_detector import traces
    from master_script.core.guard_detector import fit_detector
    path=tmp_path/"detector.json";path.write_text(json.dumps(fit_detector(traces())))
    counts={"smoke":11,"collection":3,"validation":20,"confirmation":11}
    for stage,count in counts.items():
        plan=json.loads((ROOT/f"guard/stages/{stage}_v2.json").read_text())
        if plan["requires_detector"]:
            with pytest.raises(ValueError,match="requires"):
                build_stage(plan)
        pairs=load_config_doc(build_stage(plan,detector=path if plan["requires_detector"] else None))
        assert len(pairs)==count
        assert {spec.name for _,spec in pairs} == {"amia","reference"}
        for cfg,spec in pairs:
            assert cfg.federated_rounds >= 3
            if spec.pipeline.condition == "rules_classifier_noise":
                assert cfg.observation_defense == "gaussian" and cfg.observation_noise_multiplier >= 1.
                assert spec.pipeline.client_guard.detector_sha256 == sha256(path.read_bytes()).hexdigest()


def test_pairing_never_mixes_training_hyperparameters_or_code():
    from master_script.tools.analyze_guard_study import pairing_key
    a={"attack_name":"amia","implementation_fingerprint":"v1","config":{"seed":7,"client_lr":.1}}
    b=deepcopy(a);b["config"]["client_lr"]=.2
    assert pairing_key(a)!=pairing_key(b)
    b=deepcopy(a);b["implementation_fingerprint"]="v2"
    assert pairing_key(a)!=pairing_key(b)


def test_causal_alignment_observes_only_released_gradients():
    from master_script.core.attacks.causal_probe import alignment_score
    direction=[np.array([1.,0.])]
    assert alignment_score([np.array([2.,10.])],direction)==2
    assert alignment_score([np.array([0.,10.])],direction)==0
    with pytest.raises(ValueError): alignment_score([np.ones(3)],direction)


def test_causal_request_preserves_architecture_with_tiny_torch_model():
    torch=pytest.importorskip("torch")
    from master_script.core.attacks.causal_probe import optimize_request, raw_gradients
    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__();self.emb=torch.nn.Embedding(8,4);self.head=torch.nn.Linear(4,8)
        def forward(self,input_ids,attention_mask=None,labels=None):
            logits=self.head(self.emb(input_ids))
            loss=torch.nn.functional.cross_entropy(logits[:,:-1].reshape(-1,8),labels[:,1:].reshape(-1))
            return SimpleNamespace(loss=loss)
    def tokenizer(text,*args,**kwargs):
        batch=len(text) if isinstance(text,list) else 1
        return {"input_ids":torch.tensor([[1,2,3]]*batch),"attention_mask":torch.ones(batch,3,dtype=torch.long)}
    model=Tiny();before={k:(v.shape,v.dtype) for k,v in model.state_dict().items()}
    cfg=SimpleNamespace(max_length=8,probe_epochs=2,probe_lr=.1)
    history=optimize_request(model,tokenizer,"known public candidate",cfg)
    assert history[-1] > history[0]
    assert before == {k:(v.shape,v.dtype) for k,v in model.state_dict().items()}
    assert len(raw_gradients(model,tokenizer,["candidate"],cfg)) == len(before)


def test_counterbalancing_keeps_pair_identity_and_world_balance():
    from master_script.core.attacks.amia import trial_member, AmiaConfig
    cfg=AmiaConfig(counterbalance_trials=True)
    orders=[]
    for pair in range(20):
        values=[trial_member(cfg,2*pair+i) for i in range(2)]
        assert set(values)=={False,True}
        orders.append(values[0])
    assert set(orders)=={False,True}


def test_dataset_assembly_enforces_sources_roles_and_heldout_variants(tmp_path):
    from master_script.tools.assemble_guard_dataset import assemble
    sources=[]
    for variant in ("probe_head","causal_gradient_alignment"):
        result={"implementation_fingerprint":"same", "config":{"attack_variant":variant},
                "attack_trials":[{"target_sha256":"one","target_seed":7},{"target_sha256":"two","target_seed":8}],
                "guard_events":[{"scope":f"{kind}:{seed}","client_id":0,"round_id":1,
                                 "features":parameter_features([np.ones(2)],[np.ones(2)])}
                                for seed in (7,8) for kind in ("training","observation")]}
        raw=json.dumps(result).encode();path=tmp_path/f"{variant}.json";path.write_bytes(raw)
        sources.append({"path":path.name,"sha256":sha256(raw).hexdigest()})
    manifest={"schema":"guard_splits_v2","sources":sources,"groups":{"one":"train","two":"test"},
              "held_out_variants":["causal_gradient_alignment"]}
    rows,metadata=assemble(manifest,tmp_path)
    assert not any(r["variant"]=="causal_gradient_alignment" and r["split"]=="train" for r in rows)
    assert any(r["variant"]=="causal_gradient_alignment" and r["split"]=="test" for r in rows)
    assert metadata["excluded"]["held_out_variant"]==1
    manifest["groups"]["two"]="final"
    rows,_=assemble(manifest,tmp_path)
    assert all(r["group"]=="one" for r in rows)
    (tmp_path/sources[0]["path"]).write_text("changed")
    with pytest.raises(ValueError,match="checksum"):
        assemble(manifest,tmp_path)


def test_overlap_constant_member_bias_is_not_interaction():
    from master_script.tools.analyze_guard_study import overlap_report
    cells=[{"target_sha256":"t","defense":"ordinary","training_member":t,"datastore_member":d,"pred_member":True}
           for t in (False,True) for d in (False,True)]
    report=overlap_report(cells*3)
    assert report["independent_targets"]==1
    assert report["per_target"][0]["interaction"]==0
    assert report["per_target"][0]["constant_prediction"]


def test_worker_processes_share_one_durable_budget(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import subprocess, sys
    from master_script.core.release_ledger import ReleaseLedger
    path=tmp_path/"ledger.sqlite";ReleaseLedger(path)
    script = "from master_script.core.release_ledger import ReleaseLedger; import sys; print(ReleaseLedger(sys.argv[1]).reserve('client',sys.argv[2],2))"
    def child(index):
        return subprocess.run([sys.executable,"-c",script,str(path),str(index)],check=True,
                              capture_output=True,text=True).stdout.strip()
    with ThreadPoolExecutor(max_workers=4) as workers:
        outcomes=list(workers.map(child,range(6)))
    assert outcomes.count("None")==2 and outcomes.count("budget_exhausted")==4
    assert ReleaseLedger(path).count("client")==2


def test_detector_mutation_is_not_hidden_by_a_cached_load(tmp_path):
    from master_script.core.guard_runtime import parse_guard
    from master_script.core.guard_detector import fit_detector
    from tests.test_guard_detector import traces
    local=settings(tmp_path)
    detector=fit_detector(traces());detector["threshold"]=1.
    path=tmp_path/"detector.json";path.write_text(json.dumps(detector))
    opts=replace(parse_guard({"mode":"classifier","policy_file":local.policy_file,"release_budget":2,
                             "diagnostic":True,"detector_file":str(path),
                             "detector_sha256":sha256(path.read_bytes()).hexdigest()},"<config>"),
                 runtime_directory=local.runtime_directory)
    runtime=prepare_guard(opts,"obs",[np.ones(2)],rounds=2)
    assert runtime.authorize([np.ones(2)],0,1) is not None
    detector["intercept"]+=1;path.write_text(json.dumps(detector))
    with pytest.raises(ValueError,match="SHA256"):
        runtime.authorize([np.ones(2)],0,2)


def test_full_model_victim_rejection_precedes_private_access(monkeypatch,tmp_path):
    """Actual nested causal-variant client, with ML/Flower imports stubbed."""
    import sys
    from types import ModuleType
    from master_script.core.attacks import amia, causal_probe
    from master_script.core import runtime_memory
    class Client:
        def to_client(self): return self
    class App:
        def __init__(self,client_fn=None,server_fn=None): self.client_fn,self.server_fn=client_fn,server_fn
    class Strategy:
        def __init__(self,**kwargs): pass
    exports={"flwr":{},"flwr.client":{"NumPyClient":Client,"ClientApp":App},
             "flwr.common":{"ndarrays_to_parameters":lambda a:a,"parameters_to_ndarrays":lambda a:a},
             "flwr.server":{"ServerApp":App,"ServerAppComponents":SimpleNamespace,"ServerConfig":SimpleNamespace},
             "flwr.server.strategy":{"FedAvg":Strategy},
             "transformers":{"AutoModelForCausalLM":object,"AutoTokenizer":object}}
    for name,values in exports.items():
        mod=ModuleType(name);mod.__dict__.update(values);monkeypatch.setitem(sys.modules,name,mod)
    monkeypatch.setattr(amia,"get_parameters",lambda p:[np.ones(2)])
    monkeypatch.setattr(causal_probe,"protected_gradients",lambda *a:pytest.fail("Private gradient reached after rejection"))
    runtime=prepare_guard(settings(tmp_path),"obs",[np.ones(2)],rounds=1)
    # Spend both world's single round request; restarting the same observation
    # cannot bypass the ledger, even for an architecture-preserving request.
    for world in (0,1):
        assert replace(runtime,scope=f"obs:world-{world}").check([np.ones(2)],0,1)
    def simulate(server_app,client_app,**kwargs):
        strategy=server_app.server_fn(None).strategy
        client=client_app.client_fn(SimpleNamespace(node_config={"partition-id":0}))
        for i in range(2):
            arrays,n,metrics=client.fit([np.ones(2)],{"trial_id":i,"guard_mode":"off","release_budget":999})
            assert arrays==[] and n==0
            strategy.aggregate_fit(i+1,[(None,SimpleNamespace(parameters=arrays,num_examples=n,metrics=metrics))],[])
    monkeypatch.setattr(runtime_memory,"run_simulation",simulate)
    monkeypatch.setattr(runtime_memory,"simulation_backend",lambda c:{})
    config=SimpleNamespace(num_clients=1,target_client_id=0,attack_trials=2,gradient_threshold=0.,seed=7,
                           attack_variant="causal_gradient_alignment")
    probe=SimpleNamespace(_public_direction=[np.ones(2)])
    rows=amia.run_attack_trials("unused",probe,[["private"]],config,guard_runtime=runtime)
    assert all(r["score"] is None and r["pred_member"] is None for r in rows)
