import json,tarfile,hashlib
from pathlib import Path
from statistics import mean
archive=Path('/Users/pyeshuajs23/Downloads/reference-light-results.tar.gz')
out=Path(__file__).resolve().parent
with tarfile.open(archive) as t:
 data={m.name:json.load(t.extractfile(m)) for m in t.getmembers() if m.isfile() and m.name.endswith('.json')}
manifest=data.pop('manifest.json')
def check(rows,m):
 counts={k:0 for k in ['tp','tn','fp','fn']}
 for r in rows: counts['tp' if r['truth_member'] and r['pred_member'] else 'fn' if r['truth_member'] else 'fp' if r['pred_member'] else 'tn']+=1
 assert all(counts[k]==m[k] for k in counts)
 pos=[r['score'] for r in rows if r['truth_member']];neg=[r['score'] for r in rows if not r['truth_member']]
 auc=mean([float(a>b)+.5*(a==b) for a in pos for b in neg])
 assert abs(auc-m['roc_auc'])<1e-9
 assert abs((counts['tp']/len(pos)+counts['tn']/len(neg))/2-m['adv'])<1e-9
 return counts
results=[]
base=None
for filename,d in sorted(data.items()):
 if d['status']!='complete': continue
 rows=d['attack_trials'];check(rows,d['metrics'])
 signature=[(r['trial_id'],r['seed'],r['candidate_sha256'],r['truth_member'],r['training_provenance']) for r in rows]
 if base is None:base=signature
 assert signature==base
 for r in rows:
  assert r['pred_member']==(r['score']>=r['threshold'])
  p=r['training_provenance'];present=any(p['target_token_sha256'] in v for v in p['partition_token_sha256'].values())
  assert present==r['truth_member']==r['target_exposed']
 for a,b in zip(rows[::2],rows[1::2]):
  assert a['seed']==b['seed'] and a['candidate_sha256']==b['candidate_sha256']
 evaluations=[e for e in d['pipeline_evaluations'] if e.get('rag_conditions')]
 rag={}
 for name in evaluations[0]['rag_conditions']:
  rs=[e['rag_conditions'][name] for e in evaluations]
  for r in rs:
   check(r['membership_trials'],r['metrics'])
   for k,v in r['utility'].items():assert abs(mean(x[k] for x in r['utility_trials'])-v)<1e-8
  rag[name]={**{k:mean(r['metrics'][k] for r in rs) for k in ['adv','tpr','tnr','roc_auc']},**{k:mean(r['utility'][k] for r in rs) for k in ['exact_match','token_f1','answer_nll']},'recognition_rate':mean(r['diagnostics']['recognition_rate'] for r in rs)}
 results.append({'condition':d['pipeline']['condition'],'source_file':filename,'run_id':d['run_id'],'metrics':d['metrics'],'rag':rag,'rag_validity':[e['rag_validity'] for e in evaluations],'privacy':d.get('training_privacy_composed'),'firestore_saved':d['firestore_saved']})
summary={'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'manifest':manifest,'checks':'Recomputed confusion matrices, balanced accuracy and AUROC for training and RAG; recomputed RAG utility means; verified training exposure, paired candidates and identical training provenance across conditions. All checks passed.','runs':results}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
lines=['# Reference light pilot: results analysis','',f"All {len(results)} experiments completed; none were excluded as failed. Each contains 20 training attack trials (10 member/nonmember pairs). All five files report successful Firestore saving; this is not an independent check of the remote database.",'',summary['checks'],'','## Training membership inference','','The archive field `adv` is balanced accuracy: (TPR + TNR)/2. It is not TPR minus FPR. Chance-level balanced accuracy is 50%. AUROC below is recomputed from pooled raw scores across separately trained models; per-model calibrated decisions and confusion counts should also be considered.','','| Condition | TP / TN / FP / FN | Balanced accuracy | AUROC |','|---|---|---:|---:|']
for r in results:
 m=r['metrics'];lines.append(f"| {r['condition']} | {m['tp']} / {m['tn']} / {m['fp']} / {m['fn']} | {100*m['adv']:.1f}% | {m['roc_auc']:.2f} |")
lines+=['','Baseline succeeds on every tested member and nonmember. All defended conditions detect zero members at their calibrated thresholds. This supports suppression of this particular attack in this pilot, not a general privacy guarantee or a reliable ranking between the defenses.','','## RAG membership and answer quality','','These are datastore-membership tests, distinct from the Reference training-membership attack. Means below cover two fine-tuned models per condition, evaluated on the same study. Each corpus condition has 200 candidate decisions per model and only 10 utility questions. Repeated evaluation of the same questions is not additional independent question coverage. Answer F1 measures utility; higher is better. Attack balanced accuracy measures discrimination; lower towards 50% is desirable only when utility and response recognition remain adequate.','','| Training condition | RAG condition | Attack balanced accuracy | TPR | FPR | Answer F1 | Exact match | Recognized answers |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in results:
 for name,m in r['rag'].items():lines.append(f"| {r['condition']} | {name} | {100*m['adv']:.2f}% | {100*m['tpr']:.1f}% | {100*(1-m['tnr']):.1f}% | {100*m['token_f1']:.2f}% | {100*m['exact_match']:.1f}% | {100*m['recognition_rate']:.2f}% |")
lines+=['','## Interpretation','','1. **Training privacy and retrieval privacy separate in these results.** SGD sigma 1 reduces Reference balanced accuracy from 100% to 50%, while ordinary private-datastore attack accuracy rises from 63.5% to 72.5%. Protecting training does not establish protection of retrieval outputs. This is an observed association in this pilot, not proof that SGD generally increases leakage.','2. **FedAvg settings fail the utility requirement.** Both have zero answer F1 and exact match in ordinary public/private RAG. Membership responses are almost never recognized, and unrecognized answers are scored as nonmembers. Their near-50% RAG scores therefore do not establish a useful defense. Retain these completed runs as utility failures, but exclude them from claims of successful usable RAG protection.','3. **A refusal instruction can improve attack discrimination.** With baseline training, private RAG balanced accuracy increases from 63.5% to 69.5% with the instruction. True-positive rate falls from 94.5% to 71.5%, but false-positive rate falls more, from 67.5% to 32.5%. Reporting only fewer positive answers would hide this effect.','4. **Mirabel has a substantial utility cost.** For baseline private RAG it reduces attack balanced accuracy from 63.5% to 56.5%, while answer F1 drops from 53.72% to 17.84%. Combining Mirabel and instructions reaches 53.75% attack accuracy with 18.96% answer F1. This is a tradeoff, not an unqualified success.','5. **Ordinary RAG attack sensitivity is not reliability at low false-positive rates.** Baseline private RAG detects 94.5% of members but falsely flags 67.5% of nonmembers. Its discrimination is much weaker than the baseline Reference result.','','## Limits on study claims','','There is one sweep seed (7), with ten paired trial seeds (7–16), one model and one dataset setup. These are ten paired targets, not twenty independent target samples. RAG covers just the first two trained models. No AMIA results are included. This archive cannot establish cross-model reliability, superiority over AMIA, or the general best defense.','','Only ten training nonmembers were tested, giving 10 percentage point empirical FPR resolution. Zero observed false positives is not proof of population FPR below 5% or 1%. The 200 calibration nonmembers are threshold-selection data, not additional held-out test outcomes.','','Saved training provenance and exposure flags agree with the intended labels and are matched across defenses. This is an internal consistency check, not an independent reconstruction of the entire training process. The archive does not contain generated answer text for qualitative error auditing.','','## Reported privacy accounting','','The following are composed bounds over training releases within each experiment at delta = 0.00001, not a bound for a single deployed model or the entire RAG system. They exclude private retrieval outputs and other experiments. SGD and FedAvg protect different units, so their epsilon values should not be used to rank equivalent protection. These very large bounds do not demonstrate strong formal privacy.','','| Condition | Composed epsilon |','|---|---:|']
for r in results:
 if r['privacy']:lines.append(f"| {r['condition']} | {r['privacy']['epsilon']:.2f} |")
lines+=['','## Study conclusion','','Reference is a promising successful attack baseline in this tested setting. The useful research question is now whether a defense can reduce both training and datastore membership inference while retaining answer quality. These results establish a pilot signal and expose unsuccessful defense settings; they do not yet prove broad attack reliability.','','Before a larger sweep, investigate the FedAvg utility collapse. For confirmation, repeat matched baseline/SGD comparisons across independent sweep seeds and include AMIA. Retain balanced accuracy, AUROC, TPR/FPR, answer F1, exact match, and response recognition together.','',f"Archive SHA-256: `{summary['archive_sha256']}`"]
(out/'analysis.md').write_text('\n'.join(lines)+'\n')
print(summary['checks'])
for r in results:print(r['condition'],r['metrics']['adv'],r['rag']['private_ordinary'])
print(out/'analysis.md')
