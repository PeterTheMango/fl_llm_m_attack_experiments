from pathlib import Path
import pandas as pd
import json
OUT=Path(__file__).parent
s=pd.read_csv(OUT/'attack_summary.csv');r=pd.read_csv(OUT/'run_metrics.csv')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+['| '+' | '.join(str(v) for v in row)+' |' for row in rows])
legacy=s[s.version=='legacy'];corrected=s[s.version=='theory_v2']
lt=table(['Attack','Completed runs','Mean balanced accuracy','Mean AUROC','BA > 50% runs','Mean TPR','Mean FPR'],[[x.attack_name,x.runs,f'{x.mean_balanced_accuracy:.1%}',f'{x.mean_auc:.3f}',f'{x.above_chance_runs}/{x.runs}',f'{x.mean_tpr:.1%}',f'{x.mean_fpr:.1%}'] for x in legacy.itertuples()])
ct=table(['Attack','Balanced accuracy','AUROC','TP / FN / FP / TN','Interpretation'],[[x.attack_name,f'{x.balanced_accuracy:.1%}',f'{x.auc:.3f}',f'{x.tp} / {x.fn} / {x.fp} / {x.tn}',{'amia':'Perfect on three tested batch pairs','reference':'All nonmember predictions','wbc':'All member predictions','recall':'One of three members detected','zlib':'Two members detected; one false positive','min_k':'All nonmember predictions','min_k_plus_plus':'All member predictions','samia':'All nonmember predictions'}[x.attack_name]] for x in r[r.version=='theory_v2'].sort_values(['balanced_accuracy','auc'],ascending=False).itertuples()])
big=table(['Attack','Trials','Balanced accuracy','AUROC','TPR','FPR','Run ID'],[[x.attack_name,x.n,f'{x.balanced_accuracy:.1%}',f'{x.auc:.4f}',f'{x.tpr:.1%}',f'{x.fpr:.1%}',x.run_id] for x in r[r.n==100].itertuples()])
report=f'''# Membership-inference results in federated learning and RAG

Analysis date: 12 September 2026. Evidence: the two supplied Firestore CSV exports, checked against the local implementation and its historical theory audit. Original attachments were preserved. No training jobs or source-code changes were made.

## Study decision

**AMIA is the leading candidate for a defense study against an active, malicious FL server. No attack in these exports is yet proven reliable against a functioning combined FL–RAG system.** The corrected AMIA run identifies all six batch-membership outcomes, but it covers only three paired batches, one configuration, and one seed, without LDP. Historical AMIA runs are consistently perfect but have documented implementation limitations.

For passive access to the final model’s scores, **Reference and WBC are the strongest candidates to validate next**. They show strong score separation, including AUROC 1.0 in the corrected runs. Their current corrected thresholds fail to exploit that separation. An external RAG user has a different access model: the relevant recorded attack is the Anderson datastore-membership attack, whose completed evaluations are only nonfunctional tiny-model smoke tests.

The evidence supports choosing a focused validation target. It does not support describing any method as universally reliable, combining historical and corrected results into one benchmark, or claiming that RAG defenses have already worked.

## 1. Cleaning and provenance

| Processing step | Records |
|---|---:|
| Raw CSV records, excluding headers | 342 |
| Failed experiments excluded | 144 |
| `monitor_state` metadata excluded | 1 |
| Completed records before deduplication | 197 |
| Duplicate completed record removed | 1 |
| Unique completed experiments retained | **196** |

The larger CSV’s 341 records contain 200 rows with 14 fields, 50 with 17 fields, and 91 with 20 fields, despite its 14-column header. A standard header-based import would put some statuses and nested RAG results into the wrong fields. Parsing therefore identifies nested objects by required keys, locates an exact status token, validates the configuration and trial structure, and retains source filename and original one-based CSV record number. The smaller CSV contains one record.

The duplicated LOSS run is `loss_federated_llm_adaptation_v1_6b867aebd9da4cc2`. Its trials, metrics, methodology, and timestamp agree across exports. Its configuration differs only in the Firestore collection name. It contributes once, retaining the first attachment’s version.

Failure reasons were 99 unsupported BFloat16 errors, 41 input/output errors, and four simulation errors. These are operational failures, not evidence that an attack could not infer membership. They contribute neither successes nor failures to effectiveness denominators. Completed attacks with chance-level or worse predictions remain in the analysis.

The retained evidence comprises **184 historical FL runs, eight corrected `theory_v2` FL runs, and four `pipeline_v1` RAG smoke runs**. There are 1,536 training/batch membership trial rows. RAG contributes a separate 128 membership observations across 32 condition evaluations; these are repeated observations of four candidate indices from one study and must not be described as 128 independent documents.

Every retained run’s class counts, balanced accuracy, TPR and available AUROC fields were recomputed from its recorded trials. No discrepancies were found in those comparisons. All configured trial counts match recorded trial counts. Input hashes and checks are in [validation.json](validation.json). Filtered data are in [completed_runs.csv](completed_runs.csv), exclusions in [excluded_records.csv](excluded_records.csv), and comparable per-run values in [run_metrics.csv](run_metrics.csv).

## 2. What counts as attack success

These experiments test **membership privacy**, not poisoning, backdoors, or answer-error attacks.

| Measurement | Meaning | Chance / desired direction |
|---|---|---|
| Balanced accuracy (BA) | `(TPR + TNR) / 2` at the recorded threshold | 0.5 is chance; larger means better attack |
| Membership advantage | `TPR − FPR = 2 × BA − 1` | 0 is chance; larger means better attack |
| AUROC | Fraction of member–nonmember score comparisons ordered correctly, counting ties as one-half | 0.5 is chance; measures ranking, not a validated threshold |
| TPR and FPR | Members detected and nonmembers falsely accused | Interpret together |

LOSS scores are inverted for AUROC because lower loss indicates membership. All other recorded attack scores use the higher-is-member convention. Scores are never reversed based on observed test performance. Every retained run is class balanced, so its accuracy equals BA. The stored `adv` fields in these completed records follow the BA convention; this should not be mistaken for TPR−FPR.

Historical averages below weight each run equally and average **within-run** AUROCs. They describe this uneven sweep. They are not a pooled ROC, an attack success probability, or a claim of equal exposure across attacks. “BA > 50%” is descriptive and does not mean statistically significant success. Results from six-trial and 100-trial runs are also shown separately.

Reliability would require useful held-out TPR at a prespecified low FPR, reproducibility across independent targets and seeds, a verified attacker-access boundary, and a working system with useful answers. These exports do not establish all four.

## 3. Corrected implementations: the most relevant current evidence

All eight corrected completed runs use DistilGPT2, SQuAD, seed 7, two participating clients, one FL round, one local epoch, a 64-token limit, and six attack trials. None includes RAG. Corrected LOSS, Neighborhood and SPV-MIA failed and are unassessed here.

{ct}

**AMIA:** the member scores are approximately 2.299294, while every nonmember score is zero, against a threshold of `1e-8`. The result is consistent with a target-sensitive neuron exposing membership through the returned client loss gradient. The recorded membership target is explicitly `private_client_batch`, and `ldp_mechanism` is `none`. The presence of `epsilon: 10` in the configuration does not mean LDP was applied. This is a targeted batch test, not evidence that an ordinary RAG user can recover training membership or that the attack defeats DP.

The AMIA certificate field explicitly records `certified_attack: false` with scope `not_applicable_without_input_randomization`. The result is empirical, not a certified attack guarantee.

**Reference and WBC:** both have real separation within this sample, but unsuccessful decisions at their stored thresholds. Reference’s member scores range from −0.9102 to −0.8644; nonmember scores range from −0.9843 to −0.9516. Its −0.75 threshold sits above every score. WBC’s member scores range from 0.9945 to 1.0000 and nonmember scores from 0.8454 to 0.9697; its 0.75 threshold sits below every score. These facts directly explain AUROC 1.0 alongside BA 0.5. A threshold could separate this particular sample, but selecting it from these test labels would be optimistic. Calibrate independently, freeze the rule, and evaluate on new targets.

The seven corrected scalar attacks each record only **three distinct candidate hashes**, with the same candidate evaluated in positive and negative training worlds. Their `target_exposed` flags agree with membership labels. This is good paired-experiment bookkeeping, but it also means six trial rows are not six independent target documents. AMIA records three batch-pair seeds and tests a targeted probe; broad target coverage is not established.

## 4. Historical implementations: useful patterns, limited validity

{lt}

AMIA is perfect in all 14 historical runs, covering seven dataset labels and five seed settings across the sweep. However, the repository’s [historical audit](../../plans/attack-theory-audit/AUDIT.md) documents that its older implementation computed a probe gradient locally instead of observing the actual client loss update, and used private negatives. These runs are therefore evidence about that earlier procedure, not 14 valid replications of the corrected client-gradient attack. Repeated or nearly repeated positive scores and zero negatives further caution against treating all 84 trial rows as independent target evidence.

Reference has the strongest historical passive results: mean BA 78.1%, mean AUROC 0.986, and mean FPR 4.2%. WBC has mean BA 73.3% and AUROC 0.952, but mean FPR 53.3%. A high detection rate for WBC therefore comes with many false accusations at its recorded threshold. Historical Reference uses a loss difference rather than the corrected ratio, and historical WBC pools window votes rather than averaging per-window-size fractions. Their numerical rankings should not be transferred directly to the corrected implementations. See the [correction record](../../master_script/docs/theory_corrections.md).

SPV-MIA’s mean AUROC is 0.926, but BA is only 57.8% and mean FPR is 82.2%. Its older reference construction and perturbation statistic have documented deviations. It is a lower-priority candidate until the corrected implementation completes and is evaluated independently.

Min-K predicts every record nonmember in all 19 historical runs. SaMIA does the same in all 16 runs. Min-K++ assigns a constant label in 17 of 19 runs, usually predicting member. Neighborhood has BA exactly 50% in all 19 runs. Thus a high TPR by itself, particularly when FPR is equally high, is not evidence of useful membership inference.

### More trials weaken several small-sample signals

Only four historical runs have 100 trials; each has 50 member and 50 nonmember outcomes.

{big}

These are not additional validation of AMIA, Reference or WBC: none of those leaders has a 100-trial run in the supplied data. The larger evaluations place these four tested methods much closer to chance, despite favorable AUROCs in several six-trial settings. This is evidence against treating small-run rankings as proof of population reliability.

### Federation size and training intensity

The following are descriptive comparisons within the historical baseline slice: DistilGPT2, SQuAD, seed 7, six trials, 64 tokens, one local epoch and one FL round, except for the named factor. They are not replicated causal estimates.

| Change | Observed pattern | Interpretation |
|---|---|---|
| Two to eight clients, all participating | Reference AUROC 1.000 → 0.889 and BA 0.833 → 0.500; WBC AUROC 1.000 → 0.611; SPV AUROC 1.000 → 0.778; AMIA remains 1.000 | Consistent with dilution of a single client’s contribution for some final-model attacks; AMIA targets a client batch directly |
| One to four local epochs | Reference BA 0.833 → 1.000; Min-K++ AUROC 0.667 → 1.000 while BA stays 0.500 | More training can strengthen ranking without fixing a poorly calibrated threshold |
| One to four FL rounds | Min-K AUROC 0.667 → 0.778 while BA stays 0.500; Zlib BA 0.500 → 0.667 | Some stronger exposure signals, but no universal monotonic success claim |
| 64 to 256 token budget | Most available AUROCs unchanged; SaMIA varies and remains at BA 0.500 | The sweep does not establish a general benefit from longer input limits |

Changing client count also changes aggregate training data and partitions, so these exports cannot isolate averaging dilution from those effects. All successful eight-client runs select all eight clients. No successful partial-participation condition is available to establish reliability under intermittent target-client exposure. Increasing a token budget also does not prove that a particular record actually became longer.

## 5. What the RAG experiments actually establish

All four completed pipeline runs use `sshleifer/tiny-gpt2`, synthetic client text, seed 7, two FL trials, and the same small RAG study. The enclosing FL attacks are Zlib twice, Min-K once, and Min-K++ once. **The nested RAG attack is always `anderson2024_black_box`; those enclosing names must not be used to label its datastore attack.**

Each FL trial evaluates public/private retrieval with and without Mirabel, yielding 32 condition records and 128 membership observations. Every condition has:

- BA = 0.5 and AUROC = 0.5;
- TPR = 0 and FPR = 0;
- zero recognized Yes/No membership responses;
- zero answer exact match and zero answer token F1.

All 128 membership scores are zero. The implementation maps a generated response without a recognized Yes/No token to a nonmember prediction. Consequently, chance-level BA follows mechanically from the fallback and balanced labels. It is not evidence that the retriever concealed membership. The zero utility scores independently show that useful answering has not been demonstrated. Retrieval diagnostics in these exports do not establish whether a relevant document was successfully retrieved for each target.

Mirabel and ordinary retrieval have identical membership and task scores at this floor. The defense has no demonstrated benefit under these conditions. DP-SGD, DP-FedAvg, and no-defense settings also differ across enclosing attacks, and the sole same-attack Zlib comparison has only two FL trials. These data cannot establish a meaningful privacy–utility tradeoff or an adaptive defense result.

The 99 Qwen2.5-0.5B-Instruct pipeline failures are excluded from performance analysis. Their absence nevertheless matters: the intended stronger-model FL–RAG setting has no completed evidence in this export. Training membership, private-batch membership, and retrieval-document membership remain three distinct outcomes.

## 6. Why the patterns are plausible, and what is actually proven

**Active FL leakage has a mechanism supported by prior work.** Nguyen et al. construct malicious model parameters so a chosen neuron makes a client update reveal target membership. The corrected AMIA score pattern is consistent with that mechanism. This correspondence supports an explanation, not a transfer of the original paper’s success guarantees to this single LLM run. [Nguyen et al., AISTATS 2023](https://proceedings.mlr.press/v206/nguyen23e.html).

**Training-dependent likelihood can reveal membership.** Yeom et al. formally and empirically connect generalization behavior to membership risk, while showing that overfitting is not necessary. Increased-training patterns here are compatible with memorization, but the exports lack the matched clean-utility and generalization-gap analysis needed to prove that explanation. [Yeom et al., Privacy Risk in Machine Learning](https://arxiv.org/abs/1709.01604).

**Localized comparison can retain useful signals.** WBC’s source method compares target and reference token losses over multiple window sizes and combines sign-based local evidence. That makes its observed score separation plausible. Historical aggregation differs from the source method, and the corrected result is only three target pairs. [Chen et al., Window-based Membership Inference](https://arxiv.org/html/2601.02751v2).

**Ranking quality and operational usefulness are different.** AUROC can be high while a fixed threshold predicts one class everywhere, exactly as the corrected Reference and WBC scores demonstrate. Carlini et al. argue for evaluation at low false-positive rates because average accuracy can obscure whether any members are identified confidently. [Carlini et al., Membership Inference Attacks From First Principles](https://arxiv.org/abs/2112.03570).

**RAG membership requires retrieval and an informative response.** Anderson et al. explicitly target datastore membership through generated answers. Their source evaluation used 2,000 members and 2,000 nonmembers and also assigned unrecognized answers to nonmember. The failure of response recognition in this export explains its all-negative outcome; it does not contradict the source paper’s successful results on other models. [Anderson et al., Sections 3–3.1](https://arxiv.org/html/2405.20446v2).

The threshold mismatch and RAG fallback explanations are directly demonstrated by the recorded scores and implementation rules. Explanations based on overfitting, dilution, or broad target selectivity remain hypotheses consistent with the observations. Evidence should not be selected to force an “expected behavior” conclusion.

## 7. Statistical limits

Of 196 retained runs, 188 have six trials, four have 100 trials, and four have two trials. A six-trial run has three nonmembers, so its empirical FPR step is **33.3 percentage points**. A 100-trial run has 50 nonmembers, so its step is **2 percentage points**. Neither supports a resolved 1% FPR operating point. Legacy SaMIA’s stored `tpr_at_10fpr` values from three nonmembers do not establish reliable performance at 10% population FPR.

For illustration only, with independent nonmembers, observing zero false positives in three tests gives a one-sided 95% binomial upper bound of `1 − 0.05^(1/3) = 63.2%`. The actual corrected runs share targets and paired worlds, so this is not a confidence interval for the reported study. It shows why “zero observed false positives” is not proof of low population FPR.

No formal pooled significance test or population confidence interval is claimed. Historical records lack sufficient stable target provenance for reliable cross-run clustering, configurations are uneven, and corrected paired samples share targets. A six-trial perfect AUROC is only nine member–nonmember score comparisons, which are themselves dependent. Means and counts are useful for prioritization, not confirmatory inference.

## 8. Recommended focus and confirmation study

| Study access model | Candidate to prioritize | Evidence required before defense claims |
|---|---|---|
| Active malicious FL server, client loss gradients visible | **Corrected AMIA** | Independent target probes and batches, honest public negatives, several seeds/client schedules, no-defense baseline, actual client-update provenance |
| Passive observer with final-model likelihood or logits | **Corrected Reference and WBC** | Independent threshold calibration, target-matched evaluation, low-FPR results, working reference model, stable model and dataset revisions |
| External user of a combined FL–RAG application | **Anderson datastore MIA as the initial baseline** | Functioning instruction-capable RAG model, informative responses, verified target retrieval, independent datastore members/nonmembers |

For the user’s stated FL–RAG objective, the strongest design is to measure **training leakage and retrieval leakage separately inside one functioning system**. AMIA can be the primary FL threat if the malicious-server assumption is in scope. If the intended adversary is only an external RAG user, AMIA should not be presented as that adversary’s attack.

1. **Establish a functioning baseline.** Resolve the stronger-model pipeline execution issue and verify useful held-out answers before comparing privacy outcomes. Log answer recognition, raw outputs, retrieval hits, truncation, and membership labels separately. An unrecognized attack response is an abstention/failure-to-follow-instructions diagnostic even when the source scoring convention labels it nonmember.
2. **Freeze corrected methods and access assumptions.** Keep implementation fingerprint, model/dataset revisions, scoring orientation, threat model and calibration protocol fixed. Do not pool historical variants with corrected results. Retain checkpoints and candidate hashes for the confirmatory subset.
3. **Use independent targets and calibration.** Calibrate thresholds on separate nonmembers, freeze them, and test on independent targets across at least five seeds and multiple useful model/data settings. A practical starting design is 1,000 held-out members and 1,000 held-out nonmembers per primary setting, with paired worlds and clustered uncertainty by target and training run. That supplies 0.1% empirical FPR steps, but is a planning choice, not a power guarantee. Use a power analysis for the desired effect and tail precision; lower FPR goals require more data.
4. **Separate the two membership axes.** Include targets in neither FL training nor the datastore, FL training only, datastore only, and both. Hold candidate text and the rest of the setup fixed as far as possible. Record actual FL exposure, not just assignment to a client. Add full and partial client participation and unequal client data sizes with replication.
5. **Evaluate the defense against the strongest applicable attack at matched utility.** Compare no defense, training defense only, retrieval defense only, and both. Report AUROC, TPR at prespecified FPR, calibrated BA/advantage, recognition/abstention, QA exact match/F1 and latency. Recalibrate an informed attacker on separate defended calibration data so a mere score-scale shift is not mistaken for protection. Account for repeated document queries when evaluating retrieval defenses.

For an AMIA-centered defense, investigate mechanisms that constrain malicious client-facing models and protect the information in client updates, under an explicit server-trust model. For a datastore-centered defense, investigate access control and controlled retrieval/response exposure while retaining legitimate answer utility. These are proposed research directions; neither is validated by these exports. A training-DP result does not automatically cover an independently populated private retrieval store.

### Suggested statement for the study

> After excluding failed executions and duplicate records, we retained 196 completed experiments. Corrected AMIA achieved perfect membership decisions in a six-trial targeted FL batch experiment, while corrected Reference and WBC achieved perfect score ranking but chance-level decisions at their configured thresholds. Historical results motivate these candidates but include implementation differences. Completed RAG experiments were restricted to tiny-model smoke tests with no recognized membership responses and zero answer utility. The results support prioritizing corrected AMIA for an active-server FL privacy study and Reference/WBC for passive model auditing, but do not yet establish a reliable attack or effective defense for a functioning combined FL–RAG system.

## Deliverables

- [Completed runs](completed_runs.csv): normalized nested results and source locations, failed runs and duplicate excluded.
- [Comparable run metrics](run_metrics.csv) and [attack summaries](attack_summary.csv): recalculated values, separated by implementation cohort.
- [Training/batch trials](training_trials.csv): 1,536 recorded observations with run provenance.
- [RAG condition metrics](rag_conditions.csv) and [RAG trials](rag_trials.csv): distinct datastore evaluation, 32 conditions and 128 observations.
- [Exclusion log](excluded_records.csv): every failed, duplicate and metadata record, with reasons.
- [Validation manifest](validation.json): input hashes, reconciled counts and metric comparison results.
'''
OUT.joinpath('report.md').write_text(report)
print('Report written:',len(report.split()),'words')
