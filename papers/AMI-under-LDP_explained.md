# Active Membership Inference Attack under Local Differential Privacy in Federated Learning — A Layered Explanation

## Paper identity (confirmed against the file)

- **Title:** *Active Membership Inference Attack under Local Differential Privacy in Federated Learning*
- **Authors:** Truc Nguyen (U. Florida), Phung Lai (NJIT), Khang Tran (NJIT), NhatHai Phan (NJIT), My T. Thai (U. Florida, corresponding author)
- **Venue:** Proceedings of the 26th International Conference on Artificial Intelligence and Statistics (**AISTATS**) 2023, Valencia, Spain, PMLR Volume 206. Also on arXiv as `arXiv:2302.12685v2`, 24 Jul 2023.

> **Flag (discrepancy):** the file is named `IEEE-2023-...`, but the paper's own front matter states it is an **AISTATS / PMLR 2023** paper, not an IEEE publication. The filename appears to be mislabeled; the venue above is what the document itself reports.

A note on labels below: I tag claims as **[Claim/Proof]** (what the paper asserts or proves), **[Experiment]** (empirical findings), or **[Interpretation]** (my own reading). Notation follows the paper.

---

## 1. TL;DR

The paper introduces an **active membership inference (AMI) attack** run by a *dishonest federated-learning server*. Before sending the global model to a client, the server secretly crafts the weights of one "chosen" neuron so that the neuron fires **only** for a specific target data point. By inspecting the gradient the client returns for that neuron, the server learns whether the target was in the client's private training set. The central contribution is showing — both with a **certified theoretical guarantee** and experiments — that **Local Differential Privacy (LDP) does not stop this attack** at any privacy budget mild enough to keep the model useful. The attack exploits *correlations among input features* through a *non-linear decision boundary*, which small LDP noise cannot erase, and is achieved with a minimal one-iteration modification to the model.

---

## 2. Background primer

**Federated Learning (FL).** A way to train one shared model across many parties (clients) without the clients' raw data ever leaving their devices. The paper studies *horizontal* FL: every client holds the **same features** but **different samples** (Sec. 2.1). Formally, the model `f_θ: ℝ^d → ℝ^k` is a `k`-class neural network with weights `θ`. A central server coordinates training: each round `i`, it sends the current model `f_θⁱ` to a selected subset of `M` clients; each client `u_j` computes a gradient `Gⱼⁱ = ∇_θ L(Dⱼ, θⁱ)` on its local batch `Dⱼ` and uploads it; the server averages them and updates the model, `G^i = (1/M) Σ Gⱼⁱ`, `θ^{i+1} = θ^i − η G^i` (**Eq. 1**). The promise of FL is privacy by keeping data local — but the paper stresses that "FL in its primitive form offers little to no privacy protection" (Sec. 1).

**Membership inference (MI).** An attack that decides whether *a particular data sample was part of a model's training set*. Knowing that "Alice's record was used to train this medical model" can itself be a serious privacy breach.

**Passive vs. active attacker.**
- *Passive* (a.k.a. *honest-but-curious* / *semi-honest*): the server follows the protocol exactly and only **observes** updates, trying to infer information from them.
- *Active* (a.k.a. *malicious* / *actively dishonest*): the server **deviates** from the protocol — it can modify the model architecture and/or parameters before dispatching them (Sec. 2.2). This paper studies the active case, arguing it is the realistic one and that the honest-but-curious assumption "undermines the vulnerability of the FL system as in practice" (Sec. 2.2).

**Local Differential Privacy (LDP).** A privacy-preserving mechanism applied *locally*, by the client, before data is used/shared, so the client never has to trust the server. Built on *randomized response* (Warner, 1965). Formally (**Definition 1, ε-LDP**): a randomized algorithm `M` satisfies ε-LDP if for any two inputs `x, x'` and any output `O`, `Pr[M(x)=O] ≤ e^ε · Pr[M(x')=O]`. The **privacy budget ε** controls how much the output distributions for two different inputs may differ: **smaller ε = more noise = stronger privacy but worse model utility**; larger ε = less noise = weaker privacy but better utility. This ε / utility tension is the hinge of the whole paper.

---

## 3. The threat model (Sec. 2.2, Fig. 1)

- **Who:** the *central FL server* `A`, which is also the adversary. It is **actively dishonest** — free to maliciously craft the model parameters (and the one extra neuron) it dispatches.
- **What it targets:** a **specific client** at an **arbitrary training iteration**, and a **specific target sample** `t ∈ ℝ^d`. The client's local batch is `D = {(xᵢ, yᵢ)}`.
- **What it knows:** the data distribution `𝔻` from which `D` is drawn (i.e., the adversary has access to *similar* data — "query access to 𝔻"). The paper argues this is practical because a server can collect large amounts of representative data (Sec. 2.2, citing Shokri et al., 2017). Under LDP it is *additionally* assumed the adversary knows the **LDP mechanism `M` and the budget `ε`** (Sec. 4) — realistic because clients typically rely on the server to supply the mechanism and budget.
- **What it controls:** the malicious parameters `θ` it sends. The client computes and returns `G = ∇_θ L(D, θ)`.
- **Goal:** decide whether the target `t` is in `D`. Formally a function `A^𝔻: t, G → {0,1}` where 1 means `t ∈ D` (**Eq. 2**).
- **Formalization as a security game** (**Fig. 1**, `Exp(A, L, 𝔻)`): a fair coin `b` decides whether the target is drawn from inside `D` (`b=1`) or outside (`b=0`); the adversary returns a guess `b'` and wins if `b' = b`. Success rate / advantage is `Adv^A = ½ Pr[b'=1|b=1] + ½ Pr[b'=0|b=0]` (**Eq. 3**), i.e., the average of the True Positive Rate (TPR) and True Negative Rate (TNR). Random guessing scores 0.5, so a meaningful attack must beat 0.5.

---

## 4. How the attack works (Sec. 3)

### Step 1 — The intuition: a neuron's gradient reveals activation (Sec. 3.1)
For a fully-connected layer with ReLU, neuron `i` computes `ReLU(Wᵢx + bᵢ) = max(0, Wᵢx + bᵢ)`. Key observation: if `Wᵢx + bᵢ ≤ 0`, the neuron is **not activated** by `x`, and its gradient `Gᵢ^(x)` at that point is **zero**; otherwise the gradient is **non-zero**. Because the client returns a gradient *averaged over the whole batch* `D`, a neuron's reported gradient is the average of per-sample gradients. **[Interpretation]** This turns the gradient into a detector: it is non-zero exactly when at least one batch sample activated the neuron.

### Step 2 — The dream condition: a neuron that only the target activates
If the server could build a neuron activated by the target `t` and by **no other** `x ≠ t`, then the neuron's gradient is non-zero **iff `t ∈ D`** — a perfect membership test. For a single first-layer neuron this means `Σ Wᵢⱼ tⱼ > 0` and `Σ Wᵢⱼ xⱼ ≤ 0` for all `x ≠ t` (**Eq. 4**, bias suppressed).

### Step 3 — Why a single linear neuron is not enough (the core insight)
**[Claim/Proof]** A single neuron's pre-activation `Σ Wᵢⱼ xⱼ` is a **linear** function of the input, and a linear function cannot carve out *one* point from all others. Solving Eq. 4 for arbitrary data is infeasible (the paper proves this in **Appendix A**). The fix — and the paper's key technical idea — is to **introduce non-linearity** by using a neuron in the **second** fully-connected layer that sits on top of `r` first-layer ReLU neurons. The attack now seeks weights `(h, W)` such that

`Σᵢ hᵢ · ReLU(Σⱼ Wᵢⱼ tⱼ) > 0` and `Σᵢ hᵢ · ReLU(Σⱼ Wᵢⱼ xⱼ) ≤ 0` for all `x ≠ t` (**Eq. 5**).

This composite (ReLU layer + linear combination) is a **non-linear decision boundary** that *can* enclose the target while excluding everything else. **[Interpretation]** This is the conceptual heart of the paper: membership of a single point is separable only with non-linearity, and that non-linearity is exactly what later survives LDP noise.

### Step 4 — Building the malicious neuron by training it (Sec. 3.2, Fig. 2)
Rather than solving Eq. 5 analytically, the server *trains* the chosen neuron. It puts a logistic sigmoid on the output, `s(x) = σ(h · ReLU(Wx))` (**Eq. 6**), samples an auxiliary dataset `X ∼ 𝔻^m`, labels the target `t` as **1** and all other samples as **0**, and trains with cross-entropy. Driving `s(t) > 0.5` forces `h·ReLU(Wt) > 0`, while `s(x) < 0.5` forces `h·ReLU(Wx) < 0` — exactly the conditions of Eq. 5.

### Step 5 — Inference
The server embeds these crafted weights into the global model and sends it to the target client. After receiving the averaged gradient `G`, it extracts the gradient `g_t` of the chosen neuron: **`g_t = 0` ⇒ predict `t ∉ D`; `g_t ≠ 0` ⇒ predict `t ∈ D`** (Fig. 2).

**Cost / footprint.** The attack modifies only **1 chosen neuron in the second layer plus `r` associated neurons in the first layer**, and runs **within a single FL training iteration** (Sec. 3.2) — a minimal, hard-to-notice change.

---

## 5. Why LDP doesn't stop it (Sec. 4)

**The setup.** Under LDP, each client perturbs every training sample with mechanism `M` to get a randomized set `D' = {M(x, ε)}` and returns `G = ∇_θ L(D', θ)` (Fig. 3). So the server never sees the true `t`, only noisy versions.

**Why the naive attack breaks.** A neuron trained to fire on the *exact* target `t` will likely **not** fire on its randomized version `M(t, ε)` — so `h·ReLU(W·M(t,ε)) < 0` and the attack misses (Sec. 4). LDP looks like a defense for this reason.

**The fix — attack the noisy target, not the clean one.** The server generates a set `T` of `l` independent perturbations `M(t, ε)` (by invoking the *known* mechanism `l` times), labels all of them **1**, labels a disjoint sample set `X` as **0**, and trains the chosen neuron to fire on the *distribution of noisy targets* rather than on a single point (**Eq. 7**, Fig. 4). Because the adversary knows `M` and `ε`, it can simulate the noise the client will add.

**Certified guarantee of success (Sec. 4, Theorem 1).** **[Claim/Proof]** Using the *expected-output-stability* property of DP (Lecuyer et al., 2019), the attack is certifiably robust if `E[v(t)] > 0` and `E[v(x)] ≤ 0` for `x ≠ M(t,ε)`, where `v(t) = h·ReLU(W·M(t,ε))` (**Eq. 8**). Since these expectations can't be computed exactly, the paper estimates them by **Monte Carlo sampling** (invoking `M` many times) and bounds them with **Hoeffding's inequality** to get `(1−δ)`-confidence lower/upper bounds `Ê^lb[v(t)]` and `Ê^ub[v(x)]` (**Eqs. 9–10**). **Theorem 1** then states the attack succeeds whenever `Ê^lb[v(t)] > 0` and `Ê^ub[v(x)] ≤ 0` (**Eq. 11**); **Corollary 1** gives the minimal budget `ε*` and broken probability `δ*` for which the guarantee holds (**Eq. 12**, proof in Appendix B).

**The fundamental reason LDP fails (the trade-off).** **[Claim + Interpretation]** The attack works by exploiting *correlations among input features*, captured through the non-linear decision boundary. To defeat it, LDP noise must be large enough to **break that correlation** — but that requires a **small ε**, which simultaneously **destroys the model's utility** (Sec. 4, Sec. 5). With a **large ε** (low noise, usable model), the correlation survives and so does the attack. There is no budget that both protects membership and preserves a useful model; the paper frames its results as "stress-testing the fundamental trade-offs between model utility and privacy risk" (Sec. 1).

---

## 6. Key results (Sec. 5)

**Setup.** Datasets: **CIFAR-10**, **ImageNet**, **CelebA**. The attack operates on 512-dimensional embeddings from a pre-trained ResNet-18. LDP mechanisms tested: **BitRand** and **OME** (Lai et al., 2021); main text reports BitRand, OME deferred to Appendix D. Architecture for the attack: `r = 1000` neurons in the first layer, 1 chosen neuron in the second. Batch sizes `|D|`: 20 (CelebA), 64 (ImageNet), 100 (CIFAR-10). The security game (Eq. 3) is run **10,000 times** to estimate success rate, TPR, TNR. Code: `github.com/trucndt/ami`.

**Without LDP.** **[Experiment]** ~**100% success** on all three datasets. The attack still reaches 100% with **as few as `r = 5`** first-layer neurons (Sec. 5, details in Appendix D).

**Under LDP (Fig. 5).** **[Experiment]**
- Large budgets `ε ≥ 5` give only minimal defense: model accuracy stays acceptable, yet attack success is **≥ 0.77**, approaching a near-perfect **0.99 at ε ≥ 9**.
- At `ε ∈ [3,4]`, success is still **0.67 / 0.58 / 0.62** on CelebA / ImageNet / CIFAR-10.
- Only at very low `ε ∈ [1,2]` is the attack curbed — but there the **model accuracy is severely damaged**.
- TPR and TNR both exceed 0.5 (so the attack detects both presence and absence of `t`), except TNR on ImageNet at `ε ≤ 4`.

**Visualization (Fig. 6, t-SNE on CelebA).** **[Experiment]** At `ε = 10` and `ε = 7`, the target and its noisy variants form a separate cluster that's easy to isolate. At `ε = 5` the noisy variants blend into other samples, making the boundary harder — yet the attack still attains **0.80** success.

**Certified guarantee (Fig. 7).** **[Experiment/Proof]** For `ε ∈ [1,10]` with broken probability `10⁻⁸`, the gap between the lower bound of the chosen neuron's value on noisy *target* samples and the upper bound on noisy *non-target* samples **widens as ε grows**, matching the rising empirical success rate.

**Detectability (Fig. 8).** **[Experiment]** Kernel-density estimates of the malicious weights (at `ε = 2, 3, 5`) are **indistinguishable** from normal weights — so a client inspecting the model weights cannot tell the model was tampered with.

---

## 7. Implications & defenses (Secs. 6–7)

**Practical takeaway.** **[Claim]** "Current implementations of FL provide virtually no privacy protection for clients" — even with LDP, because any LDP budget weak enough to keep the model useful leaves the membership signal intact, and the tampering is undetectable from the weights.

**Defenses the paper discusses (without endorsing a working one):**
- **Noisy gradients via DPSGD** (Abadi et al., 2016): clients add DP noise to *gradients* before sending. The paper notes two problems: (1) recent work (Boenisch et al.; Tramèr & Boneh) indicates DPSGD makes it hard to train good models on CIFAR-10/ImageNet; (2) an attacker can **aggregate noisy gradients over multiple FL iterations** to cancel the added noise (detailed in Appendix E).
- **Detecting malicious weights:** shown to be **challenging** — Fig. 8 demonstrates malicious and honest weights are statistically indistinguishable.

The paper explicitly defers an effective defense to **future work** (Sec. 7), so it diagnoses the vulnerability rather than solving it.

---

## 8. Open questions / limitations

These are gaps I did not find resolved in the main text; some may be addressed in the appendices, which I did not fully read (flagged where relevant).

- **No working defense.** The paper proposes none; DPSGD and weight-inspection are discussed and then shown to be inadequate or unproven.
- **Strong adversary assumptions.** The server is assumed to know the data distribution `𝔻` and, under LDP, the exact mechanism `M` and budget `ε`. How the attack degrades if these are imperfectly known is not quantified in the main text.
- **Architecture-modification capability.** The attack assumes the server can inject a crafted neuron / modify parameters before dispatch. Settings with attestation, secure aggregation, or architecture verification are not analyzed.
- **Embedding-space evaluation.** Experiments run on ResNet-18 embeddings (512-dim) rather than end-to-end raw-image FL; generalization to other modalities/architectures is asserted but not broadly tested in the main text.
- **One target at a time.** The game/metrics evaluate a single target sample; scaling to many simultaneous targets or to the full membership profile of a client isn't detailed here.
- **Secure aggregation / many clients.** The averaging `G = (1/M) Σ Gⱼ` is over one targeted client's contribution; interaction with secure-aggregation protocols that hide individual updates is not addressed in the main text.
- **Appendices not fully verified.** Appendix A (linearity infeasibility), B (Theorem 1 proof), C (BitRand/OME background), D (OME results, `r`-ablation), and E (DPSGD circumvention) were referenced but not read in full for this summary.
