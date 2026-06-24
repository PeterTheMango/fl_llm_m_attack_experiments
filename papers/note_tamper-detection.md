# Note: Detectability of the malicious weights (Fig. 8)

**From the paper (Sec. 6, Fig. 8) — [Experiment].** Kernel-density estimates (KDE) of the chosen neuron's *malicious* weights, measured under LDP at `ε = 2, 3, 5`, are **statistically indistinguishable** from the distribution of *normal* (honest-server) weights. Implication: a client that simply inspects the **distribution of model weights** cannot tell the dispatched model was tampered with. The paper uses this to argue that detecting the attack by weight inspection is not a viable defense.

---

## Would tamper detection be a good defense direction?

Short answer: **detecting tampering is a sensible goal, but the paper's own evidence says naive weight-distribution inspection won't work** — so it's promising only if you detect the attack a different way than the one Fig. 8 rules out.

What Fig. 8 actually rules out: looking at the *marginal distribution* of weight values and asking "do these look abnormal?" The malicious weights are crafted by training a neuron with ordinary gradient descent, so they land in the same numeric range as normal weights — nothing looks out of place.

Why detection could still be worth pursuing, just along other axes:

- **Functional / behavioral signatures, not value distributions.** The attack's whole point is a neuron that activates for essentially *one* input and stays silent for everything else. That extreme selectivity is unusual and might be detectable by probing activations on held-out data (e.g., a neuron whose output is non-zero for a vanishingly small fraction of inputs), even though the raw weights look normal.
- **Architecture / protocol integrity instead of statistics.** Rather than judging whether weights "look" malicious, verify the model the server sent matches an agreed specification — e.g., model attestation, hashing/signing the dispatched architecture and parameters, or reproducibility checks. This sidesteps the indistinguishability result entirely because you're checking provenance, not appearance.
- **It complements, not replaces, noise-based defenses.** The paper shows LDP fails at any usable budget and notes DPSGD can be circumvented by averaging noisy gradients across rounds. Detection is attractive precisely because the "add more noise" lever is blocked by the utility trade-off.

Caveats / open questions:

- The paper does **not** propose or test a tamper-detection defense; this is a direction, not a validated result. Treat behavioral-detection ideas above as **interpretation**, not something the paper demonstrates.
- Behavioral detection is an arms race: an adversary aware of the detector could regularize the chosen neuron to fire on a small *region* rather than a single point, softening the signature.
- Detection assumes the client can run such checks. In many FL deployments clients are resource-constrained and rely on the server for the model and the privacy mechanism — the same trust assumption the attack exploits.

Bottom line: tamper **detection by weight-distribution inspection is closed off by Fig. 8**, but **behavioral activation analysis and model attestation/integrity verification are reasonable, unexplored directions** worth considering — ideally alongside, not instead of, privacy-noise defenses.
