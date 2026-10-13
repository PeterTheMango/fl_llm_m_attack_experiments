# Structured related-work search: grounded federated RAG and three-channel leakage

**Study:** "Does Retrieval Grounding Keep Private Facts Out of the Model? Leakage Through Weights, Client Updates and the Datastore in Federated RAG" ([`proposal.md`](proposal.md)).
**Search date:** 2026-10-12 (America/Toronto). Server timestamps returned by the APIs read 2026-10-13 UTC.
**Second check:** the session that commissioned the search re-queried the arXiv API for 10 of the 2023–2026 preprints closest to our design (FedRepRAG, PromptMIA, SMA, Behavioral Canaries, Marek et al., FedMosaic, Flemings et al., Huang et al., EPD, DCMI); all 10 ids, titles, first authors and author counts matched.
**Done by:** one agent (Claude), in one session. This is the Stage 0 item 1 deliverable ("structured related-work search") named in `proposal.md` §2.

This search is structured: the sources, query strings, criteria and counts are written down so it can be repeated. It is **not** a PRISMA systematic review. One agent screened everything, there was no second screener, there was no full-text screening of every candidate, and two important indexes (DBLP and OpenReview forum pages) blocked automated access. The verdict in §3 should be read with those limits in mind.

---

## 1. Method

### 1.1 Seed documents read first

- `proposal.md` §2 (related work and novelty) and [`novelty_review_20260930.md`](novelty_review_20260930.md), an earlier targeted review that was not systematic (14 papers in its table).
- `written_paper/references.bib` on this branch (15 entries, already verified; the main checkout's copy has 36) and `written_paper/reference_papers/queue_defense_rag_sources.md` (11 verified entries plus later additions such as `bun2016`). Where those files already hold verified metadata, it is reused here, and every cited paper was checked again against its primary record on 2026-10-12.
- `written_paper/reference_papers/references_log.md` (the 2026-10-04 verification log) is kept only in the main checkout, not on this branch, so the search agent did not open it. Its verified metadata for the shared entries (RA-DIT, RAFT, FedRAG, Zeng et al., Mirabel, Riddle Me This, MEntA, ProjRes, FISGuard, FedE4RAG, Anderson et al., LiRA) agrees with the tables below.

### 1.2 Sources and access

| Source | Reachable? | How it was used |
|---|---|---|
| **arXiv API** (`export.arxiv.org/api/query`) | Yes | 28 search queries (§1.3) and 3 title lookups. Metadata (title, authors, first-version date, journal_ref, DOI, comment, abstract) fetched by `id_list` for 72 records. |
| **ACL Anthology** | Record pages: yes. It has no search API. | Searched through web search restricted to `aclanthology.org` (2 queries) and through Crossref. Each cited ACL paper was checked on its Anthology page (citation meta tags: title, authors, venue, pages, DOI). |
| **OpenReview** | **Partly.** The API v2 search endpoint worked. Forum pages (`openreview.net/forum?id=…`) and the API v1 `notes` endpoint returned a Cloudflare Turnstile "challenge required" page. API v1 search returned HTTP 429 (rate limit). | 6 topic queries and 10 exact-title lookups through API v2 search, which returns venue and venueid. The challenge was **not** worked around. Where a forum page was needed, ICLR's own `iclr.cc/virtual/...` pages were used instead. |
| **DBLP** (`dblp.org/search/publ/api`) | **No.** All 8 queries returned a "Making sure you're not a bot!" proof-of-work page instead of JSON (the earlier log also recorded DBLP blocking). | Not used. The challenge was not worked around. |
| **Google Scholar** | Yes, for 6 queries (first results page only, 10 hits each). No CAPTCHA appeared. | Titles parsed from the results page and used as leads only. |
| **Crossref API** | Yes | Bibliographic search and DOI lookups to confirm venue, year and DOI for 30 papers. DOIs were found this way, never typed from memory. |
| **Semantic Scholar API** | **No** (HTTP 429 on the test call) | Not used. |
| **General web search** | Yes | 13 discovery queries (§1.3). Results were used only as leads. Every paper cited below was then checked on its primary record. 12 more web searches were used only to locate a primary record (proceedings URL or forum id). |
| **Publisher / proceedings pages** | Yes | NeurIPS proceedings (`proceedings.neurips.cc`, `papers.nips.cc`), PMLR, USENIX, ICLR virtual pages, and the IEEE S&P 2026 accepted-papers page. |

### 1.3 Exact query strings

**arXiv API** (`search_query=`, sorted by relevance, max 25 per query):

1. `all:"federated" AND all:"retrieval-augmented" AND all:"membership inference"`
2. `all:"federated" AND all:"retrieval augmented generation" AND all:privacy`
3. `all:"retrieval-augmented" AND all:"fine-tuning" AND all:memorization AND all:privacy`
4. `all:"retrieval augmented" AND all:"membership inference" AND all:"fine-tuning"`
5. `all:RAG AND all:"fine-tuning" AND all:privacy AND all:leakage`
6. `all:"retrieval-augmented" AND all:memorization AND all:training`
7. `all:federated AND all:RAG AND all:gradient AND all:leakage`
8. `all:"retrieval augmented" AND all:"differential privacy" AND all:"fine-tuning"`
9. `all:"RAG" AND all:"fine-tuning" AND all:"membership inference"`
10. `all:"retrieval" AND all:"fine-tuning" AND all:"privacy" AND all:"membership inference" AND all:"knowledge"`
11. `ti:"membership inference" AND (ti:"retrieval" OR ti:RAG)`
12. `all:"federated" AND all:"large language model" AND all:"membership inference"`
13. `all:"federated" AND all:"language model" AND all:"gradient inversion"`
14. `all:"malicious server" AND all:"federated learning" AND all:"membership"`
15. `all:"differential privacy" AND (ti:"retrieval-augmented" OR ti:RAG)`
16. `ti:federated AND (ti:RAG OR ti:"retrieval-augmented") AND (abs:"fine-tune" OR abs:"fine-tuning")`
17. `abs:"retrieval" AND abs:"fine-tuning" AND abs:"privacy" AND abs:"memorization" AND abs:"language model"`
18. `(abs:"RAG" OR abs:"retrieval-augmented") AND abs:"fine-tuning" AND abs:"membership inference"`
19. `(ti:RAG OR ti:retrieval) AND (ti:"fine-tuning" OR ti:SFT) AND (ti:privacy OR ti:memorization OR ti:leakage)`
20. `abs:"knowledge injection" AND abs:privacy AND (abs:RAG OR abs:retrieval)`
21. `abs:"federated" AND abs:"retrieval" AND abs:"membership inference"`
22. `abs:"retrieval-augmented" AND abs:"training data" AND abs:"memorization" AND abs:"reduce"`
23. `(abs:"RAG" OR abs:"retrieval") AND abs:"federated" AND abs:"gradient" AND abs:"privacy" AND abs:"attack"`
24. `abs:"context" AND abs:"fine-tuning" AND abs:"membership inference" AND abs:"answer"`
25. `abs:"open-book" AND abs:"fine-tuning" AND (abs:memorization OR abs:privacy)`
26. `abs:"retrieval" AND abs:"membership inference" AND abs:"training data" AND abs:"retrieval database"`
27. `abs:"federated" AND abs:"RAG" AND abs:"differential privacy" AND abs:"fine-tun"`
28. `abs:"retrieval-augmented" AND abs:"federated" AND abs:"memorization"`

Title lookups: `ti:"Who Said That" AND abs:membership`, `ti:LAMP AND ti:"Extracting Text from Gradients"`, `ti:"FedShield-LLM"`, `ti:PrivAuditor` (no hit), `ti:"Benchmarking Empirical Privacy Protection"`.

**DBLP** (all blocked): `federated retrieval augmented generation privacy`, `federated RAG membership inference`, `retrieval augmented membership inference`, `RAG membership inference`, `retrieval augmented fine-tuning privacy`, `RAG fine-tuning privacy`, `federated large language model membership inference`, `federated RAG`.

**OpenReview API v2 search** (`term=`, limit 25): `federated retrieval augmented generation privacy`, `retrieval augmented generation membership inference`, `retrieval augmented fine-tuning memorization privacy`, `federated large language model membership inference`, `RAG fine-tuning privacy leakage`, `federated RAG fine-tuning`. These were followed by 10 exact-title lookups for venue confirmation.

**Google Scholar** (`as_ylo=2016`): `federated retrieval augmented generation privacy membership inference`, `retrieval augmented fine-tuning memorization privacy`, `RAG fine-tuning membership inference training data vs retrieval data`, `privacy leakage federated RAG gradient`, `retrieval-augmented training reduces memorization`, `RAG vs fine-tuning privacy`.

**Web search, for discovery** (13 queries):

1. `federated retrieval-augmented generation fine-tuning membership inference weights gradients datastore privacy leakage`
2. `RAG versus fine-tuning privacy membership inference comparison knowledge injection leakage 2025`
3. `retrieval-augmented fine-tuning reduces memorization of training data privacy language models`
4. `membership inference attack federated instruction tuning large language models malicious server 2025`
5. `does training with context (open-book fine-tuning) reduce memorization of answers membership inference closed-book comparison`
6. `privacy of federated RAG fine-tuning differential privacy client updates retrieval corpus membership audit`
7. `parametric vs non-parametric memory privacy leakage retrieval-based language model fine-tuning membership inference same data`
8. `"Benchmarking Data Protection Vulnerabilities in LLM Adaptation Techniques"` (follow-up of an OpenReview hit)
9. `federated retrieval-augmented generation generator fine-tuning privacy attack evaluation 2026 arXiv`
10. `retrieval-augmented generation privacy membership inference fine-tuning memorization` (restricted to aclanthology.org)
11. `federated learning language model privacy leakage retrieval` (restricted to aclanthology.org)
12. `retrieval augmented fine-tuning privacy membership inference federated` (restricted to openreview.net)
13. `membership inference document in both fine-tuning set and RAG retrieval database disentangle training membership retrieval membership`

### 1.4 Window and criteria

- **Window:** 2016-01-01 to 2026-10-12. The newest item screened is dated 2026-10-08.
- **Include** a record if all of these hold:
  - it is a peer-reviewed paper or an arXiv preprint;
  - it is in English;
  - it is primary research;
  - it does at least one of these:
    - measures privacy leakage (membership, extraction, reconstruction or attribution) for language models, RAG systems or federated learning;
    - proposes or evaluates retrieval-augmented or grounded fine-tuning, or federated RAG training;
    - supplies a DP mechanism or accountant used for LLM or FL fine-tuning or RAG.
- **Exclude** a record for any of these:
  - it is a survey, SoK or mapping study (used only to find more papers);
  - it is an anonymous submission still under review;
  - it covers only images or multimodal data with no text-LM transfer, except RAPID, kept as an adjacent precedent;
  - it is a robustness, poisoning or routing paper with no privacy measurement;
  - it is a pure systems or encryption framework with no leakage measurement, unless it is a federated RAG training framework (kept for the grounded-training theme);
  - its primary record could not be opened. These are listed in §4, not cited.

### 1.5 Counts

| Stage | Count | Note |
|---|---|---|
| Unique titles returned by the API searches (arXiv 125, OpenReview 26, Google Scholar 49, with overlap) | 183 | Exact, from the search log. OpenReview review/comment notes without a title (120) were skipped. |
| Extra leads from web search and the seed documents, after removing duplicates by hand | about 55 | Approximate: web-search results overlapped heavily with the API results. |
| **Title-screened in total** | **about 240** | |
| Primary record opened (arXiv metadata and abstract, or an ACL/proceedings/USENIX page) | 75 | |
| Excluded after opening | 9 | Three surveys or mapping studies, one threat-model position paper, two off-topic background papers, one defense with no measurement, one foundational paper not needed here (Yeom et al. 2018), and one wrong arXiv id (§4). |
| **Included and verified** | **66** | A 11, B 16, C 19, D 7, E 5, F 8 (§2). |

### 1.6 Verification procedure

For every included paper:

- the primary-record id resolves;
- the title matches;
- the first author and the author count match;
- the year, and the venue where one is claimed, were checked against the primary record.

The "Verified" column says exactly what was checked:

- **yes:** id, title, first author, author count and venue all confirmed on a primary or publisher record.
- **partial:** the paper is real and matches, but the **venue** rests only on the arXiv comment or a search-result listing, because the venue page itself could not be opened (OpenReview challenge). For these, cite the arXiv version, or confirm the venue by hand before the camera-ready.

"Relevance" lines are paraphrased from abstracts. For EPD and FedRepRAG, they also draw on the method and privacy sections of the arXiv HTML full text.

---

## 2. Thematic tables

Keys reuse those already in `references.bib` or `queue_defense_rag_sources.md` where an entry exists. "+N" means "and N more authors".

### A. Grounded / retrieval-augmented fine-tuning and federated RAG training (11)

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| lewis2020 | Patrick Lewis +11. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020. | [arXiv 2005.11401](https://arxiv.org/abs/2005.11401); [NeurIPS page](https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html) | The RAG foundation. It trains a generator conditioned on retrieved passages; our inference RAG is a causal-LM adaptation of it. | yes: arXiv id/title/12 authors; NeurIPS proceedings page resolves with the title |
| lin2024radit | Xi Victoria Lin +11. RA-DIT: Retrieval-Augmented Dual Instruction Tuning. ICLR 2024. | [arXiv 2310.01352](https://arxiv.org/abs/2310.01352) | The closest objective-level precedent for RG: the LM is fine-tuned on instructions prefixed with retrieved text, with loss on the output only. Reports no privacy measurement. | yes: arXiv v4 (comment: ICLR 2024 camera-ready), 12 authors; OpenReview API v2 venue "ICLR 2024 poster" (forum 22OTbutug9) |
| zhang2024raft | Tianjun Zhang +6. RAFT: Adapting Language Model to Domain Specific RAG. COLM 2024. | [arXiv 2403.10131](https://arxiv.org/abs/2403.10131) | Fine-tunes on question + relevant and distractor documents + answer. Shows "train to read documents" is established. No privacy measurement. | yes: arXiv, 7 authors; OpenReview API v2 venueid COLM 2024 (forum rzQGHXNReU) |
| chen2022retroprompt | Xiang Chen +8. Decoupling Knowledge from Memorization: Retrieval-augmented Prompt Learning. NeurIPS 2022. | [arXiv 2205.14704](https://arxiv.org/abs/2205.14704); DOI [10.52202/068431-1736](https://doi.org/10.52202/068431-1736) | Argues that retrieval during training lowers reliance on rote memorization. This is the generalization-side version of our H1/H4 question, measured without any privacy attack. | yes: arXiv (comment NeurIPS 2022), 9 authors; Crossref NeurIPS 35 |
| ovadia2024 | Oded Ovadia +3. Fine-Tuning or Retrieval? Comparing Knowledge Injection in LLMs. EMNLP 2024, pp. 237–250. | [ACL 2024.emnlp-main.15](https://aclanthology.org/2024.emnlp-main.15/); [arXiv 2312.05934](https://arxiv.org/abs/2312.05934) | Compares fine-tuning with retrieval for adding knowledge, on utility only. This is background for the RG vs CB-AO vs P0 utility comparison (H5, H7). | yes: ACL page (title, 4 authors, pages, DOI) and arXiv |
| fajardo2025fedrag | Val Andrei Fajardo +7. FedRAG: A Framework for Fine-Tuning Retrieval-Augmented Generation Systems. CODEML Workshop at ICML 2025. | [arXiv 2506.09200](https://arxiv.org/abs/2506.09200) | Supports federated (FedAvg) fine-tuning of RAG generators and retrievers, including retrieval-augmented LM training. It is the federated precedent for RG, a framework with no leakage study. | partial: arXiv id/title/8 authors; the workshop venue comes only from the arXiv comment |
| mao2025fede4rag | Qianren Mao +13. Privacy-Preserving Federated Embedding Learning for Localized Retrieval-Augmented Generation (FedE4RAG). arXiv 2025. | [arXiv 2504.19101](https://arxiv.org/abs/2504.19101) | Federates **retriever** training with distillation and homomorphic encryption. The generator is not trained, and it does not measure leakage through weights, updates or the datastore. | yes (preprint): arXiv id/title/14 authors; no venue found |
| he2025pfedrag | Hangyu He +4. pFedRAG: A Personalized Federated Retrieval-Augmented Generation System with Depth-Adaptive Tiered Embedding Tuning. Findings of EMNLP 2025, pp. 14255–14268. | [ACL 2025.findings-emnlp.769](https://aclanthology.org/2025.findings-emnlp.769/) | Federated personalization of retrieval embeddings, an architecture-level federated RAG precedent. It does not study generator memorization. | yes: ACL page (title, 5 authors, pages, DOI 10.18653/v1/2025.findings-emnlp.769) |
| addison2024cfedrag | Parker Addison +15. C-FedRAG: A Confidential Federated Retrieval-Augmented Generation System. arXiv 2024. | [arXiv 2412.13163](https://arxiv.org/abs/2412.13163) | Federated RAG across data silos, protected with confidential computing. Retrieval only, with no fine-tuning or leakage attacks. | yes (preprint): arXiv id/title/16 authors |
| liang2026fedmosaic | Zhilin Liang +5. FedMosaic: Federated Retrieval-Augmented Generation via Parametric Adapters. arXiv 2026. | [arXiv 2602.05235](https://arxiv.org/abs/2602.05235) | Encodes client documents into adapters instead of sending text. It moves the datastore into parameters, the opposite of RG's aim, and does not measure leakage. | yes (preprint): arXiv id/title/6 authors |
| qian2025hyfedrag | Cheng Qian +4. HyFedRAG: A Federated Retrieval-Augmented Generation Framework for Heterogeneous and Privacy-Sensitive Data. arXiv 2025. | [arXiv 2509.06444](https://arxiv.org/abs/2509.06444) | A federated RAG built on Flower with anonymization. Privacy is judged by an LLM grader, not by membership attacks, and there is no training. | yes (preprint): arXiv id/title/5 authors |

### B. Membership inference on fine-tuned LLMs and on RAG (16)

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| shokri2017 | Reza Shokri +3. Membership Inference Attacks Against Machine Learning Models. IEEE S&P 2017. | [arXiv 1610.05820](https://arxiv.org/abs/1610.05820); DOI [10.1109/SP.2017.41](https://doi.org/10.1109/SP.2017.41) | Defines the membership-inference game behind every endpoint we report. | yes: arXiv + Crossref (S&P 2017, 4 authors) |
| carlini2021extracting | Nicholas Carlini +11. Extracting Training Data from Large Language Models. USENIX Security 2021, pp. 2633–2650. | [arXiv 2012.07805](https://arxiv.org/abs/2012.07805); [USENIX page](https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting) | Source of the **Reference** score: the target's log-perplexity relative to a reference model, here the pretrained model. That is our H1/H6 weights-channel endpoint. | yes: arXiv (12 authors) + USENIX page; already in `references.bib` |
| carlini2022 | Nicholas Carlini +5. Membership Inference Attacks From First Principles (LiRA). IEEE S&P 2022. | [arXiv 2112.03570](https://arxiv.org/abs/2112.03570); DOI [10.1109/SP46214.2022.9833649](https://doi.org/10.1109/SP46214.2022.9833649) | The case for calibrated, low-FPR evaluation and paired in/out models. Our paired W1/W0 worlds are a per-target version of its in/out idea (we do not run LiRA). | yes: arXiv + Crossref (S&P 2022, 6 authors) |
| mattern2023membership | Justus Mattern +5. Membership Inference Attacks against Language Models via Neighbourhood Comparison. Findings of ACL 2023, pp. 11330–11343. | [ACL 2023.findings-acl.719](https://aclanthology.org/2023.findings-acl.719/) | A reference-free LM attack, an alternative to Reference for the weights channel. | yes: ACL page (6 authors, pages, DOI) + arXiv 2305.18462 |
| shi2024detecting | Weijia Shi +7. Detecting Pretraining Data from Large Language Models (Min-K%). ICLR 2024. | [arXiv 2310.16789](https://arxiv.org/abs/2310.16789) | A token-level LM membership score, another weights-channel baseline. | yes: arXiv (8 authors); OpenReview API v2 venue "ICLR 2024 poster" (forum zWqr3MQuNs) |
| fu2024membership | Wenjie Fu +5. Membership Inference Attacks against Fine-tuned Large Language Models via Self-prompt Calibration (SPV-MIA). NeurIPS 2024. | [arXiv 2311.06062](https://arxiv.org/abs/2311.06062); DOI [10.52202/079017-4290](https://doi.org/10.52202/079017-4290) | An attack aimed specifically at fine-tuned LLMs. The arXiv title adds "Practical"; the proceedings title omits it. | yes: arXiv journal_ref NeurIPS 2024 + Crossref; in `references.bib` |
| duan2024 | Michael Duan +9. Do Membership Inference Attacks Work on Large Language Models? COLM 2024. | [arXiv 2402.07841](https://arxiv.org/abs/2402.07841) | Finds pretraining-data MIAs near chance and sensitive to distribution shift. Supports our use of paired worlds and article-disjoint non-members rather than loose member/non-member sets. | yes: arXiv (10 authors, comment COLM 2024); OpenReview API v2 venueid COLM 2024 |
| mireshghallah2022 | Fatemehsadat Mireshghallah +4. An Empirical Analysis of Memorization in Fine-tuned Autoregressive Language Models. EMNLP 2022, pp. 1816–1826. | [ACL 2022.emnlp-main.119](https://aclanthology.org/2022.emnlp-main.119/) | Shows that how you fine-tune (head, adapters, full) changes membership exposure. That supports isolating the training objective (RG vs CB-AO) with everything else fixed. | yes: ACL page (5 authors, pages, DOI). Its arXiv twin 2205.12506 carries an older title. |
| zhu2024privauditor | Derui Zhu +6. PrivAuditor: Benchmarking Data Protection Vulnerabilities in LLM Adaptation Techniques. NeurIPS 2024 Datasets and Benchmarks. | [NeurIPS page](https://proceedings.neurips.cc/paper_files/paper/2024/hash/12b18a15dcd73e1991e9959a94375fab-Abstract-Datasets_and_Benchmarks_Track.html); DOI [10.52202/079017-0308](https://doi.org/10.52202/079017-0308) | A benchmark of privacy attacks across LLM adaptation methods (weights channel only). | partial: proceedings page and Crossref confirm title, first author, 7 authors and venue; the abstract was not read on the primary record, so relevance rests on the title |
| anderson2024 | Maya Anderson +2. Is My Data in Your Retrieval Database? Membership Inference Attacks Against Retrieval Augmented Generation. ICISSP 2025, pp. 474–485. | [arXiv 2405.20446](https://arxiv.org/abs/2405.20446); DOI [10.5220/0013108300003899](https://doi.org/10.5220/0013108300003899) | Our verbatim yes/no datastore baseline. | yes: arXiv journal_ref + Crossref |
| li2025s2mia | Yuying Li +3. Generating Is Believing: Membership Inference Attacks against Retrieval-Augmented Generation. ICASSP 2025. | [arXiv 2406.19234](https://arxiv.org/abs/2406.19234); DOI [10.1109/ICASSP49660.2025.10889013](https://doi.org/10.1109/ICASSP49660.2025.10889013) | A datastore MIA based on similarity between the document and the generated text. An alternative to our natural-question attacks. | yes: arXiv + Crossref |
| liu2025mba | Mingrui Liu +2. Mask-based Membership Inference Attacks for Retrieval-Augmented Generation. WWW 2025. | [arXiv 2410.20142](https://arxiv.org/abs/2410.20142); DOI [10.1145/3696410.3714771](https://doi.org/10.1145/3696410.3714771) | A datastore MIA that masks spans and asks the RAG system to fill them in. Verbatim-style, so it is easy for gates like Mirabel to detect. | yes: arXiv + Crossref |
| naseh2025riddle | Ali Naseh +5. Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation. ACM CCS 2025. | [arXiv 2502.00306](https://arxiv.org/abs/2502.00306); DOI [10.1145/3719027.3744840](https://doi.org/10.1145/3719027.3744840) | Natural, document-specific questions. Our F1-scored natural-question attack (H3) is adapted from it. | yes: arXiv + Crossref |
| choi2025 | Yujin Choi +4. Safeguarding Privacy of Retrieval Data against Membership Inference Attacks: Is This Query Too Close to Home? (Mirabel). Findings of EMNLP 2025, pp. 8241–8258. | [ACL 2025.findings-emnlp.438](https://aclanthology.org/2025.findings-emnlp.438/); [arXiv 2505.22061](https://arxiv.org/abs/2505.22061) | Detects queries that are too close to a document and hides it. In this repo it is a retrieval-removal control, not a defense. | yes: ACL page + arXiv |
| nguyen2026menta | Nguyen Linh Bao Nguyen +6. Five Queries Are Enough: Query-Efficient and Surrogate-Free Membership Inference Attacks on RAG via Entailment (MEntA). USENIX Security 2026. | [arXiv 2605.24312](https://arxiv.org/abs/2605.24312); [USENIX page](https://www.usenix.org/conference/usenixsecurity26/presentation/nguyen-nguyen) | Entailment-scored natural questions. Our NLI-scored datastore attack is adapted from it. | yes: arXiv (7 authors) + USENIX page |
| gao2025dcmi | Xinyu Gao +4. DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation. arXiv 2025. | [arXiv 2509.06026](https://arxiv.org/abs/2509.06026) | Calibrates datastore MIA against how hard each query is, using a reference RAG. Similar in spirit to our library-only vs neither contrast. | yes (preprint): arXiv id/title/5 authors |

### C. Gradient and update leakage in FL, including FL fine-tuning of LLMs (19)

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| melis2019 | Luca Melis +3. Exploiting Unintended Feature Leakage in Collaborative Learning. IEEE S&P 2019. | [arXiv 1805.04049](https://arxiv.org/abs/1805.04049); DOI [10.1109/SP.2019.00029](https://doi.org/10.1109/SP.2019.00029) | Early evidence that collaborative-learning updates leak membership and properties. | yes: arXiv + Crossref |
| nasr2019 | Milad Nasr +2. Comprehensive Privacy Analysis of Deep Learning: Passive and Active White-box Inference Attacks against Centralized and Federated Learning. IEEE S&P 2019. | [arXiv 1812.00910](https://arxiv.org/abs/1812.00910); DOI [10.1109/SP.2019.00065](https://doi.org/10.1109/SP.2019.00065) | The original active, white-box FL membership attack. Our malicious-server causal attack (H2) belongs to this family. | yes: arXiv + DOI (Crossref) |
| zhu2019dlg | Ligeng Zhu +2. Deep Leakage from Gradients. NeurIPS 2019. | [arXiv 1906.08935](https://arxiv.org/abs/1906.08935); [NeurIPS page](https://papers.nips.cc/paper/9617-deep-leakage-from-gradients) | Shows that training inputs can be rebuilt from shared gradients, which motivates the client-update channel. | yes: arXiv + proceedings page (title, 3 authors) |
| geiping2020 | Jonas Geiping +3. Inverting Gradients – How easy is it to break privacy in federated learning? NeurIPS 2020. | [arXiv 2003.14053](https://arxiv.org/abs/2003.14053); [NeurIPS page](https://proceedings.neurips.cc/paper/2020/hash/c4ede56bbd98819ae6112b20ac6bf145-Abstract.html) | Gradient inversion that also works on trained networks. Background for the update channel. | yes: arXiv + proceedings page |
| fowl2022robbing | Liam Fowl +4. Robbing the Fed: Directly Obtaining Private Data in Federated Learning with Modified Models. ICLR 2022. | [arXiv 2110.13057](https://arxiv.org/abs/2110.13057) | A malicious server changes the model to pull out client data. Supports our malicious-server threat model. | partial: arXiv id/title/5 authors; ICLR 2022 rests on the OpenReview forum listing (fwzUgo0FM9v) seen in search results, because the forum page is behind a bot challenge |
| boenisch2023 | Franziska Boenisch +5. When the Curious Abandon Honesty: Federated Learning Is Not Private. IEEE EuroS&P 2023. | [arXiv 2112.02918](https://arxiv.org/abs/2112.02918); DOI [10.1109/EuroSP57164.2023.00020](https://doi.org/10.1109/EuroSP57164.2023.00020) | Active-server data recovery through crafted weights. Same adversary class as H2. | yes: arXiv + Crossref |
| fowl2023decepticons | Liam Fowl +6. Decepticons: Corrupted Transformers Breach Privacy in Federated Learning for Language Models. ICLR 2023. | [arXiv 2201.12675](https://arxiv.org/abs/2201.12675); [ICLR page](https://iclr.cc/virtual/2023/poster/10839) | A malicious-server attack on federated **transformer LMs**: the language-model case of our update channel. | yes: arXiv (comment ICLR 2023, 7 authors) + iclr.cc page |
| gupta2022film | Samyak Gupta +5. Recovering Private Text in Federated Learning of Language Models (FILM). NeurIPS 2022. | [arXiv 2205.08514](https://arxiv.org/abs/2205.08514); DOI [10.52202/068431-0590](https://doi.org/10.52202/068431-0590) | Text recovery from federated LM training through memorization in the shared model. It spans our weights and updates channels. | yes: arXiv + Crossref |
| balunovic2022lamp | Mislav Balunović +3. LAMP: Extracting Text from Gradients with Language Model Priors. NeurIPS 2022. | [arXiv 2202.08827](https://arxiv.org/abs/2202.08827); DOI [10.52202/068431-0555](https://doi.org/10.52202/068431-0555) | Text gradient inversion. Background for the update channel. | yes: arXiv + Crossref |
| petrov2024dager | Ivo Petrov +4. DAGER: Exact Gradient Inversion for Large Language Models. NeurIPS 2024. | [arXiv 2405.15586](https://arxiv.org/abs/2405.15586); DOI [10.52202/079017-2787](https://doi.org/10.52202/079017-2787) | Exact text recovery from LLM gradients, a stronger update-channel threat than membership. | yes: arXiv + Crossref |
| nguyen2023active | Truc Nguyen +4. Active Membership Inference Attack under Local Differential Privacy in Federated Learning. AISTATS 2023, PMLR 206:5714–5730. | [arXiv 2302.12685](https://arxiv.org/abs/2302.12685) | A dishonest-server membership attack that still works under local DP, the AMIA lineage of this repo's update attacks. | yes: arXiv journal_ref (PMLR 206); in `references.bib` |
| vu2024 | Minh N. Vu +3. Analysis of Privacy Leakage in Federated Large Language Models. AISTATS 2024, PMLR 238. | [arXiv 2403.04784](https://arxiv.org/abs/2403.04784); [PMLR page](https://proceedings.mlr.press/v238/vu24a.html) | Active membership attacks with stated success guarantees on federated LMs. It is the closest prior work for H2's adversary, but has no retrieval or grounding. | yes: arXiv + PMLR page |
| ngo2024secagg | Khac-Hoang Ngo +3. Secure Aggregation Is Not Private Against Membership Inference Attacks. ECML PKDD 2024. | [arXiv 2403.17775](https://arxiv.org/abs/2403.17775); DOI [10.1007/978-3-031-70365-2_11](https://doi.org/10.1007/978-3-031-70365-2_11) | Aggregation alone does not stop membership inference. This justifies measuring per-client releases and adding release noise (our C2 pass). | yes: arXiv + Crossref |
| rashid2026 | Md Rafi Ur Rashid +4. Gradient-Free Privacy Leakage in Federated Language Models through Selective Weight Tampering. PoPETs 2026. | [arXiv 2310.16152](https://arxiv.org/abs/2310.16152); DOI [10.56553/popets-2026-0083](https://doi.org/10.56553/popets-2026-0083) | Intermediate FL model snapshots and tampered weights leak other clients' data. It covers the weights channel in FL, without retrieval. | yes: arXiv + Crossref |
| guo2025 | Wenkai Guo +5. Can Federated Learning Safeguard Private Data in LLM Training? Vulnerabilities, Attacks, and Defense Evaluation. Findings of EMNLP 2025, pp. 23986–24013. | [ACL 2025.findings-emnlp.1303](https://aclanthology.org/2025.findings-emnlp.1303/) | Extracts training data from federated LLM global models and evaluates DP and other defenses. Weights channel only, no retrieval. | yes: ACL page + arXiv 2509.20680 |
| hu2025 | Yingqi Hu +6. Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models. Findings of IJCNLP-AACL 2025, pp. 1808–1827. | [ACL 2025.findings-ijcnlp.113](https://aclanthology.org/2025.findings-ijcnlp.113/) | Cross-client PII extraction from a federated LLM: another weights-channel FL result. | yes: ACL page + arXiv 2506.06060 |
| nguyen2026promptmia | Quan Minh Nguyen +5. Leveraging Soft Prompts for Privacy Attacks in Federated Prompt Tuning (PromptMIA). arXiv 2026. | [arXiv 2601.06641](https://arxiv.org/abs/2601.06641) | A malicious server plants crafted prompts and reads membership from client updates. This is close to our crafted-request causal attack, but for prompt tuning and without retrieval. | yes (preprint): arXiv id/title/6 authors |
| deng2026projres | Guilin Deng +8. Toward Efficient Membership Inference Attacks against Federated Large Language Models: A Projection Residual Approach (ProjRes). IEEE S&P 2026. | [arXiv 2604.21197](https://arxiv.org/abs/2604.21197); [S&P 2026 accepted papers](https://sp2026.ieee-security.org/accepted-papers.html) | A passive update-channel MIA for federated LLMs that remains effective under DP noise. It is a stronger passive alternative to our active attack. | yes: arXiv (9 authors) + the title appears on the S&P 2026 accepted list |
| jiang2026fisguard | Haocheng Jiang +1. FISGuard: Defending Against Membership Inference via Fixed Input Subspaces. arXiv 2026. | [arXiv 2608.27836](https://arxiv.org/abs/2608.27836) | A defense against ProjRes for federated PEFT. A possible comparator if a follow-up defense study targets the update channel. | yes (preprint): arXiv id/title/2 authors |

### D. DP fine-tuning of LMs and accounting (7)

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| abadi2016 | Martín Abadi +6. Deep Learning with Differential Privacy. ACM CCS 2016, pp. 308–318. | [arXiv 1607.00133](https://arxiv.org/abs/1607.00133); DOI [10.1145/2976749.2978318](https://doi.org/10.1145/2976749.2978318) | DP-SGD with per-example clipping and Gaussian noise, the mechanism behind our DP arms (ε = 16 in Stage 2; ε = 64 only in the pilot). | yes: arXiv journal_ref + Crossref |
| mcmahan2018 | H. Brendan McMahan +3. Learning Differentially Private Recurrent Language Models. ICLR 2018. | [arXiv 1710.06963](https://arxiv.org/abs/1710.06963) | User-level DP-FedAvg for language models (clipped and noised client updates). | partial: arXiv id/title/4 authors; ICLR 2018 rests on the arXiv comment ("camera-ready ICLR 2018"), because the OpenReview forum (BJ0hF1Z0b) is behind a bot challenge |
| bun2016 | Mark Bun, Thomas Steinke. Concentrated Differential Privacy: Simplifications, Extensions, and Lower Bounds. Theory of Cryptography (LNCS), 2016, pp. 635–658. | [arXiv 1605.02065](https://arxiv.org/abs/1605.02065); DOI [10.1007/978-3-662-53641-4_24](https://doi.org/10.1007/978-3-662-53641-4_24) | zCDP, the conservative accountant (`defenses.privacy_bound`) used to match ε across arms. | yes: arXiv + Crossref (container "Theory of Cryptography") |
| mironov2019 | Ilya Mironov +2. Rényi Differential Privacy of the Sampled Gaussian Mechanism. arXiv 2019. | [arXiv 1908.10530](https://arxiv.org/abs/1908.10530) | Explains why we claim no sampling amplification for fixed-size shuffled batches. | yes (preprint): arXiv id/title/3 authors |
| li2022 | Xuechen Li +3. Large Language Models Can Be Strong Differentially Private Learners. ICLR 2022. | [arXiv 2110.05679](https://arxiv.org/abs/2110.05679); [ICLR page](https://iclr.cc/virtual/2022/poster/6894) | Full-model DP fine-tuning of pretrained LMs. Our full-model private optimizer follows this line. | yes: arXiv + iclr.cc page title |
| yu2022 | Da Yu +11. Differentially Private Fine-tuning of Language Models. ICLR 2022. | [arXiv 2110.06500](https://arxiv.org/abs/2110.06500); [ICLR page](https://iclr.cc/virtual/2022/oral/7161) | DP with parameter-efficient methods (LoRA and others). The alternative design we did not choose (we use full-model DP). | yes: arXiv (12 authors, comment ICLR 2022) + iclr.cc page title |
| marek2026 | Bartłomiej Marek +6. Benchmarking Empirical Privacy Protection for Adaptations of Large Language Models. ICLR 2026. | [arXiv 2606.09401](https://arxiv.org/abs/2606.09401); [ICLR page](https://iclr.cc/virtual/2026/poster/10007869) | Finds that adaptation data close to the pretraining distribution leaks more empirically at the same ε. This matters for SQuAD, which is likely in pretraining, and for interpreting H6. | yes: arXiv (7 authors, comment ICLR 2026 Oral) + iclr.cc page title |

### E. Formal or heuristic protection of RAG datastores (5; brief)

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| koga2025 | Tatsuki Koga +3. Privacy-Preserving Retrieval-Augmented Generation with Differential Privacy. arXiv 2024 (v3 2025). | [arXiv 2412.04697](https://arxiv.org/abs/2412.04697) | A DP mechanism over generated tokens for the datastore. It protects the third channel, which our study only measures. | yes (preprint): arXiv id/title/4 authors |
| grislain2024 | Nicolas Grislain. RAG with Differential Privacy. arXiv 2024. | [arXiv 2412.19291](https://arxiv.org/abs/2412.19291) | DP token generation for RAG. Same role as `koga2025`. | yes (preprint): arXiv, single author |
| mori2026dpsynrag | Junki Mori +3. Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG) (DP-SynRAG). Findings of ACL 2026, pp. 1216–1235. | [ACL 2026.findings-acl.62](https://aclanthology.org/2026.findings-acl.62/); [arXiv 2510.06719](https://arxiv.org/abs/2510.06719) | A DP synthetic datastore built once, so later queries add no privacy cost. The natural formal comparator for H3's datastore leak. | yes: ACL page + arXiv |
| zeng2025synthetic | Shenglai Zeng +9. Mitigating the Privacy Issues in Retrieval-Augmented Generation (RAG) via Pure Synthetic Data. EMNLP 2025, pp. 24527–24558. | [ACL 2025.emnlp-main.1247](https://aclanthology.org/2025.emnlp-main.1247/) | Replaces the datastore with synthetic text, a heuristic alternative to DP-SynRAG. | yes: ACL page + arXiv 2406.14773 |
| jiang2025rapid | Tanqiu Jiang +3. RAPID: Retrieval Augmented Training of Differentially Private Diffusion Models. ICLR 2025. | [arXiv 2502.12794](https://arxiv.org/abs/2502.12794) | Public retrieval during DP training improves utility, an adjacent precedent for "grounding helps under DP" (H6). It concerns images, not LMs or FL. | yes: arXiv (comment ICLR 2025); OpenReview API v2 venue "ICLR 2025 Poster" (forum txZVQRc2ab) |

### F. Work that combines retrieval or grounded training with privacy measurement on more than one channel (8)

These queries were searched hardest (§1.3: arXiv 1–10, 16–28; Google Scholar 1–6; web 1–3, 5–7, 9, 12–13). The papers below are the closest found. None measures all three of our channels.

| Key | Citation | Primary record | Relevance to our design | Verified |
|---|---|---|---|---|
| huang2023 | Yangsibo Huang +4. Privacy Implications of Retrieval-Based Language Models. EMNLP 2023, pp. 14887–14902. | [ACL 2023.emnlp-main.921](https://aclanthology.org/2023.emnlp-main.921/); [arXiv 2305.14888](https://arxiv.org/abs/2305.14888) | Compares leakage from a kNN-LM's private datastore with leakage from parametric models trained on the same private data, and finds that the datastore leaks more. It is the earliest "weights vs datastore" comparison, with no FL, no update channel and no grounded-vs-closed-book fine-tuning. | yes: ACL page (5 authors, pages, DOI) + arXiv |
| zeng2024goodbad | Shenglai Zeng +10. The Good and The Bad: Exploring Privacy Issues in Retrieval-Augmented Generation (RAG). Findings of ACL 2024, pp. 4505–4524. | [ACL 2024.findings-acl.267](https://aclanthology.org/2024.findings-acl.267/) | Retrieval leaks the datastore but **reduces** leakage of training data, measured on both channels in one system. The effect is from retrieval at **inference**, not from a training objective, and there is no FL. | yes: ACL page (11 authors, pages, DOI) + arXiv 2402.16893 |
| fu2025epd | Haowei Fu +5. Ensemble Privacy Defense for Knowledge-Intensive LLMs against Membership Inference Attacks. arXiv 2025. | [arXiv 2512.03100](https://arxiv.org/abs/2512.03100) | Runs seven MIAs against a fine-tuned (LoRA) LLM and a RAG LLM, including on SQuAD, then proposes an inference-time ensemble defense. The SFT and RAG systems are separate; SFT appears to be closed-book; there is no FL, no paired worlds, and DP appears only as a logit-noise baseline. | yes (preprint): arXiv id/title/6 authors; HTML full text read for setup |
| flemings2025 | James Flemings +4. Estimating Privacy Leakage of Augmented Contextual Knowledge in Language Models. ACL 2025 (long), pp. 25092–25108. | [ACL 2025.acl-long.1220](https://aclanthology.org/2025.acl-long.1220/); [arXiv 2410.03026](https://arxiv.org/abs/2410.03026) | A DP-style "context influence" measure that separates leakage of retrieved context from what the model already knows. It addresses the same attribution problem as our four-cell design, at inference only, with no training-membership or FL. | yes: ACL page (5 authors, pages, DOI) + arXiv |
| salemi2025 | Alireza Salemi, Hamed Zamani. Comparing Retrieval-Augmentation and Parameter-Efficient Fine-Tuning for Privacy-Preserving Personalization of Large Language Models. ACM ICTIR 2025. | [arXiv 2409.09510](https://arxiv.org/abs/2409.09510); DOI [10.1145/3731120.3744595](https://doi.org/10.1145/3731120.3744595) | Compares RAG, PEFT and both, in a setting labelled "privacy-preserving", but reports **utility only** (no leakage attack). It shares our arm structure, not our measurement. | yes: arXiv + Crossref (ICTIR 2025, 2 authors) |
| peng2026fedreprag | Can Peng +6. Beyond Raw Context Transfer: Representation-based Federated Retrieval-Augmented Generation (FedRepRAG). arXiv 2026. | [arXiv 2609.22162](https://arxiv.org/abs/2609.22162) | Federated RAG with a privacy evaluation: inversion of exchanged representations, membership of documents used to train a shared projector, and DP-SGD on one stage. The generator stays frozen, the datastore MIA is not evaluated, the adversary is honest-but-curious, and there is no grounded-vs-closed-book comparison. | yes (preprint): arXiv id/title/7 authors; HTML privacy section read (appendix only partly read) |
| sun2025sma | Shixuan Sun +4. SMA: Who Said That? Auditing Membership Leakage in Semi-Black-box RAG Controlling. arXiv 2025. | [arXiv 2508.09105](https://arxiv.org/abs/2508.09105) | Attributes generated content to pretraining, retrieval or user input: a membership audit across sources. Not FL, and no training-objective comparison. | yes (preprint): arXiv id/title/5 authors |
| chen2026canaries | Chaoran Chen +2. Behavioral Canaries: Auditing Private Retrieved Context Usage in RL Fine-Tuning. arXiv 2026. | [arXiv 2604.22191](https://arxiv.org/abs/2604.22191) | Audits whether retrieved documents were used in (RL) fine-tuning, where fact-memorization MIAs fail. It addresses "context passages entering training", the H4 concern, but not for supervised grounded training or FL. | yes (preprint): arXiv id/title/3 authors |

---

## 3. Verdict

**Does any prior work duplicate our design? No.** This search found no work that does all of the following in one system:

- trains the same model grounded and closed-book;
- uses paired in/out training worlds per target, crossed with in/out of the retrieval library (the four cells);
- measures leakage through the weights, the client updates (an active malicious server) and the datastore;
- compares utility and leakage at matched, accounted DP budgets.

**Partial overlaps exist on every individual piece**, so the contribution is the controlled combination and its results, not any single component. Consistent with `proposal.md`, nothing here supports a "first" claim. The defensible wording is "this search found no work that …".

Closest works, one line each (what they share / what they lack):

- **Zeng et al. 2024 (The Good and The Bad):** shares the measurement of training-data leakage and datastore leakage in one RAG system; lacks a training-objective comparison, FL and the update channel. Its effect comes from retrieval at inference.
- **Huang et al. 2023:** shares the comparison of datastore and parametric leakage for the same private data; lacks FL, updates and grounded fine-tuning (it studies kNN-LMs).
- **Fu et al. 2025 (EPD):** shares MIAs on SFT vs RAG LLMs, including on SQuAD; lacks grounded training, a shared-record four-cell design, FL and accounted DP.
- **Peng et al. 2026 (FedRepRAG):** shares federated RAG with membership, inversion and DP-SGD measurements; lacks generator fine-tuning, a datastore MIA, an active server and the grounded-vs-closed-book contrast.
- **Flemings et al. 2025:** shares the separation of context leakage from parametric knowledge; lacks training membership, FL and updates.
- **Sun et al. 2025 (SMA):** shares attribution of membership signal across training vs retrieval sources; lacks FL, a training-objective comparison and DP.
- **Salemi & Zamani 2025:** shares the RAG vs fine-tuning vs both arm structure; lacks any leakage measurement.
- **RA-DIT / RAFT / FedRAG:** share grounded (answer-from-passage) fine-tuning, which FedRAG federates; lack any privacy measurement.
- **Vu et al. 2024, Nguyen et al. 2023, PromptMIA 2026, ProjRes 2026:** share update-channel membership attacks on federated (L)LMs, mostly active-server; lack retrieval, a datastore and a grounding comparison.
- **Guo et al. 2025, Hu et al. 2025, Rashid et al. 2026:** share weights-channel leakage in federated LLM fine-tuning, with DP defenses in Guo et al.; lack retrieval and the update/datastore channels.
- **Marek et al. 2026:** shares empirical membership under DP adaptation at matched ε; lacks RAG, FL and a grounding comparison.

**Confidence.** Moderate. Indexing delays matter: several of the closest works are from 2026, and two appeared in the last six weeks. DBLP and OpenReview forum pages were blocked, Google Scholar was read only to its first results page, and other wording ("open-book", "context-augmented", "knowledge-grounded" training) may hide relevant work. A manual check of DBLP and OpenReview in a browser before submission is advisable.

---

## 4. Not verified, or excluded

**Not verified** (seen only as a search-result title or snippet; primary record not opened; **not cited** above):

- RAG-leaks: difficulty-calibrated membership inference attacks on retrieval-augmented generation (Sci. China Inf. Sci. 2025, per an OpenReview listing)
- RAGLeak: Membership Inference Attacks on RAG-Based Large Language Models (Springer chapter)
- Few-MIA: Few-Shot Membership Inference Attacks on Retrieval-Augmented Generation
- Seeing Is Believing: Black-Box Membership Inference Attacks Against Retrieval Augmented Generation (possibly an earlier title of `li2025s2mia`)
- E-MIA; CIG-MIA; BudgetLeak; ImageAuditor; MrM; "Do Multimodal RAG Systems Leak Data?"; "Membership Inference Attacks for Retrieval Based In-Context Learning for Document Question Answering" (further RAG MIAs, titles only)
- Adaptive privacy-aware retrieval for federated clinical language models
- DP-RAG: A Differentially Private Retrieval-Augmented Generation Framework for Privacy-Preserving Knowledge Assistance; DP2-RAG; Private-RAG; Differentially Private Retrieval-Augmented Generation (arXiv 2602.14374); Differentially Private Datastore Generation for Retrieval-Augmented Inference; Privacy-Aware Decoding (Wang et al.)
- Learning to Erase Private Knowledge from Multi-Documents for Retrieval-Augmented Large Language Models
- RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation
- Exploring Membership Inference Vulnerabilities in Clinical Large Language Models
- Mitigating Membership Inference Attacks in Large Language Models (DuoLearn, Findings of ACL 2025)
- Gradient Inversion Attacks on Parameter-Efficient Fine-Tuning; ReCIT; Reconstructing Training Data from Adapter-based Federated Large Language Models; Uncovering Gradient Inversion Risks in Practical Language Model Training
- FD-RAG: Federated Dual-System Retrieval-Augmented Generation; Supercharging Federated Intelligence Retrieval
- Federated Training, Fine-Tuning, RAG, and Inference using Flower (OpenReview poster/talk)
- A claim (from a secondary summary) that a May 2025 paper by Jiang et al. found weak membership advantage on federated LLMs; no primary record was found.

**Excluded by criteria:**

- Anonymous submissions under review: Federated Graph Retrieval-Augmented Generation (ICLR 2027 submission); OCDR (ICLR 2027 submission); SkillGuard-RAG (anonymous preprint).
- Opened, then excluded:
  - surveys and mapping studies, used only to find more papers: arXiv 2505.18906 (Federated RAG mapping study), 2606.25533 and 2503.19338. The SoK on gradient leakage (2404.05403) was excluded at title level;
  - a threat-model position paper with no measurement: arXiv 2509.20324;
  - off-topic background: arXiv 2412.13720 (FL + RAG medical framework, no leakage measurement) and 2211.08411 (long-tail knowledge);
  - a defense with no leakage measurement: arXiv 2506.05640 (FedShield-LLM);
  - Yeom et al. 2018 (arXiv 1709.01604): the loss-attack paper is already verified in `references.bib`. It is not needed beyond Shokri et al. 2017 for this design;
  - a wrong id: 2202.12241 is a physics paper, my mistaken guess for LAMP, whose correct id 2202.08827 was then found by title.
- Robustness or routing papers on federated RAG with no privacy measurement, screened at title: arXiv 2609.33037, 2605.28112, 2608.07913.

---

## 5. Notes for the write-up

- Cite FedRAG as an ICML 2025 **workshop** paper (CODEML). Cite Robbing the Fed and McMahan et al. 2018 with their ICLR venue only after a manual check of OpenReview, or cite the arXiv versions.
- SPV-MIA's proceedings title omits "Practical". Mireshghallah et al.'s EMNLP title differs from its arXiv title. Use the proceedings titles.
- Two different first authors named Fu appear: Wenjie Fu (SPV-MIA, NeurIPS 2024) and Haowei Fu (EPD, arXiv 2025).
- Zeng et al. 2024 and Huang et al. 2023 are the two papers a reviewer is most likely to raise against a "channel shift" claim. Our design differs from both because it changes the **training objective** while holding the inference RAG fixed, adds the client-update channel, and places the same target record in training and the library in a controlled way.
- Marek et al. 2026 is relevant to how DP results on SQuAD (a public, likely pretrained-on corpus) should be read (H6).
