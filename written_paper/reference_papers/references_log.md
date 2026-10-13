# references.bib verification log (2026-10-04)

Applies to `written_paper/references.bib` (36 entries, keys and order unchanged). The verified entries were installed on 2026-10-04. The first 15 (author-supplied) entries are unchanged; the updates proposed for them in section B await the researcher's decision.

Sources checked: the arXiv API (all 25 arXiv ids involved), ACL Anthology `.bib` exports, the OpenReview API, USENIX Security '26 presentation pages (BibTeX), Crossref DOI records, PMLR, and NeurIPS proceedings. DBLP blocked automated access (Anubis bot check), so I did not use it. Springer chapter pages were also blocked. I took the LNCS details from Crossref, and the TCC volume number (9985) from secondary sources (the Harvard Privacy Tools page and a search summary).

## A. Entries added 2026-10-03 (now verified and completed)

| key | verified? | what changed | primary record URL |
|---|---|---|---|
| lin2024radit | yes | Full author list (12) from the ICLR 2024 OpenReview record. That record uses "Richard James" and "Wen-tau Yih"; arXiv uses "Rich James" and "Scott Yih". Added `file`. arXiv v4 is the ICLR 2024 camera-ready version. | https://openreview.net/forum?id=22OTbutug9 ; https://arxiv.org/abs/2310.01352 |
| zhang2024raft | yes | Changed @misc to @inproceedings. The paper was published at COLM 2024; booktitle is "First Conference on Language Modeling". Full author list (7). Added `file`. | https://openreview.net/forum?id=rzQGHXNReU ; https://arxiv.org/abs/2403.10131 |
| fajardo2025fedrag | partial | Full author list (8) taken from arXiv v2. The venue (ICML 2025 CODEML workshop, non-archival) is confirmed by the arXiv comment and OpenReview. Problem: the workshop OpenReview record lists the authors in a different order and with different names ("D. B. Emerson", "Chi Ho Cheung", "Ravi Theja Desetty"). I kept the arXiv list; check with the authors' preferred form. Booktitle expanded. Added `file`. | https://arxiv.org/abs/2506.09200 ; https://openreview.net/forum?id=I2is1QsDuD |
| zeng2024good | yes | Full author list (11) in the ACL Anthology order. Crossref gives a slightly different order; the Anthology and arXiv orders agree. Added pages 4505--4524, doi, and eprint 2402.16893. Removed `url` because the doi replaces it. Added `file`. | https://aclanthology.org/2024.findings-acl.267/ |
| fu2025ensemble | yes (venue changed) | Changed @misc to @inproceedings. **Now published in Findings of EACL 2026**, pp. 2786--2799, doi 10.18653/v1/2026.findings-eacl.145. The year is now 2026; the key still says 2025 and was kept. Full author list (6). Added `file`. | https://aclanthology.org/2026.findings-eacl.145/ ; https://arxiv.org/abs/2512.03100 |
| choi2025mirabel | yes | Full author list (5). Added pages 8241--8258 and doi 10.18653/v1/2025.findings-emnlp.438. Added `file`. | https://aclanthology.org/2025.findings-emnlp.438/ |
| naseh2025riddle | yes | Full author list (6). Booktitle is now "Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security". Added pages 1245--1259 and doi 10.1145/3719027.3744840. Added `file`. | https://doi.org/10.1145/3719027.3744840 ; https://arxiv.org/abs/2502.00306 |
| nguyen2026menta | yes | The arXiv id and title match (v4, "Accepted by USENIX Security 2026"). Full author list (7): the first author is "Nguyen Linh Bao Nguyen", surname Nguyen. Booktitle is "35th USENIX Security Symposium" (same style as carlini2021extracting). Added pages 1147--1166. Added `file`. | https://www.usenix.org/conference/usenixsecurity26/presentation/nguyen-nguyen ; https://arxiv.org/abs/2605.24312 |
| deng2026projres | yes (venue changed) | Changed @misc to @inproceedings. **Published at IEEE S&P 2026**, pp. 2366--2384, doi 10.1109/SP63933.2026.00205. Full author list (9). Added `file`. | https://doi.org/10.1109/SP63933.2026.00205 ; https://arxiv.org/abs/2604.21197 |
| jiang2026fisguard | yes | The arXiv id and title match. The paper is v1 only (28 Aug 2026), with no venue. Authors confirmed: Haocheng Jiang and Hua Shen. Added primaryClass cs.CR and `file`. | https://arxiv.org/abs/2608.27836 |
| mao2025fede4rag | yes | arXiv only (v1). No published version found on Crossref or OpenReview. Full author list (14). Added primaryClass cs.CL and `file`. | https://arxiv.org/abs/2504.19101 |
| anderson2024membership | yes (venue changed) | Changed @misc to @inproceedings. **Published at ICISSP 2025** (Vol. 2, SciTePress), pp. 474--485, doi 10.5220/0013108300003899. The year is now 2025; the key still says 2024 and was kept. Full author list (3). Added `file`. | https://doi.org/10.5220/0013108300003899 ; https://arxiv.org/abs/2405.20446 |
| carlini2022first | yes | Full author list (6, with Tram{\`e}r). Booktitle is "2022 IEEE Symposium on Security and Privacy". Added pages 1897--1914 and doi 10.1109/SP46214.2022.9833649. Added `file`. | https://doi.org/10.1109/SP46214.2022.9833649 |
| abadi2016deep | yes | Full author list (7). Booktitle is now the full CCS '16 proceedings title. Added pages 308--318 and doi 10.1145/2976749.2978318. Accent changed to {\'\i}. Added `file`. | https://doi.org/10.1145/2976749.2978318 |
| bun2016zcdp | yes | Booktitle is "Theory of Cryptography: 14th International Conference, TCC 2016-B, Proceedings, Part I". Added series LNCS, volume 9985, pages 635--658, publisher, and doi 10.1007/978-3-662-53641-4_24. Volume 9985 comes from secondary sources, because Springer and DBLP blocked automated access. Added `file`. | https://doi.org/10.1007/978-3-662-53641-4_24 |
| biderman2023emergent | yes | Full author list (7). Added volume 36 and pages 28072--28090 (NeurIPS 2023 proceedings). Added `file`. | https://papers.nips.cc/paper_files/paper/2023/hash/59404fb89d6194641c69ae99ecdf8f6d-Abstract-Conference.html |
| rajpurkar2016squad | yes | Added pages 2383--2392 and doi 10.18653/v1/D16-1264. Added `file`. Authors were already complete. | https://aclanthology.org/D16-1264/ |
| qwen2025 | yes | Full author list: "{Qwen Team}" plus 43 named authors, as listed on arXiv. Added primaryClass cs.CL and a note "Version 2, 3 January 2025" (same style as chen2026window). Added `file`. Note: v1 is dated 19 Dec 2024, so some styles would use year 2024. | https://arxiv.org/abs/2412.15115 |
| beutel2020flower | yes | arXiv only (v5, 2022); no archival venue. Full author list (11, with Gusm{\~a}o). Added primaryClass cs.LG and `file`. | https://arxiv.org/abs/2007.14390 |
| mcmahan2017communication | yes | Added series PMLR, volume 54, and pages 1273--1282. Added `file`. Authors unchanged (PMLR prints "Brendan McMahan"; I kept "H. Brendan McMahan" as on arXiv). | https://proceedings.mlr.press/v54/mcmahan17a.html |
| lewis2020rag | yes | Full author list (12, with accents). Added volume 33 and pages 9459--9474. Added `file`. | https://proceedings.neurips.cc/paper_files/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html |

All 21 arXiv ids exist and match the titles in the source file, including the recent ids 2512.03100, 2604.21197, 2605.24312 and 2608.27836. zeng2024good had no arXiv id; I added 2402.16893, which has the same title and authors as the ACL record.

### Proposed PDF filenames (reference_papers/)
Lin_2024_RA_DIT.pdf, Zhang_2024_RAFT.pdf, Fajardo_2025_FedRAG.pdf, Zeng_2024_Good_Bad_RAG.pdf, Fu_2026_Ensemble_Privacy_Defense.pdf, Choi_2025_Mirabel.pdf, Naseh_2025_Riddle_Me_This.pdf, Nguyen_2026_MEntA.pdf, Deng_2026_ProjRes.pdf, Jiang_2026_FISGuard.pdf, Mao_2025_FedE4RAG.pdf, Anderson_2025_RAG_MIA.pdf, Carlini_2022_LiRA.pdf, Abadi_2016_DP_SGD.pdf, Bun_2016_zCDP.pdf, Biderman_2023_Emergent_Memorization.pdf, Rajpurkar_2016_SQuAD.pdf, Qwen_2025_Qwen2_5.pdf, Beutel_2020_Flower.pdf, McMahan_2017_FedAvg.pdf, Lewis_2020_RAG.pdf

## B. Author-supplied 15 entries (kept unchanged in the output; findings and proposals only)

I found no clear factual errors: the titles, authors, pages, DOIs and arXiv ids all match their primary records. The proposals below are publication updates and optional completions.

| key | verified? | finding / proposed change | primary record URL |
|---|---|---|---|
| nguyen2023active | yes | Matches PMLR 206:5714--5730. | https://proceedings.mlr.press/v206/nguyen23e.html |
| nguyen2025theoretically | yes | Optional: add `pages = {46211--46241}`. PMLR and OpenReview give the first author as "Quan Minh Nguyen"; arXiv gives "Quan Nguyen". | https://proceedings.mlr.press/v267/nguyen25j.html |
| yao2026federated | yes, **now published** | **Accepted at PAKDD 2026** (per the arXiv comment). It appears in *Advances in Knowledge Discovery and Data Mining: PAKDD 2026, Proceedings, Part IV*, LNCS (LNAI), Springer, pp. 203--221, doi 10.1007/978-981-92-1468-6_12. Proposed change: switch to @inproceedings with that booktitle, pages and doi. The LNAI volume number is unconfirmed: the set spans 16597--16600, so Part IV is probably 16600, but I could not check this because Springer blocked automated access. | https://doi.org/10.1007/978-981-92-1468-6_12 ; https://arxiv.org/abs/2409.15723 |
| yan2025federated | yes, **now published** | **Published in IEEE Communications Magazine** 63(10):52--58, Oct 2025, doi 10.1109/MCOM.001.2400770. Proposed change: switch to @article with journal, volume 63, number 10, pages 52--58 and doi. | https://doi.org/10.1109/MCOM.001.2400770 |
| wu2026survey | partial | arXiv v3 says "Accepted by TMLR, 2026". I could not open the TMLR OpenReview page (it requires a challenge), so the forum id is not confirmed. The arXiv title uses "Fine-tuning" in lowercase, which is cosmetic. | https://arxiv.org/abs/2503.12016 |
| cheng2025toward | yes | Matches Crossref: vol 27, no 4, pp. 2733--2764. | https://doi.org/10.1109/COMST.2024.3503680 |
| yeom2018privacy | yes | Matches. | https://doi.org/10.1109/CSF.2018.00027 |
| carlini2021extracting | yes | Matches (arXiv drops the accents; the entry is correct to keep them). | https://www.usenix.org/conference/usenixsecurity21/presentation/carlini-extracting |
| shi2024detecting | yes | Matches (ICLR 2024). | https://arxiv.org/abs/2310.16789 |
| mattern2023membership | yes | Matches. | https://aclanthology.org/2023.findings-acl.719/ |
| zhang2025minkpp | yes | Matches (ICLR 2025 spotlight). | https://arxiv.org/abs/2404.02936 |
| chen2026window | yes, **now published** | **Published at USENIX Security 2026**: "35th USENIX Security Symposium", pp. 1187--1206. Proposed change: switch to @inproceedings with booktitle {35th USENIX Security Symposium} and pages, and drop primaryClass and the version note. The official title capitalizes as "Window-based ... Fine-tuned", which is cosmetic. | https://www.usenix.org/conference/usenixsecurity26/presentation/chen-yuetian |
| xie2024recall | yes | Matches. | https://aclanthology.org/2024.emnlp-main.493/ |
| kaneko2024sampling | yes, **now published** | **Published in Findings of ACL 2025**, pp. 8894--8907, doi 10.18653/v1/2025.findings-acl.465. Proposed change: switch to @inproceedings, and the year becomes 2025 (the key says 2024). The official title is "Sampling-based", which is cosmetic. | https://aclanthology.org/2025.findings-acl.465/ |
| fu2024membership | yes | The NeurIPS 2024 title matches the entry; arXiv v4 is titled "Practical Membership Inference Attacks ...". Optional: add `pages = {134981--135010}` and `doi = {10.52202/079017-4290}` from Crossref. | https://doi.org/10.52202/079017-4290 |

## C. PDFs

On 2026-10-04 the 21 PDFs named in the `file` fields of section A were downloaded from arXiv (`https://arxiv.org/pdf/<eprint>`, the latest version) into `reference_papers/`. Every `file` field in `references.bib` now resolves to a PDF in this folder (36 in total). Where a paper later appeared at a venue, the arXiv PDF may differ from the published version.
