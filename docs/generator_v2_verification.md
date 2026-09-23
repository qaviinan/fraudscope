# Generator v2: what was built and what the verification shows

Companion to `foundational_flow_review.md` (section 10). Every number here comes from
`python scripts/verify_generator_v2.py --out docs/generator_v2_verification.json`
(120,000 transactions, seed 42, about 4 minutes on 4 cores). The raw JSON sits next to this file.

## 1. What changed

| Layer | v1 | v2 |
|---|---|---|
| Generator | `synthetic.py`: independent per-user draws from IEEE marginals, label = noisy logistic on transaction terms | `synthetic_v2.py`: actor-level population with planted fraud economics (below) |
| Graph features | SVD of one-hot categoricals | `graph_features.py`: causal neighbour aggregates per card / address / device, plus two-hop reach; SVD kept as a baseline |
| Objective / scores | asymmetric focal, uncalibrated (all scores 0.42–0.55) | `--objective logistic --calibrate`: log-loss plus isotonic calibration on validation |
| Graph entities | 7 columns incl. email domain, OS, browser | card1, addr1, DeviceInfo only (identity-like); attributes are never nodes |
| Star score | degree × mean risk (a degree ranking) | (mean risk − base rate) × √degree; missing values excluded |
| Ring candidates | pairs sharing ≥2 attributes, mostly one account with itself | must be distinct accounts, missing values excluded |
| Evaluation | ROC, PR-AUC, recall@90%P | plus precision@k, recall@k, fraud dollars captured@k for k ∈ {100, 500, 1000} |
| Snapshot export | seeded from one overview transaction | seeded from overview + star hubs + ring candidates |

The build is opt-in and v1 remains the default: `python scripts/build_backend_api_artifacts.py --generator v2 --objective logistic --causal-graph-features --calibrate`.

### The mechanisms planted in v2

- **Households.** Customers grouped 1–4 per household sharing an address, half sharing a device, a third sharing a card. This is the benign sharing that makes "shared entity" ambiguous.
- **Rings.** 90 rings, each owning 4–15 stolen cards, 2–6 devices, 1–4 drop addresses and a few email domains, running 6–25 synthetic accounts that draw from the pool with reuse probability 0.6–0.9. Rings live 7–30 days, act in bursts (median in-burst gap 60 min; 45% of rings are "low and slow" at 6× that), open a third of bursts with 1–3 micro-amount card tests, and burn cards after 5–25 uses. Half the ring accounts are freshly created, half are aged.
- **Account takeover.** 1.5% of customers have a short high-value burst from a ring device on their own card.
- **First-party fraud.** 20% of fraud is chargeback abuse on ordinary transactions. It is structurally invisible, which is what keeps the Bayes ceiling below 1.
- **Labels.** Ring and ATO transactions are fraud with an 8% unreported fraction; every ordinary transaction carries the small first-party probability. `true_p` records the generator's own probability so the ceiling is measurable.
- **Identity table.** Present for 4% of W, 62–92% of C/H/R/S transactions (as in the real data), with device fingerprints (`model|fpNNNNNN`), OS and browser consistent with the device.
- **Identity resolution.** `uid_true` is hidden. The pipeline sees `uid_clean = uid_proxy = card1 + addr1 + (day − D1)`, the Kaggle construction, degraded by card reissue on 6% of accounts (configurable for the noise sweep).
- **Causal C columns.** C1–C5 are running counts on the account, card, device, email and address; C6–C10 are running "distinct addresses per card", "distinct cards per device" and similar, which is what the real Vesta C columns are.
- **Overlap by design.** Customers also use anonymous.com (7%), also open new accounts (25% inside the horizon), also buy in sessions (35% follow-up purchases within two hours), and rings mostly buy W products on gmail addresses. Without this overlap the first version of v2 reached ROC 0.998 from transaction features alone, which no fraud scientist would believe.

## 2. Does the data look like a fraud book?

| Check | Result |
|---|---|
| Transactions / reported fraud rate | 119,976 / 3.44% |
| Actor mix | 115,567 customer, 3,201 ring, 438 ATO, 770 first-party |
| True accounts / observable proxy accounts | 10,584 / 11,319 (reissue splits, ring merges) |
| Identity table coverage, by product | 24.0% overall; W 4%, C 92%, H 75%, S 81%, R 63% |
| Fraud rate with / without identity row | 5.5% / 2.8% |
| Fraud rate by product | W 2.6%, R 4.5%, S 5.7%, C 6.0%, H 7.6% |
| Median gap to the account's previous transaction | customer 95 h, ring 34 h, ATO 0.17 h |
| Share of ring transactions under $5 (card testing) | 10.7% vs 0.1% for customers |
| Median account age (D1) | customer 549 d, ring 308 d |
| Fraud transactions per ring, min / median / max | 19 / 36 / 53 |

## 3. Is there graph structure, and is it the right kind?

Entity sharing across *true* accounts, and what sharing means for fraud:

| Entity | Values used by >1 account | Fraud rate on shared vs unshared values | Share of shared traffic that is benign |
|---|---|---|---|
| card1 | 9.1% | 8.6% vs 2.4% | 90.7% (household cards) |
| addr1 | 41.8% | 3.6% vs 3.1% | 96.2% (households) |
| DeviceInfo | 7.1% | 29.5% vs 2.4% | 67.9% (household devices) |

This is the shape a fraud team expects: an address shared by several accounts means almost nothing on its own, a shared device is a strong but not conclusive signal, and the discriminating information is in *how many unrelated accounts* touch an entity *in a short window* combined with the entity's outcome history. That is exactly what the causal graph features measure, and they are what the model picks up: the top features of the graph model are `other accounts on this device in 24 h`, `C6 (distinct addresses on this card)`, `other accounts on this address in 24 h`, `addresses reachable through this card's devices (2-hop)` and `prior known fraud rate on this address`.

Graph-store queries on calibrated scores:

- **Overview** (single highest-scored seed, 2 hops): 12 transactions from 5 accounts, all fraud, linked through one card and one drop address. The deployed snapshot's overview is a 34-transaction ring (31 fraud) through one address and one shared iOS device.
- **Star patterns** (top 10): all ring drop addresses, 28–43 transactions each, mean score 0.86–0.91, true fraud rate 84–100%. No `__MISSING__` hubs.
- **Ring candidates** (top 200): 98.5% link two *different* accounts, 96% have both transactions fraudulent.

## 4. The Bayes ceiling and the ablation

Test split: last 15% of the horizon, 18,000 transactions. Log-loss XGBoost, isotonic-calibrated on validation.

| Variant | Features | PR-AUC | ROC-AUC | P@100 | P@500 | R@500 | Fraud $ @500 |
|---|---|---|---|---|---|---|---|
| 1. transaction features (incl. causal C1–C5, D, M) | 56 | 0.672 | 0.871 | 0.95 | 0.658 | 0.697 | 0.807 |
| 2. + UID aggregates (v1 style) | 63 | 0.383 | 0.854 | 0.76 | 0.400 | 0.424 | 0.454 |
| 3. + Vesta-style C6–C10 counts | 68 | 0.638 | 0.876 | 0.92 | 0.614 | 0.650 | 0.746 |
| 4. + causal graph features | 100 | **0.697** | 0.867 | 0.93 | **0.698** | **0.739** | **0.833** |
| 4b. + SVD embeddings | 132 | 0.675 | 0.863 | 0.91 | 0.666 | 0.706 | 0.804 |
| 5. causal graph features only | 32 | 0.641 | 0.865 | 0.96 | 0.594 | 0.629 | 0.750 |
| 6. SVD embeddings only | 32 | 0.061 | 0.727 | 0.00 | 0.012 | 0.013 | 0.016 |
| Oracle (generator's `true_p`) | – | 0.716 | 0.881 | 0.90 | 0.720 | 0.763 | 0.855 |

Reading it:

- **The ceiling is realistic.** PR-AUC 0.72 / ROC 0.88 at a 3.4% base rate, held down by first-party fraud and the unreported fraction. Against v1's ceiling of 0.088 this is a different problem class.
- **Graph features buy something measurable.** Transaction features alone reach 0.672; adding the causal graph block reaches 0.697, i.e. 97% of the ceiling, and lifts recall at a 500-case queue from 0.70 to 0.74 and dollars captured from 0.81 to 0.83. Graph features *alone* (32 numbers, no amount, no product, no card) reach 0.641. This is the "finding" the review asked for, and it is honest: the lift is a few points, not a miracle, because velocity on the account's own card already captures most burst behaviour.
- **The SVD embedding is confirmed useless.** Alone it is barely above the base rate; added on top it subtracts.
- **v1's UID aggregates are actively harmful (0.672 → 0.383).** They are computed on the training window and applied to the training rows themselves, so `uid_tx_count` counts an account's *future* training transactions and `uid_seen_in_train` is 1 for every training row and 0 for most ring accounts at test time. The model learns a train-time artifact that inverts at test time. The fix is the same as for the graph block: make account aggregates causal (trailing windows, prior rows only). This should be the next change to `features.py`.
- **The UID blend now helps a little** (variant 4: 0.697 → 0.718) because fraud is account-level in v2. In v1 it hurt.
- **Calibration holds.** Reliability on the test split for variant 4: predicted 0.85 → observed 0.86, predicted 0.92 → observed 0.94, predicted 0.01 → observed 0.008. Scores now span 0.00–1.00 and the UI's 0.65 threshold flags 1.9% of test rows.

## 5. Breaking the clean-identity assumption

Card reissue rate swept 0 → 30% (each rate regenerates the dataset, so single-run values carry a few points of noise):

| Reissue rate | Proxy accounts per true account | PR-AUC, transaction + UID aggregates | PR-AUC, transaction + causal graph |
|---|---|---|---|
| 0% | 1.04 | 0.465 | 0.743 |
| 6% | 1.07 | 0.383 | 0.712 |
| 15% | 1.13 | 0.426 | 0.717 |
| 30% | 1.23 | 0.365 | 0.653 |

Graph features degrade gracefully (0.74 → 0.65 at 30% identity noise) because entity-level counts do not need the account id at all; the UID-aggregate model is already broken at 0% for the in-sample reason above and stays there. Once account aggregates are made causal this sweep becomes the clean "graph is more robust than identity resolution" comparison the portfolio write-up wants.

## 6. What is still open

1. Causal account aggregates in `features.py` (replace the in-sample UID block; see section 4).
2. The demo explainer still reads `isFraud` and its clipped waterfall does not sum; label it as scripted or remove the label dependence.
3. Frontend cosmetics: colour the embedding scatter by label/score, label restraint on dense clusters, a default "why is this risky" state. The calibrated scores and named graph features already make the existing panels read correctly.
4. Ring detection is still pairwise; connected components on the account-projected graph would show whole rings rather than pairs.
