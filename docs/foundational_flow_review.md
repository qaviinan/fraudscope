# FraudScope: foundational flow review

A stage-by-stage walkthrough of what the pipeline does, the math and economic reasoning behind each step, and what each step actually produces. Every number below is either read from the deployed snapshot in `frontend/public/snapshots/main.json` or reproduced by `scripts/diagnose_foundations.py` (see the appendix). The reproduction uses the repo's own generator, features, objective, and graph store unchanged.

## 0. The verdict in one page

The dashboard looks broken because the data layer is broken, not because the model or the UI are badly built. Five upstream facts explain every visible symptom.

1. **The synthetic labels are almost pure noise.** The generator draws each label as a Bernoulli coin whose probability is 0.03 for the median transaction and never exceeds 0.53. Even a model that knows the generator's exact probabilities (the Bayes oracle) only reaches PR-AUC 0.088 and ROC-AUC 0.656. The trained model reaches PR-AUC 0.073. There is nothing left to find; no modelling change can make this data look good.
2. **There is no graph structure to discover.** Every user's card, address, email, device, OS, and browser are independent draws. Two accounts only share a card by rounding collision, and share an address because the address column has 564 values for 12,000 users. Fraud rings, device farms, mule addresses, and bursty cash-out behaviour do not exist in the generator, so no graph method can find them. The one "shared device" term in the label equation is dead code and fires on 0.0% of transactions.
3. **The graph embeddings are a categorical embedding, not a graph embedding.** The "SVD of the transaction–entity incidence matrix" is mathematically an SVD of one-hot encoded categoricals. Ablation: removing all 32 embedding dimensions changes PR-AUC from 0.0755 to 0.0766. They buy nothing, as expected from point 2.
4. **The asymmetric focal objective destroys probability calibration.** With a 3.5% positive rate, its constant-predictor optimum is p*=0.355 (plain log-loss: 0.035). Every deployed score lives in [0.42, 0.55]. The UI's colour thresholds are 0.45 and 0.75, so 0% of the main dataset can ever render as high risk and the header always reads "0 high-risk". Ranking is unaffected; only the scale is.
5. **The pattern queries measure degree, not risk.** Star score = degree × mean risk, and mean risk is nearly constant (0.36–0.46), so the "star patterns" list is a degree ranking topped by `__MISSING__` hubs and gmail.com. Ring score rewards pairs of transactions sharing ≥2 attributes; in this generator all 7 attributes are constant per user, so the "rings" are two purchases by the same customer. The overview graph is seeded from exactly one transaction (a hard-coded `seed_count = 1`), and the degree filter removes 5 of its 7 entities, leaving one address hub with ~100–200 random transactions hanging off it.

What this means for the three planned upgrades: the ablation (upgrade 1) is the right instrument, but run on today's generator its honest result is "graph features add nothing", which is true and useless. The identity-noise experiment (upgrade 2) is a null result today: merging or splitting 30% of identities moves PR-AUC by less than 0.002 because the UID blend already contributes nothing (it lowers PR-AUC from 0.0755 to 0.0697). Cosmetics (upgrade 3) cannot fix a colour scale whose inputs are all 0.43.

The fix is still bounded, but it starts one layer lower than planned: rebuild the generator so that it contains the economic mechanisms that make fraud graph-shaped (section 10), swap or calibrate the objective, replace the two pattern scores, and then run the ablation and the noise experiment. That is a week of focused work; the current API, snapshot exporter, and frontend can stay.

---

## 1. The data

### 1.1 The real dataset the project is modelled on

IEEE-CIS Fraud Detection (Vesta Corporation, Kaggle 2019). Two tables joined on `TransactionID`:

| Table | Rows | Columns | What it is |
|---|---|---|---|
| `train_transaction` | 590,540 | 394 | One row per card-not-present e-commerce transaction over ~6 months, `isFraud` label (3.5% positive) |
| `train_identity` | 144,233 | 41 | Device, browser, OS and network fingerprints, present for only ~24% of transactions |

Column families and the economic meaning that matters here:

| Family | Meaning | Why fraud teams care |
|---|---|---|
| `TransactionDT` | Seconds since a reference instant | Ordering, velocity, and the fact that test data is *later* than train (concept drift) |
| `TransactionAmt` | Purchase amount (USD) | Loss size; fraud is skewed to specific amount bands and to "card testing" micro-amounts |
| `ProductCD` | Product category (W, C, H, R, S) | Resale value and refund policy differ by product, so fraud rates differ (C and H are much higher than W) |
| `card1`–`card6` | Card issuer/BIN-like identifiers, network, debit/credit | `card1` + `addr1` is the closest thing to a customer identity in the data; a stolen card reused across many "customers" is the core fraud signature |
| `addr1`, `addr2` | Billing region and country codes | A drop address serving many cards is a mule signature |
| `dist1`, `dist2` | Distances (billing to shipping, etc.) | Shipping far from billing is a classic risk signal |
| `P_emaildomain`, `R_emaildomain` | Purchaser and recipient email *domains* | A domain is an attribute, not an identity: gmail.com is 39% of rows. Mismatch and throwaway domains carry mild signal |
| `C1`–`C14` | Vesta counts: how many addresses, emails, devices, etc. are associated with this card | These *are* graph features, precomputed by Vesta: degree counts of the card node |
| `D1`–`D15` | Time deltas: days since the card was first seen, since last transaction, etc. | Account age and velocity |
| `M1`–`M9` | Match flags: does the name on the card match the billing name, etc. | Identity consistency |
| `V1`–`V339` | Vesta-engineered features (mostly rank/count transforms) | Redundant, high-value; not used by this project |
| `id_01`–`id_38`, `DeviceType`, `DeviceInfo` | Fingerprint numerics and strings | Device reuse across accounts is the strongest ring signal; missing for 76% of rows |

Two facts about the real data drive everything downstream. First, **there is no customer ID**. The well-known "magic" of the competition (Chris Deotte's `uid = card1 + addr1 + D1n`, the notebook `ieee-uid-detection-v6.ipynb` in this repo) is an entity-resolution heuristic: reconstruct the account from card, region, and account-birth date, then aggregate. Second, **graph structure in the real data is carried by reuse of identity artifacts across accounts**: the C-columns are literally degree counts and top solutions built hundreds of "how many distinct X per uid" aggregates.

### 1.2 What the pipeline actually takes from the real data

`synthetic.load_reference_profiles` reads the first 150,000 rows of each CSV (time-ordered, so roughly the first 6 weeks) and keeps only:

- the marginal frequency table of 13 categorical columns (`ProductCD`, `card4`, `card6`, both email domains, `DeviceType`, `DeviceInfo`, `id_30`, `id_31`, `id_33`, `id_36`, `id_37`, `id_38`), with NaN mapped to the literal string `__MISSING__`;
- the median and standard deviation of 8 numeric columns (`TransactionAmt`, `card1`, `card2`, `card3`, `card5`, `addr1`, `addr2`, `D1`);
- the fraud rate.

Everything else, in particular every joint distribution, every dependency between columns, every C, D, V and M column, every `dist`, and every relationship between fraud and any column, is discarded. The real data contributes marginal histograms and nothing else. The raw CSVs are not in the repo (`data/` is git-ignored), so nobody reviewing the repo can inspect them; the diagnostic script therefore falls back to hand-coded approximations of the public marginals (appendix).

### 1.3 The synthetic generator, equation by equation

`synthetic.generate_synthetic_ieee_data` with `n_tx = 120,000` (the deployed snapshot reports `transaction_count: 120000`).

**Users.** `n_users = max(10,000, n_tx / 10) = 12,000`. Each user u gets an activity weight from a Pareto draw, `a_u = Pareto(α=1.4) + 0.05`, and each transaction picks its user with probability `a_u / Σa`. Pareto with α=1.4 has infinite variance, so activity is extremely concentrated:

| Statistic (reproduced) | Value |
|---|---|
| Users with ≥1 transaction | 9,825 of 12,000 |
| Transactions per active user, median / p90 / p99 / max | 4 / 22 / 125 / 3,906 |
| Share of all transactions from the top 1% of users | 32.7% |

Why this matters: a single "user" with 3,906 purchases in 180 days is not a consumer, and the per-user aggregates and timelines downstream are dominated by these tails.

**Time.** Each transaction draws a day uniformly on [0, 180) and a second uniformly within the day, independently of the user and of every other transaction. There is no daily or weekly seasonality, no session structure, and no bursts. Consequence for the "velocity" story: the median gap between a user's consecutive transactions is 47.8 hours and the median user is active on 4 distinct days.

**Attributes.** Every categorical attribute is drawn once per user from the real marginal and then copied to all of the user's transactions: `ProductCD`, `card4`, `card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`, `id_30`, `id_31`. Numeric identifiers are drawn per user from a clipped normal using the real median and std:

```
card1_u = max(1000, round(N(9678, 4901)))     addr1_u = max(1, round(N(299, 101)))
card2_u = max(100,  round(N(361, 157)))       addr2_u = max(1, round(N(87, 2.7)))
card3_u = max(100,  round(N(150, 11.3)))      card5_u = max(100, round(N(226, 41)))
```

Three consequences, all visible in the graph later:

- `card1` is meant to be an identity-like key, and mostly is: 7,206 distinct values, median 1 user per value. But 25% of values are shared by more than one user through rounding collisions, and the clip at 1000 piles up the lower tail: the single value `card1 = 1000` carries 4,051 transactions. Neither is a fraud mechanism; both create spurious links.
- `addr1` has 564 distinct values for 12,000 users, so the median address is shared by 14 unrelated users. In the real data `addr1` is a *region* code, which is why sharing it means nothing; the pipeline nonetheless treats it as a linking entity.
- The five string columns (`P_emaildomain` 21 values, `R_emaildomain` 10, `DeviceInfo` 13, `id_30` 16, `id_31` 20) are attributes with a handful of values each. Between 96.6% and 99.3% of transactions sit in a value with degree above 700, and the `__MISSING__` value alone carries 76–89% of rows for the identity columns.

**Device spoofing.** With probability 0.08 per transaction, `DeviceInfo` and `id_31` are redrawn from the marginal. This is the only within-user attribute variation in the generator. It represents "a user's device changed", which in isolation is a weak signal.

**Amounts.** Per user, `μ_u = N(log(1+68.77), 0.45·σ_amt)`, `σ_u = clip(N(0.55, 0.20), 0.15, 1.2)`; per transaction `amt = clip(exp(N(μ_u, σ_u)), 1, 40,000)`. A reasonable lognormal customer model with no fraud-specific behaviour.

**D columns.** `D1 = day − birth_day_u` with `birth_day_u ~ U(−1300, −20)`, i.e. account age in days, which is exactly the quantity Deotte's UID normalises. `D2…D15 = max(0, D1 − U(0, 15·i))`: noisy copies of D1 with no independent meaning.

**C columns.** `C1 = number of transactions by the user over the whole 180 days`, `C2 = count of transactions with the same card1`, `C3 = same DeviceInfo`, `C4 = same P_emaildomain`, `C5 = same addr1`; `C6…C14 = clip(0.45·C1 + 0.25·C2 + N(0, 3), 0)`. Two points. These are computed over the full horizon including the future, so every row's features contain information about transactions that have not happened yet (feature leakage across the time split, though not label leakage). And `C3`, `C4`, `C5` are degrees of hub attributes, so they mostly encode "is your device string missing" and "is your email gmail".

**M columns.** Independent coin flips, `P(M_i = T) = max(0.5, 0.93 − 0.04i)`, unrelated to anything.

**Identity numerics.** `id_k = N(0,1) + 0.6·risk_u` for the 24 non-categorical id columns, where `risk_u ~ N(0,1)` is a hidden per-user latent that also enters the label. This is the only place the generator plants a shared latent, and it leaks the label's user-level component into 24 noisy features at once.

### 1.4 The label model

```
z_i = 1.8·[amt_i > q97(amt)]            high amount
    + 1.2·[C3_i > q90(C3)]              "shared device"
    + 0.9·[ProductCD_i ∈ {C, W}]        "risky product"
    + 0.8·[P_email_i ≠ R_email_i]       email mismatch
    + 0.7·[spoofed_i]                   device switch
    + 0.5·[C2_i > q92(C2)]              card with many transactions
    + 0.6·[risk_u(i) > 1.1]             latent user risk
z_i ← z_i − mean(z);   choose shift b by bisection so that mean σ(z + b) = 0.035
y_i ~ Bernoulli(σ(z_i + b))
```

The economic intent is legible: large purchases, shared infrastructure, risky product lines, identity inconsistency, device changes, and a bad-actor latent. The execution defeats the intent:

| Term | Fires on | Fraud rate when true / false (reproduced) | Comment |
|---|---|---|---|
| high amount | 3.0% | 15.1% / 3.2% | The only strong term |
| shared device | **0.0%** | n/a | Dead: `DeviceInfo` is 80% `__MISSING__`, so the 90th percentile of `C3` *is* the missing-value count and nothing exceeds it |
| risky product | 87.8% | 3.8% / 1.9% | W is the *low*-risk product in the real data (it is 74% of volume); this term inverts reality and fires on almost everyone |
| email mismatch | 80.0% | 4.0% / 1.8% | Fires on 80% of rows because `R_emaildomain` is 77% missing; a "mismatch" mostly means "recipient unknown" |
| device switch | 8.0% | 6.7% / 3.3% | Fine, but weak |
| busy card | 6.6% | 5.4% / 3.4% | Mostly the `card1 = 1000` clip artifact and heavy users |
| latent user risk | 16.7% | 5.2% / 3.2% | Weak, and leaked through the 24 `id_k` features |

Because the terms are small and mostly fire on everyone or no one, the resulting Bernoulli probabilities are compressed:

| Quantile of the true probability p_i | min | p50 | p90 | p97 | p99 | max |
|---|---|---|---|---|---|---|
| Value | 0.006 | 0.030 | 0.053 | 0.101 | 0.157 | 0.530 |

Only 40 distinct probability values exist. The correlation between p_i and the drawn label is 0.15. This is the single most important number in the report: **the generator's own oracle, ranking by p_i, achieves PR-AUC 0.088, ROC-AUC 0.656, precision@100 of 0.28 on the 18,000-row test split.** The real IEEE-CIS competition reached ROC-AUC ≈ 0.94–0.95 on held-out data; a plausible synthetic target is a ceiling around ROC 0.90–0.95 and PR-AUC 0.5–0.7 at a 3.5% base rate. This generator is roughly an order of magnitude less learnable than the data it imitates.

### 1.5 What the generator structurally omits

The economics that make fraud graph-shaped, and why each is absent:

- **Infrastructure reuse across accounts.** A fraud operation has fixed costs: stolen card data, drop addresses, mule bank accounts, devices, phone numbers. Those assets are amortised across many synthetic identities because minting a truly independent identity per attempt is expensive. This reuse is *why* graph features work: a device seen with 30 different cards, or a shipping address serving 12 different emails, is the signature. The generator draws every account's assets independently; there is no ring, no shared pool, no reuse probability.
- **Temporal burstiness.** Stolen credentials decay quickly (cards get blocked, accounts get locked), so fraud concentrates in short windows: card testing at small amounts followed by cash-out within hours. Velocity features exist to measure this. The generator draws timestamps uniformly and independently.
- **Benign sharing.** Households share addresses and devices; corporate cards share billing addresses; email domains are shared by everyone. The detection problem is separating benign sharing from adversarial reuse, which needs both to exist. The generator has neither.
- **Loss-weighted outcomes.** A missed \$900 fraud and a missed \$9 fraud are not the same event to a fraud team; a false positive costs friction and margin, not the ticket. The label model ignores amount except through the top-3% indicator.
- **Missingness as a mechanism.** In the real data identity fields are present for ~24% of rows and the presence pattern itself is informative (mobile app vs web, product code). Here missingness is an i.i.d. marginal draw. The `__MISSING__` entries the screenshot shows are not "realism": they are 76–89% of the identity columns, and the graph module later filters them out entirely (section 7).

---

## 2. Cleaning and encoding

### 2.1 Merge and time split (`backend_builder.build_backend_artifacts`, `_time_split`)

The identity table is left-joined onto transactions, rows are sorted by `TransactionDT`, and split 70/15/15 in time: train 84,000, valid 18,000, test 18,000 (636 fraud). A time split is the correct choice for fraud because the production model always scores the future, and the real competition's test set is later than train. Two caveats specific to this pipeline: the C-columns already contain full-horizon counts (section 1.3), and `TransactionDT` itself is a model feature, which for a time split means the test set lies entirely outside the training range of that feature.

### 2.2 UID aggregates (`features.fit_uid_aggregates`, `apply_uid_aggregates`)

For each `uid_clean` in the training split:

```
uid_tx_count = n_u,  uid_amt_mean, uid_amt_std, uid_amt_median,  uid_d1_mean,  uid_c1_mean,  uid_seen_in_train = 1
```

Unseen UIDs in valid/test receive the column mean and `uid_seen_in_train = 0`. This is the standard "group aggregation" family from the Kaggle solutions and the intent is right: fraud is an account-level phenomenon, and a transaction's deviation from its account's norm is informative.

The catch is the key. `uid_clean` is the generator's user index, written straight into the data. In the real problem the UID must be *reconstructed* from `card1 + addr1 + D1n` and the reconstruction is noisy (card reissues split an account, shared regions merge accounts). The project's README calls this "assumes clean entity IDs for initial implementation". Because of the Pareto activity, 97.1% of test transactions belong to a UID already seen in train, so the aggregates are almost always populated.

Measured value: removing the 7 aggregate features lowers PR-AUC from 0.0766 to 0.0704 (variants C vs D in section 6). They are the second-most useful feature block, after amount and product code.

### 2.3 Categorical encoding (`backend_builder._encode_categorical`)

For each of the 22 categorical columns, values are ranked by training frequency and mapped to 1, 2, 3, …; unseen values map to 0. Trees do not need one-hot for this, and frequency-rank order is a sensible default because rarity is itself a fraud signal. NaN is mapped to `__MISSING__` before encoding, so "missing" becomes an ordinary (and usually the most frequent) category, which is the right treatment for tree models.

---

## 3. "Graph embeddings" (`graph_embeddings.SparseGraphEmbedder`)

**What it does.** For the 7 relation columns (`card1`, `addr1`, `P_emaildomain`, `R_emaildomain`, `DeviceInfo`, `id_30`, `id_31`), build the transaction × entity incidence matrix `A` (one row per transaction, one column per entity value with training frequency ≥ 2, plus an "unknown" column per relation). `A` has 120,000 rows and 6,126 columns, of which 5,495 are `card1` values and 546 are `addr1` values. Then `TruncatedSVD` to 32 components, fit on train, applied to all rows: `E = A·V₃₂`.

**What it is mathematically.** A row of `A` is the one-hot encoding of 7 categorical features concatenated. `E` is therefore the first 32 principal directions of one-hot categoricals. That is a categorical embedding (it is exactly the "PCA on one-hot" trick). A graph embedding would use the *neighbourhood*: the bipartite two-hop matrix `A·Aᵀ·A` (which transactions are reachable through shared entities), or an entity-projected adjacency `AᵀA`, or a random-walk method. Nothing in `E` depends on any transaction other than its own row, so `E` cannot represent "this card is connected to 30 addresses".

**What it produces.** 32 dimensions explain 53.7% of the variance of `A`. The 2-D PCA of `E` used by the "Embedding Space Snapshot" explains 14.6% and 10.3%. The second axis has correlation 0.77 with "DeviceInfo is missing" and −0.58 with "id_31 is missing". The scatter plot is a picture of the missingness pattern of the identity table, which is the highest-variance thing in a one-hot matrix dominated by `__MISSING__` columns.

**Measured value.** Section 6: with the embeddings 0.0755 PR-AUC, without them 0.0766, embeddings alone 0.0493 (base rate is 0.035). The model nonetheless assigns 33% of its total split gain to `graph_emb_*` features, which is a symptom of fitting noise, not of signal.

---

## 4. The model (`modeling.train_model`)

### 4.1 XGBoost configuration

`max_depth 8, eta 0.05, subsample 0.85, colsample_bytree 0.85, λ = 2, 600 rounds, early stopping 50 on validation aucpr`. Standard and defensible. In the reproduction early stopping triggers at iteration 34 (repo objective) and 8 (plain logistic): the validation PR-AUC stops improving after a handful of trees because there is almost nothing to learn.

### 4.2 The asymmetric focal objective (`asymmetric_focal_objective`)

Definition (Ridnik et al., "Asymmetric Loss"), with p = σ(z), q = 1 − p, γ₊ = 1, γ₋ = 4:

```
L(y, z) = − y · q^γ₊ · log p  −  (1 − y) · p^γ₋ · log q
```

Gradient and Hessian with respect to the margin z, as implemented (verified by differentiation; the code is correct):

```
∂L₊/∂z = q^γ₊ (γ₊ p log p + p − 1)                      ∂L₋/∂z = p^γ₋ (p − γ₋ q log q)
∂²L₊/∂z² = q^γ₊ [ −γ₊ p (γ₊ p log p + p − 1) + p q (γ₊ log p + γ₊ + 1) ]
∂²L₋/∂z² = p^γ₋ [ γ₋ q (p − γ₋ q log q) + p q (1 + γ₋ (log q + 1)) ]
```

The Hessian is clipped at 1e-6 because the focal loss is not convex.

**Intent.** Down-weight the gradient of easy negatives by p^4 so the 96.5% majority does not drown the positives. That is a legitimate way to improve ranking on extreme imbalance.

**Consequence the project did not account for.** The objective changes what a probability means. Consider the best *constant* predictor, the value p* that a model outputs for a row it cannot distinguish. It solves π·∂L₊/∂z + (1−π)·∂L₋/∂z = 0 with π the positive rate. For plain log-loss the solution is p* = π = 0.035. For this objective, solving numerically:

| Positive rate π | p* under focal(γ₊=1, γ₋=4) | p* under log-loss |
|---|---|---|
| 0.035 | **0.355** | 0.035 |
| 0.050 | 0.379 | 0.050 |
| 0.100 | 0.431 | 0.100 |

Check against the deployed artifact: the explainer reports `base_logit = −0.665`, i.e. base probability σ(−0.665) = 0.340. At p = 0.43 the magnitude of the positive-class gradient is 0.53 while the negative-class gradient is 0.059; the objective treats the negatives as if they were 9× rarer than they are. The output is not a probability of fraud and cannot be compared to a threshold.

Observed effect: deployed main-dataset scores span 0.419–0.548 across all 1,600 exported transactions; reproduced scores span 0.359–0.561. With plain log-loss the same data yields 0.025–0.126, which is at least honest about how little the model knows. Ranking quality is identical (ROC 0.626 vs 0.6255) because the objective is monotone in the margin. Everything downstream that consumes the *value* of the score, not its rank, is broken by this: the risk colours (thresholds 0.45 and 0.75), the "high-risk" counter (≥ 0.65: 0.0% of rows qualify), the entity `risk_mean`, the star score, and the waterfall's probability axis.

### 4.3 Explainability (`explain_prediction`)

`pred_contribs=True` returns per-feature SHAP-style logit contributions plus a bias, and the waterfall walks them through the sigmoid. This is the right mechanism. What it has to explain is the problem: the deployed sample explanation's largest contribution is `graph_emb_05` at −0.031 logits, followed by `ProductCD` at +0.019. An investigator cannot act on "latent SVD component 5 moved the logit by three hundredths". With a meaningful generator and named graph features ("this device has been seen with 14 other cards in 48 h") the same code produces a useful panel.

---

## 5. The UID blend (`features.uid_label_propagation_blend`)

```
score_i = (1 − α) · p̂_i + α · mean{ p̂_j : uid_j = uid_i },   α = 0.6
```

computed over all splits at once. The name "label propagation" overstates it: no labels are propagated, only model outputs, and only within an account. Economically it says "an account's history of looking suspicious is 60% of the verdict on its next purchase", which is a reasonable prior when fraud is account-level (an account-takeover or a mule account is bad on every transaction).

In this generator the label is mostly *transaction*-level (the strongest term is the amount), so averaging over the account dilutes it. Measured: PR-AUC drops from 0.0755 to 0.0697 (repo objective: 0.0729 to 0.0657). The deployed score is the blended one, so the UI shows the worse of the two.

Because the blend contributes nothing, the planned identity-noise experiment has nothing to degrade:

| UID noise applied before blending | PR-AUC (blended) |
|---|---|
| none | 0.0697 |
| 5% / 15% / 30% of identities merged into another | 0.0699 / 0.0681 / 0.0693 |
| 5% / 15% / 30% of identities split in two | 0.0698 / 0.0699 / 0.0694 |

The clean-UID assumption is not load-bearing here. It becomes load-bearing only once account-level structure carries the label (section 10).

---

## 6. Evaluation

**As implemented** (`modeling.evaluate_scores`): ROC-AUC, PR-AUC, recall at 90% precision, on validation and test, for raw and blended scores. Written to `outputs/backend_metadata.json`, which is git-ignored, so the repo carries no record of model quality. There is no baseline, no ablation, no precision@k, no dollar-weighted metric, and no calibration check.

**Why the queue metrics matter.** A fraud team reviews a fixed number of cases per day, k. The only part of the ranking that has economic value is the top k: precision@k is the analyst hit rate, recall@k is the share of fraud that gets stopped, and fraud-dollars-captured@k is the loss avoided. ROC-AUC integrates over the whole ranking, 96.5% of which is negatives nobody looks at; on extreme imbalance a model can gain ROC-AUC by reordering the bottom 90% while the queue does not change. PR-AUC and precision@k are the metrics a fraud scientist expects.

**Reproduced results** (test split, 18,000 rows, 636 fraud, `scripts/diagnose_foundations.py`, seed 42):

| Variant | Features | PR-AUC | ROC-AUC | P@100 | P@500 | R@500 | Fraud \$ captured @500 | Score range |
|---|---|---|---|---|---|---|---|---|
| A. repo: focal + all features | 98 | 0.0729 | 0.626 | 0.19 | 0.142 | 0.112 | 0.376 | 0.359–0.561 |
| B. log-loss + all features | 98 | 0.0755 | 0.6255 | 0.18 | 0.142 | 0.112 | 0.395 | 0.025–0.126 |
| C. log-loss, no graph embeddings | 66 | **0.0766** | 0.615 | 0.23 | 0.130 | 0.102 | 0.360 | 0.014–0.219 |
| D. log-loss, no graph, no UID aggregates | 59 | 0.0704 | 0.621 | 0.20 | 0.130 | 0.102 | 0.352 | 0.019–0.151 |
| E. log-loss, graph embeddings only | 32 | 0.0493 | 0.583 | 0.10 | 0.068 | 0.053 | 0.076 | 0.023–0.091 |
| Oracle: generator's own p_i | – | **0.0879** | 0.656 | 0.28 | 0.180 | 0.142 | 0.417 | 0.006–0.530 |
| A/B after UID blend (α = 0.6) | | 0.0657 / 0.0697 | 0.621 / 0.622 | 0.13 / 0.17 | | | | |

Reading it:

- The model is at 83–87% of the Bayes ceiling. Modelling is not the bottleneck; the label mechanism is.
- Graph embeddings: B − C = −0.001 PR-AUC. Nothing to buy because nothing was planted.
- UID aggregates: C − D = +0.006. The only feature block with a measurable contribution, and it is fed by an oracle key.
- Focal vs log-loss: same ranking, different scale. Keep whichever ranks better, then calibrate on validation (isotonic or Platt) before anything consumes the value.
- Nobody should present recall at 90% precision here: it is 0.0016, i.e. one transaction.

---

## 7. The graph store and pattern queries (`graph_module.FraudGraphStore`)

### 7.1 The graph

Bipartite: transaction nodes and entity nodes `ent:<column>:<value>` for the 7 relation columns. Every transaction has exactly 7 entity edges. `include_missing_entities = False` drops `__MISSING__` nodes, and `max_entity_degree` (700 for the overview, 900 for neighbourhoods) drops hubs. Given section 1.3, the filter removes almost every email-domain, device, OS, and browser node. Entities surviving per transaction (reproduced, threshold 700):

| Surviving entities | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Transactions | 16,678 | 14,779 | 78,481 | 9,503 | 486 | 73 |

For 65% of transactions the graph is "my card1 and my addr1"; for 14% it is nothing. The effective graph is two bipartite layers, one of which (`card1`) is nearly one-to-one with users, and the other (`addr1`) is a random hub assignment.

### 7.2 Overview (`overview_graph`)

`max_transactions` is accepted and ignored; `seed_count = 1` is hard-coded. The seed is the single highest-scoring transaction, expanded 2 hops: transaction → its ≤2 surviving entities → the entities' transactions (highest-scored first, capped at `max_entity_fanout = 180`). Deployed result: seed `tx:2033865`, entities `card1:13168` (degree 3) and `addr1:237` (degree 419, capped to 180), 182 transactions of which 2 are fraud, all with scores 0.42–0.55. Reproduced: `card1:13124` (degree 1) and `addr1:147` (degree 89), 89 transactions, 4 fraud. The default view of the product is one region code and the customers who happen to live there.

### 7.3 Neighbourhood expansion (`subgraph_from_seeds`)

Breadth-first over the bipartite graph up to `hops`, with node and edge caps, dropping missing and high-degree entities, and taking the top-scored transactions when an entity's fan-out exceeds the cap. This is a correct and reasonably efficient investigation primitive. It just has nothing to expand into.

### 7.4 Star patterns (`star_patterns`)

```
score(e) = degree(e) × mean{ score_i : i touches e }
```

over entities with degree ≥ `min_degree` (4 in the snapshot export). Intent: a hub whose transactions are risky is a device farm or a mule address. Two failures:

- Scale: across the 1,719 entities with degree ≥ 20, `risk_mean` ranges 0.359–0.464 while degree ranges 20–106,932. The product is a degree ranking. Deployed top 5: `id_30:__MISSING__` (87,106), `DeviceInfo:__MISSING__` (75,852), `R_emaildomain:__MISSING__` (71,850), `id_31:__MISSING__` (67,904), `P_emaildomain:gmail.com` (46,948). The filter that hides these hubs in the graph does not apply to the pattern list.
- Semantics: a star should be scored by *lift* with uncertainty, not by mass. Something like the log-likelihood ratio of the entity's fraud rate against the base rate, or a shrunken excess `degree × (r_e − π) / sqrt(π(1−π)/degree)`, both of which rank a 30-degree device at 40% fraud above a 90,000-degree "missing" at 3.5%. `risk_mean` does carry information (correlation 0.51 with the true entity fraud rate); it is simply swamped.

### 7.5 Ring candidates (`ring_patterns`)

Take the 2,500 highest-scored transactions; for each relation column, group them by value; for every pair within a group (first 40 members only) record the shared relation; keep pairs sharing ≥ 2 relations; score = (score_a + score_b) × number of shared relations. In this generator every relation is constant within a user, so two purchases by one customer share all 7 attributes and score ≈ 0.53 × 2 × 7 ≈ 7.4, which is what the deployed list shows: the top 30 all have `shared_relation_count: 7`. Reproduced: 100% of the top 10 and 55.5% of the top 200 are same-UID pairs.

A ring in the fraud sense is a *cycle through distinct accounts*: account A shares a card with B, B shares an address with C, C shares a device with A. The demo scenario hand-builds exactly this motif; the main pipeline neither generates it nor looks for it. The right primitive is the entity-projected multigraph over accounts (edge = shared card / device / address, excluding attribute-type columns and excluding same-account pairs), then connected components or short cycles ranked by fraud mass.

---

## 8. Dashboard views, one at a time

Static mode reads `main.json` (18.9 MB: 184 precomputed neighbourhoods, 1,600 transaction details of which 61 are fraud, 195 timelines) and `demo.json`.

| Panel | What the snapshot actually shows | Upstream cause |
|---|---|---|
| Force graph (default, main) | One address square with ~180 transaction circles, all amber (0.42–0.55), header "0 high-risk" | 7.2 (single seed, degree filter), 4.2 (score scale), 1.5 (no structure) |
| Header counter "high-risk" | Always 0 on main | `risk_score ≥ 0.65` never occurs (4.2) |
| "Why Is This Risky?" waterfall | Bars of ±0.03 logits labelled `graph_emb_05`, `ProductCD`, `TransactionDT`, `graph_emb_15`… ; "final probability 32.9%" for a benign row | 3 (unnamed latent features), 4.2 (all probabilities ≈ 0.33–0.43), 1.4 (nothing to explain) |
| Velocity timeline (UID) | 1–3 points for most accounts, one bar per day, no bursts | 1.3 (uniform independent timestamps, Pareto tails) |
| Embedding space snapshot | 2,400 identically coloured dots forming bands | 3 (PCA of one-hot missingness), plus the scatter uses a single fill colour and ignores `risk_score`/`isFraud` |
| Star patterns | `id_30: __MISSING__`, `DeviceInfo: __MISSING__`, … each at "risk 42.6%" | 7.4 (degree × constant) |
| Ring candidates | "TX a ↔ TX b, shared relations: 7, score 7.40" | 7.5 (same-customer pairs) |
| Search | 1,600 IDs and 27 entity strings | Export only covers what the single-seed overview reached |

**The demo dataset** (`demo_scenario.py`) is 50 hand-authored rows with hand-assigned scores, a device-farm star, an 8-node card/address ring, a household cluster, and 8 isolated rows. It is a good UX fixture and shows what the product is *for*. Three rigour problems to fix before a fraud scientist sees the code:

1. `DemoFraudApiService.explain_transaction` adds the contribution "Benign cluster context −0.18" when `isFraud == 0`. The explanation reads the ground-truth label.
2. The demo waterfall is in probability units, clipped to [0.01, 0.99] at each step, and scaled so that the sum equals the target shift; after clipping the steps no longer sum. Deployed example: 0.05 → 0.99 → 0.84 with a stated final probability of 0.92.
3. The demo returns `backend: "demo_heuristic_explainer"`, but the panel presents it identically to the model's SHAP output. Label it as scripted.

---

## 9. Symptom → cause → size of fix

| Symptom | Cause | Fundamental? | Fix size |
|---|---|---|---|
| Model looks useless (PR-AUC 0.07) | Label mechanism is near-noise; oracle ceiling 0.088 | Yes, data layer | Generator rewrite: label from planted mechanisms with a target ceiling (ROC ≈ 0.9) |
| Graph features add nothing | No cross-account reuse in generator; embedding is one-hot SVD | Yes, data layer + feature layer | Generator rewrite; replace SVD with in-time neighbour aggregates |
| All scores ≈ 0.43, nothing red | Asymmetric focal objective shifts the constant optimum to 0.355 | No | Use log-loss, or keep focal and calibrate on validation; keep UI thresholds |
| Stars = `__MISSING__` hubs | degree × mean-risk; missing not excluded from patterns | No | Lift-with-shrinkage score; exclude missing and attribute-type columns |
| Rings = same-customer pairs | Pairwise shared-attribute count over user-constant columns | Partly (needs generator to contain rings) | Account-projected components / cycles across distinct accounts |
| Overview is one address hub | `seed_count = 1`; degree filter removes 5/7 entity types | No | Seed from top-k components by fraud mass |
| Waterfall shows `graph_emb_05 ±0.03` | Unnamed latent features on a noise label | Follows from data fix | Named graph features ("distinct cards on this device, 7 d") explain themselves |
| Timeline has 1–3 points | Uniform independent timestamps | Yes, data layer | Generator: sessions, seasonality, bursts for fraud |
| Embedding scatter is one colour | Frontend ignores score/label | No | Colour by label and calibrated score |
| Demo explainer reads the label | `isFraud` used in heuristic | No | Remove; label the demo explainer as scripted |
| Identity-noise experiment is a null result | UID blend contributes nothing on this label | Follows from data fix | Run after generator rewrite; reconstruct UID à la Deotte from noisy card1/addr1/D1 |

---

## 10. What the foundation needs, and how the three planned upgrades fit

The plan ("bounded rigor pass, not a rebuild") is right about the product surface and wrong about where the pass starts. The API, snapshot exporter, frontend, and graph-store primitives are reusable. The generator, the objective's role, and the two pattern scores must change first, otherwise upgrades 1 and 2 produce true but empty results.

### 10.1 Generator v2: plant the economics, then measure whether the graph recovers them

Everything below is a few hundred lines in `synthetic.py` and keeps the IEEE column schema, so the rest of the stack is untouched.

1. **Actors.** Legitimate customers (≈ 12,000) grouped into households of 1–4 that share `addr1` and, with probability ≈ 0.5, a `DeviceInfo`. Fraud rings (≈ 40–80) each own a pool: m stolen `card1`s, d devices, a drop `addr1`s, a few `P_emaildomain`s. Each ring runs k synthetic accounts; every account draws each attribute from the ring pool with reuse probability ρ ≈ 0.6–0.9, otherwise fresh. This is the amortisation mechanism from 1.5 and it is the *only* thing that makes "distinct cards per device" informative.
2. **Time.** Customers: Poisson with daily and weekly seasonality. Rings: a cluster process. Burst start times are Poisson; within a burst, inter-arrivals are minutes to hours; a ring's cards burn after a finite number of uses. This gives velocity features something to measure.
3. **Amounts and products.** Customers: the current per-user lognormal. Rings: card-testing micro-amounts (\$1–5) preceding cash-out amounts drawn from the upper tail, concentrated in resellable product codes (C, H in the real data, not W).
4. **Labels.** A transaction is fraud if its account belongs to a ring (with a small unreported fraction to mimic chargeback lag, say 10% flipped to 0), plus a small rate of account-takeover on legitimate accounts (ATO = a legit account transacting from a ring device). Choose ring count and ρ so that the oracle sits near ROC 0.90–0.95 and PR-AUC 0.5–0.7; report the oracle alongside every model.
5. **Missingness.** Identity table present for ≈ 25% of rows, with presence dependent on `ProductCD` (as in the real data) and higher for ring accounts using mobile devices. Missing is then a mechanism, and `__MISSING__` gets excluded from entity nodes explicitly.
6. **Identity.** Keep `uid_true` for evaluation only. Expose the observable `uid_proxy = card1 + addr1 + (day − D1)` exactly as the Kaggle UID, and inject the two noise processes that make entity resolution hard: card reissue (a fraction of accounts change `card1` mid-horizon, splitting them) and shared regions (merging). Upgrade 2 then measures something real: how far detection falls when the proxy is 5–15% wrong, and whether graph features (which do not need the UID) are more robust than UID aggregates (which do).

### 10.2 Feature and model layer

- Replace the SVD with **in-time neighbour aggregates**, which are what fraud teams mean by graph features: for each transaction, over a trailing window (1 h, 24 h, 7 d), the number of distinct *other* accounts seen on the same `card1`, `DeviceInfo`, `addr1`; the fraud rate among labelled transactions on those entities up to that time; the 2-hop count (accounts reachable through a shared device that also share an address). Computed causally, they leak neither labels nor the future. Keep the SVD as a second, weaker baseline if desired.
- Objective: train with log-loss, or keep the asymmetric focal objective for ranking and fit an isotonic calibrator on validation. Either way, `pred_score` shown in the UI must be a calibrated probability, and the report should show a reliability curve.
- Re-run the ablation of upgrade 1 as three nested models: transaction features only; + UID aggregates; + graph neighbour aggregates. Report PR-AUC, precision@k and recall@k at k ∈ {100, 500, 1,000} on the test split, dollars captured at k, and the oracle ceiling. That table is the "finding".
- Fix the feature leakage: compute C-style counts causally (up to the transaction's timestamp), and drop `TransactionDT` as a raw feature.

### 10.3 Graph layer

- Entity types: identity-like columns only (`card1`, `DeviceInfo` when specific, `addr1` when the generator makes it an address rather than a region). Email domain, OS, browser, and `__MISSING__` are attributes and never nodes.
- Star score: shrunken lift, not degree × mean.
- Ring detection: account-projected multigraph, connected components across distinct accounts, ranked by fraud mass and dollar exposure; show the cycle in the demo and the component in main.
- Overview: seed from the top-k components by fraud mass, honouring `max_transactions`.

### 10.4 Product surface (upgrade 3)

Colour by calibrated probability; a "Why is this risky?" default that lists the top named graph features for the selected component; colour the embedding scatter by label and score; label restraint on dense clusters; mark the demo explainer as scripted and remove its label dependence.

### 10.5 Order of work for the focused week

1. Generator v2 with an oracle-ceiling check (2 days).
2. Causal graph aggregates, log-loss or calibration, ablation table with queue metrics (1.5 days).
3. Identity-noise experiment on `uid_proxy` (0.5 day).
4. Pattern scores and overview seeding; re-export snapshots (1 day).
5. Cosmetics and the write-up (1 day).

---

## Appendix: reproducing the numbers

```
pip install -r requirements.txt
python scripts/diagnose_foundations.py --n-transactions 120000 --out outputs/diagnostics.json
```

Runs in about 20 seconds on 4 cores. If `data/train_transaction.csv` and `data/train_identity.csv` are present, reference profiles are loaded exactly as the build does; otherwise the script uses hand-coded approximations of the public IEEE-CIS marginals (3.5% fraud, `TransactionAmt` median 68.77, `card1` median 9,678 / std 4,901, `addr1` median 299 / std 101, `P_emaildomain` 39% gmail, identity columns 76–87% missing, and so on). The structural conclusions do not depend on the exact marginals; the deployed snapshot, built from the real profiles, shows the same overview shape (one `addr1` hub), the same star list (`__MISSING__` hubs), the same 7-shared-relation "rings", and the same 0.42–0.55 score band as the reproduction.

The script prints the ablation table and writes a JSON with: user activity statistics, label-component firing rates and fraud lifts, the distribution of the generator's true probabilities, per-relation entity degree statistics, the degree-filter survival count, SVD widths and explained variance, PCA axis correlations with missingness, the five model variants and the oracle with queue metrics, the identity-noise sweep, and the graph store's overview / star / ring outputs.
