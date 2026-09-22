"""
Generator v2: an actor-level synthetic IEEE-CIS-style dataset with planted fraud economics.

What is planted (see docs/foundational_flow_review.md, section 10.1):
  * Legitimate customers grouped into households that share an address and often a device.
  * Fraud rings that own a pool of stolen cards, devices, drop addresses and email domains and
    run many synthetic accounts drawing from that pool with a reuse probability (asset
    amortisation), acting in short bursts with card-testing micro-amounts before cash-out,
    on resellable product lines, with cards that burn after a few uses.
  * Account takeover: a legitimate account transacting from a ring device.
  * Identity-table presence that depends on product code (as in the real data).
  * A hidden true account id (uid_true) and the observable Kaggle-style proxy
    uid_proxy = card1 + addr1 + (day - D1), degraded by card reissue.
  * Labels with an unreported fraction (chargeback lag).

The output keeps the v1 column schema (TRANSACTION_COLUMNS / IDENTITY_COLUMNS) so the rest of
the stack is unchanged, and adds diagnostic columns that must never be used as features:
uid_true, actor_type, ring_id, true_p.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .synthetic import IDENTITY_COLUMNS, TRANSACTION_COLUMNS

DIAGNOSTIC_COLUMNS = ["uid_true", "uid_proxy", "actor_type", "ring_id", "true_p"]

PRODUCTS = np.array(["W", "C", "R", "H", "S"], dtype=object)
LEGIT_PRODUCT_P = np.array([0.745, 0.116, 0.064, 0.056, 0.019])
RING_PRODUCT_P = np.array([0.50, 0.22, 0.09, 0.15, 0.04])
# Identity table presence by product (real data: identity rows are mostly C/H/R/S, rarely W).
IDENTITY_PRESENCE = {"W": 0.04, "C": 0.92, "R": 0.62, "H": 0.75, "S": 0.80}

DEVICE_MODELS = [
    ("Windows", "desktop", "Windows 10", "chrome 63.0", 0.34),
    ("Windows", "desktop", "Windows 7", "ie 11.0 for desktop", 0.10),
    ("MacOS", "desktop", "Mac OS X 10_13_6", "safari generic", 0.09),
    ("iOS Device", "mobile", "iOS 11.2.1", "mobile safari 11.0", 0.20),
    ("SM-G960U Build/R16NW", "mobile", "Android 8.0.0", "chrome 65.0 for android", 0.07),
    ("SM-G930V Build/NRD90M", "mobile", "Android 7.0", "chrome 63.0 for android", 0.05),
    ("Moto G (5) Build/NPPS25.137-93-14", "mobile", "Android 7.0", "chrome 64.0 for android", 0.04),
    ("Trident/7.0", "desktop", "Windows 8.1", "ie 11.0 for desktop", 0.05),
    ("rv:57.0", "desktop", "Linux", "firefox 57.0", 0.03),
    ("Pixel 2 Build/OPM1.171019.021", "mobile", "Android 8.1.0", "chrome 66.0 for android", 0.03),
]
LEGIT_EMAIL = (["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "aol.com", "comcast.net", "icloud.com", "outlook.com", "msn.com", "att.net", "live.com", "protonmail.com", "mail.com"],
               [0.42, 0.18, 0.08, 0.07, 0.05, 0.04, 0.04, 0.04, 0.03, 0.02, 0.02, 0.006, 0.004])
RING_EMAIL = (["gmail.com", "anonymous.com", "outlook.com", "yahoo.com", "hotmail.com", "protonmail.com", "mail.com"],
              [0.50, 0.14, 0.10, 0.12, 0.07, 0.04, 0.03])
CARD4 = (["visa", "mastercard", "american express", "discover"], [0.653, 0.320, 0.015, 0.012])
CARD6 = (["debit", "credit"], [0.75, 0.25])
HOUR_PROFILE = np.array([0.30, 0.20, 0.15, 0.12, 0.12, 0.18, 0.35, 0.60, 0.85, 1.05, 1.15, 1.20,
                         1.20, 1.15, 1.10, 1.10, 1.15, 1.20, 1.30, 1.35, 1.30, 1.15, 0.85, 0.55])
WEEKDAY_PROFILE = np.array([1.0, 1.05, 1.05, 1.05, 1.0, 0.85, 0.80])


@dataclass
class SyntheticV2Config:
    n_transactions: int = 120_000
    horizon_days: int = 180
    target_fraud_rate: float = 0.035
    random_state: int = 42
    # legitimate population
    tx_per_customer: float = 12.0
    household_size_p: Tuple[float, ...] = (0.55, 0.25, 0.13, 0.07)
    household_device_share_p: float = 0.5
    household_card_share_p: float = 0.35   # multi-person households that share one card (benign card sharing)
    card_reissue_rate: float = 0.06        # accounts whose card1 changes mid-horizon (splits uid_proxy)
    # rings
    n_rings: int = 90
    ring_accounts_range: Tuple[int, int] = (6, 25)
    ring_cards_range: Tuple[int, int] = (4, 15)
    ring_devices_range: Tuple[int, int] = (2, 6)
    ring_addresses_range: Tuple[int, int] = (1, 4)
    ring_reuse_p_range: Tuple[float, float] = (0.6, 0.9)
    ring_lifetime_days_range: Tuple[int, int] = (7, 30)
    burst_size_range: Tuple[int, int] = (2, 12)
    burst_interarrival_minutes: float = 60.0     # median gap inside a cash-out burst
    low_and_slow_ring_share: float = 0.45        # rings that spread activity over days instead of bursting
    low_and_slow_multiplier: float = 6.0
    card_burn_uses_range: Tuple[int, int] = (5, 25)
    card_testing_p: float = 0.35           # bursts that open with 1-3 micro-amount probes
    # account takeover of legitimate accounts
    ato_account_rate: float = 0.015
    # labels
    unreported_fraud_rate: float = 0.08
    first_party_fraud_share: float = 0.20   # share of fraud that is chargeback abuse on ordinary accounts (structurally invisible)
    # identity noise
    identity_missing_extra: float = 0.0


@dataclass
class _Account:
    uid: int
    actor: str                 # "legit" | "ring"
    ring_id: int
    household: int
    card1: int
    card1_reissue: int         # -1 if none
    reissue_day: int
    card2: int
    card3: int
    card4: str
    card5: int
    card6: str
    addr1: int
    addr2: int
    p_email: str
    r_email: str
    devices: List[int]         # device fingerprint ids
    birth_day: int
    amt_mu: float
    amt_sigma: float
    product_p: np.ndarray
    ato_day: int = -1
    ato_device: int = -1


def _choice(rng: np.random.Generator, pairs: Tuple[List[str], List[float]], size: int) -> np.ndarray:
    vals, p = pairs
    p = np.asarray(p, dtype=float)
    return rng.choice(np.array(vals, dtype=object), size=size, p=p / p.sum())


def _draw_time(rng: np.random.Generator, n: int, day_lo: int, day_hi: int) -> np.ndarray:
    days = np.arange(day_lo, day_hi)
    wday = WEEKDAY_PROFILE[days % 7]
    day = rng.choice(days, size=n, p=wday / wday.sum())
    hour = rng.choice(24, size=n, p=HOUR_PROFILE / HOUR_PROFILE.sum())
    return day * 86_400 + hour * 3600 + rng.integers(0, 3600, size=n)


def _build_population(cfg: SyntheticV2Config, rng: np.random.Generator):
    n_customers = max(2_000, int(cfg.n_transactions / cfg.tx_per_customer))
    card_pool = rng.permutation(np.arange(1_000, 1_000 + 12 * n_customers))
    addr_pool = rng.permutation(np.arange(100, 100 + 6 * n_customers))
    device_counter = [0]
    card_ptr = [0]
    addr_ptr = [0]

    def new_card() -> int:
        card_ptr[0] += 1
        return int(card_pool[card_ptr[0] - 1])

    def new_addr() -> int:
        addr_ptr[0] += 1
        return int(addr_pool[addr_ptr[0] - 1])

    def new_device() -> int:
        device_counter[0] += 1
        return device_counter[0]

    device_model: Dict[int, int] = {}
    model_p = np.array([m[4] for m in DEVICE_MODELS])

    def device_with_model() -> int:
        d = new_device()
        device_model[d] = int(rng.choice(len(DEVICE_MODELS), p=model_p / model_p.sum()))
        return d

    accounts: List[_Account] = []
    uid = 0
    # --- households of legitimate customers ---
    households = 0
    while len(accounts) < n_customers:
        size = int(rng.choice(len(cfg.household_size_p), p=np.array(cfg.household_size_p) / sum(cfg.household_size_p))) + 1
        addr = new_addr()
        shared_dev = device_with_model() if rng.random() < cfg.household_device_share_p else -1
        shared_card = new_card() if (size > 1 and rng.random() < cfg.household_card_share_p) else -1
        p_email_hh = str(_choice(rng, LEGIT_EMAIL, 1)[0])
        for member in range(size):
            devs = [device_with_model()]
            if shared_dev >= 0 and rng.random() < 0.8:
                devs.append(shared_dev)
            p_email = p_email_hh if rng.random() < 0.3 else str(_choice(rng, LEGIT_EMAIL, 1)[0])
            reissue = rng.random() < cfg.card_reissue_rate
            card = shared_card if (shared_card >= 0 and (member == 0 or rng.random() < 0.7)) else new_card()
            accounts.append(_Account(
                uid=uid, actor="legit", ring_id=-1, household=households,
                card1=card, card1_reissue=new_card() if reissue else -1,
                reissue_day=int(rng.integers(cfg.horizon_days // 4, 3 * cfg.horizon_days // 4)) if reissue else -1,
                card2=int(np.clip(rng.normal(361, 157), 100, 600)), card3=150,
                card4=str(_choice(rng, CARD4, 1)[0]), card5=int(np.clip(rng.normal(226, 41), 100, 237)),
                card6=str(_choice(rng, CARD6, 1)[0]), addr1=addr, addr2=87,
                p_email=p_email, r_email=p_email if rng.random() < 0.85 else str(_choice(rng, LEGIT_EMAIL, 1)[0]),
                devices=devs,
                birth_day=int(rng.integers(-1300, -20)) if rng.random() < 0.75 else int(rng.integers(-20, cfg.horizon_days - 10)),
                amt_mu=float(rng.normal(np.log(60.0), 0.45)), amt_sigma=float(np.clip(rng.normal(0.55, 0.2), 0.15, 1.2)),
                product_p=LEGIT_PRODUCT_P,
            ))
            uid += 1
        households += 1
    legit_accounts = list(accounts)

    # --- fraud rings ---
    rings: List[Dict[str, object]] = []
    for r in range(cfg.n_rings):
        cards = [new_card() for _ in range(int(rng.integers(*cfg.ring_cards_range)))]
        devices = [device_with_model() for _ in range(int(rng.integers(*cfg.ring_devices_range)))]
        addrs = [new_addr() for _ in range(int(rng.integers(*cfg.ring_addresses_range)))]
        emails = list(_choice(rng, RING_EMAIL, int(rng.integers(1, 4))))
        reuse = float(rng.uniform(*cfg.ring_reuse_p_range))
        start = int(rng.integers(0, cfg.horizon_days - 10))
        life = int(rng.integers(*cfg.ring_lifetime_days_range))
        n_acc = int(rng.integers(*cfg.ring_accounts_range))
        ring_uids = []
        for _ in range(n_acc):
            card = int(rng.choice(cards)) if rng.random() < reuse else new_card()
            dev = [int(rng.choice(devices)) if rng.random() < reuse else device_with_model()]
            addr = int(rng.choice(addrs)) if rng.random() < reuse else new_addr()
            email = str(rng.choice(np.array(emails, dtype=object))) if rng.random() < reuse else str(_choice(rng, RING_EMAIL, 1)[0])
            accounts.append(_Account(
                uid=uid, actor="ring", ring_id=r, household=-1,
                card1=card, card1_reissue=-1, reissue_day=-1,
                card2=int(np.clip(rng.normal(361, 157), 100, 600)), card3=150,
                card4=str(_choice(rng, CARD4, 1)[0]), card5=int(np.clip(rng.normal(226, 41), 100, 237)),
                card6="credit" if rng.random() < 0.4 else "debit", addr1=addr, addr2=87,
                p_email=email, r_email=email if rng.random() < 0.5 else str(_choice(rng, RING_EMAIL, 1)[0]),
                devices=dev,
                # half the ring accounts are freshly created, half are aged (bought) accounts
                birth_day=(-int(rng.integers(1, 30)) - start) if rng.random() < 0.5 else int(rng.integers(-1300, -20)),
                amt_mu=float(rng.normal(np.log(105.0), 0.5)), amt_sigma=0.65,
                product_p=RING_PRODUCT_P,
            ))
            ring_uids.append(uid)
            uid += 1
        rings.append({"id": r, "uids": ring_uids, "cards": cards, "devices": devices, "start": start, "end": min(cfg.horizon_days, start + life),
                      "slow": bool(rng.random() < cfg.low_and_slow_ring_share),
                      "card_uses_left": {c: int(rng.integers(*cfg.card_burn_uses_range)) for c in cards}})

    # --- account takeover victims ---
    n_ato = int(cfg.ato_account_rate * len(legit_accounts))
    for a in rng.choice(len(legit_accounts), size=n_ato, replace=False):
        ring = rings[int(rng.integers(len(rings)))]
        legit_accounts[a].ato_day = int(rng.integers(ring["start"], max(ring["start"] + 1, ring["end"])))
        legit_accounts[a].ato_device = int(rng.choice(ring["devices"]))
    return accounts, rings, device_model


def generate_synthetic_v2(cfg: SyntheticV2Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(cfg.random_state)
    accounts, rings, device_model = _build_population(cfg, rng)
    by_uid = {a.uid: a for a in accounts}
    legit = [a for a in accounts if a.actor == "legit"]

    # Fraud volume so that the *reported* fraud rate lands near target.
    n_fraud_target = int(cfg.target_fraud_rate * cfg.n_transactions)
    n_first_party = int(cfg.first_party_fraud_share * n_fraud_target)
    n_actor_fraud = int((n_fraud_target - n_first_party) / (1.0 - cfg.unreported_fraud_rate))
    n_ato_tx_target = int(0.12 * n_actor_fraud)
    n_ring_tx_target = n_actor_fraud - n_ato_tx_target
    n_legit_tx = cfg.n_transactions - n_actor_fraud

    rows: List[Dict[str, object]] = []

    # ---------------- legitimate activity: heterogeneous rates, seasonality, baskets ----------------
    rates = rng.lognormal(0.0, 1.1, size=len(legit))          # heavy-tailed customer activity (some people shop a lot)
    counts = rng.multinomial(n_legit_tx, rates / rates.sum())
    for a, n in zip(legit, counts):
        if n == 0:
            continue
        t = _draw_time(rng, n, max(0, a.birth_day), cfg.horizon_days)
        basket = rng.random(n) < 0.35
        t = np.where(basket, t + rng.integers(60, 7200, size=n), t)   # follow-up purchases within two hours (sessions, baskets)
        for ti in np.sort(t):
            day = int(ti // 86_400)
            card = a.card1_reissue if (a.card1_reissue >= 0 and day >= a.reissue_day) else a.card1
            rows.append(dict(uid_true=a.uid, actor_type="legit", ring_id=-1, dt=int(ti), card1=card, addr1=a.addr1,
                             device=int(rng.choice(a.devices)), product=str(rng.choice(PRODUCTS, p=a.product_p)),
                             amt=float(np.exp(rng.normal(a.amt_mu, a.amt_sigma))), fraud=0, acc=a))

    # ---------------- ring bursts: card testing then cash-out, cards burn ----------------
    n_ring_tx = 0
    ring_order = rng.permutation(len(rings))
    guard = 0
    while n_ring_tx < n_ring_tx_target and guard < 50_000:
        guard += 1
        ring = rings[int(ring_order[guard % len(rings)])]
        live_cards = [c for c, u in ring["card_uses_left"].items() if u > 0]
        if not live_cards:
            ring["card_uses_left"] = {c: int(rng.integers(*cfg.card_burn_uses_range)) for c in ring["cards"]}  # ring sources fresh dumps
            live_cards = ring["cards"]
        size = int(rng.integers(*cfg.burst_size_range))
        start = int(_draw_time(rng, 1, ring["start"], ring["end"])[0])
        gap_median = cfg.burst_interarrival_minutes * 60.0 * (cfg.low_and_slow_multiplier if ring["slow"] else 1.0)
        gaps = rng.lognormal(np.log(gap_median), 0.9, size=size)
        times = start + np.cumsum(gaps).astype(int)
        n_probe = int(rng.integers(1, 4)) if rng.random() < cfg.card_testing_p else 0
        uids = rng.choice(ring["uids"], size=size)
        for j, (ti, u) in enumerate(zip(times, uids)):
            a = by_uid[int(u)]
            card = a.card1
            if card in ring["card_uses_left"]:
                if ring["card_uses_left"][card] <= 0:
                    live = [c for c, k in ring["card_uses_left"].items() if k > 0]
                    if not live:
                        break
                    card = int(rng.choice(live))
                ring["card_uses_left"][card] -= 1
            amt = float(rng.uniform(1.0, 5.0)) if j < n_probe else float(np.exp(rng.normal(a.amt_mu, a.amt_sigma)))
            rows.append(dict(uid_true=a.uid, actor_type="ring", ring_id=ring["id"], dt=int(ti), card1=card, addr1=a.addr1,
                             device=int(rng.choice(a.devices)), product=str(rng.choice(PRODUCTS, p=a.product_p)),
                             amt=amt, fraud=1, acc=a))
            n_ring_tx += 1

    # ---------------- account takeover: legitimate card, ring device, short high-value burst ----------------
    victims = [a for a in legit if a.ato_day >= 0]
    n_ato_tx = 0
    for a in victims:
        if n_ato_tx >= n_ato_tx_target:
            break
        size = int(rng.integers(2, 7))
        t0 = a.ato_day * 86_400 + int(rng.integers(0, 86_400))
        times = t0 + np.cumsum(rng.exponential(600.0, size=size)).astype(int)
        for ti in times:
            rows.append(dict(uid_true=a.uid, actor_type="ato", ring_id=-1, dt=int(ti), card1=a.card1, addr1=a.addr1,
                             device=a.ato_device, product=str(rng.choice(PRODUCTS, p=RING_PRODUCT_P)),
                             amt=float(np.exp(rng.normal(a.amt_mu + 0.8, 0.5))), fraud=1, acc=a))
            n_ato_tx += 1

    df = pd.DataFrame([{k: v for k, v in r.items() if k != "acc"} for r in rows])
    df = df[df["dt"] < cfg.horizon_days * 86_400].copy()
    df = df.sort_values("dt", kind="mergesort").reset_index(drop=True)
    n = len(df)

    # account-level attributes (vectorised through uid lookup)
    uid_arr = df["uid_true"].to_numpy()
    def per_uid(attr: str, dtype=object) -> np.ndarray:
        return np.array([getattr(by_uid[int(u)], attr) for u in uid_arr], dtype=dtype)

    day = (df["dt"].to_numpy() // 86_400).astype(int)
    birth = per_uid("birth_day", int)
    d1 = np.maximum(1, day - birth).astype(float)

    # ---------------- true label probability and reported label ----------------
    # Ring / ATO transactions are fraud with probability (1 - unreported). Ordinary transactions carry a
    # small first-party fraud probability that no feature can see (the Bayes ceiling is below 1).
    actor_fraud = df["fraud"].to_numpy() == 1
    n_legit_rows = int((~actor_fraud).sum())
    q_first_party = min(0.5, n_first_party / max(1, n_legit_rows))
    true_p = np.where(actor_fraud, 1.0 - cfg.unreported_fraud_rate, q_first_party)
    is_fraud = actor_fraud.astype(np.int8)
    unreported = actor_fraud & (rng.random(n) < cfg.unreported_fraud_rate)
    is_fraud = np.where(unreported, 0, is_fraud).astype(np.int8)
    first_party = (~actor_fraud) & (rng.random(n) < q_first_party)
    is_fraud = np.where(first_party, 1, is_fraud).astype(np.int8)
    actor_col = np.where(first_party, "first_party", df["actor_type"].to_numpy().astype(object))

    # ---------------- identity table presence ----------------
    product = df["product"].to_numpy().astype(object)
    presence_p = np.array([IDENTITY_PRESENCE[p] for p in product]) * (1.0 - cfg.identity_missing_extra)
    has_identity = rng.random(n) < presence_p
    dev_ids = df["device"].to_numpy()
    models = np.array([DEVICE_MODELS[device_model[int(d)]] for d in dev_ids], dtype=object)
    device_info = np.where(has_identity, np.array([f"{m[0]}|fp{int(d):06d}" for m, d in zip(models, dev_ids)], dtype=object), None)
    device_type = np.where(has_identity, models[:, 1], None)
    id_30 = np.where(has_identity, models[:, 2], None)
    id_31 = np.where(has_identity, models[:, 3], None)

    # ---------------- causal C columns (counts up to and including this row, in time order) ----------------
    card1 = df["card1"].to_numpy()
    addr1 = df["addr1"].to_numpy()
    p_email = per_uid("p_email")
    r_email = per_uid("r_email")
    uid_proxy = np.array([f"{c}_{a}_{int(dd - d1v)}" for c, a, dd, d1v in zip(card1, addr1, day, d1)], dtype=object)

    def running_count(keys: np.ndarray) -> np.ndarray:
        return pd.Series(np.ones(n)).groupby(pd.Series(keys)).cumsum().to_numpy().astype(int)

    c1 = running_count(uid_proxy)
    c2 = running_count(card1)
    dev_key = np.where(has_identity, dev_ids, -1)
    c3 = np.where(has_identity, running_count(dev_key), 0)
    c4 = running_count(p_email)
    c5 = running_count(addr1)
    # distinct addresses / emails seen so far on this card (Vesta-style "how many X per card")
    def running_distinct(key: np.ndarray, val: np.ndarray) -> np.ndarray:
        tmp = pd.DataFrame({"k": key, "v": val, "i": np.arange(n)})
        first = ~tmp.duplicated(["k", "v"])
        return first.groupby(tmp["k"]).cumsum().to_numpy().astype(int)
    c6 = running_distinct(card1, addr1)
    c7 = running_distinct(card1, p_email)
    missing_seen = pd.Series(~has_identity).groupby(pd.Series(card1)).cummax().to_numpy().astype(int)
    c8 = running_distinct(card1, dev_key) - missing_seen          # distinct devices per card, ignoring "no device"
    c9 = np.where(has_identity, running_distinct(dev_key, card1), 0)
    c10 = running_distinct(addr1, card1)
    c_cols = {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5, "C6": c6, "C7": c7, "C8": c8, "C9": c9, "C10": c10}
    for i in range(11, 15):
        c_cols[f"C{i}"] = np.clip(0.5 * c1 + 0.3 * c2 + rng.normal(0, 2, size=n), 0, None).astype(int)

    # D columns: D1 account age; D2 days since previous transaction on this card; rest noisy copies
    prev_dt = pd.Series(df["dt"].to_numpy()).groupby(pd.Series(card1)).shift(1).to_numpy()
    d2 = np.where(np.isnan(prev_dt), np.nan, (df["dt"].to_numpy() - prev_dt) / 86_400.0)
    d_cols = {"D1": d1, "D2": d2}
    for i in range(3, 16):
        d_cols[f"D{i}"] = np.maximum(0, d1 - rng.integers(0, 15 * i, size=n)).astype(float)

    actor = actor_col
    m_true_p = np.where(np.isin(actor, ["legit", "first_party"]), 0.90, 0.78)
    m_cols = {f"M{i}": np.where(rng.random(n) < 0.3, None, np.where(rng.random(n) < m_true_p, "T", "F")) for i in range(1, 10)}

    tx_id = np.arange(2_000_000, 2_000_000 + n, dtype=np.int64)
    transaction_df = pd.DataFrame({
        "TransactionID": tx_id, "isFraud": is_fraud, "TransactionDT": df["dt"].to_numpy().astype(np.int64),
        "TransactionAmt": np.clip(df["amt"].to_numpy(), 1.0, 40_000.0),
        "ProductCD": product, "card1": card1, "card2": per_uid("card2", int), "card3": per_uid("card3", int),
        "card4": per_uid("card4"), "card5": per_uid("card5", int), "card6": per_uid("card6"),
        "addr1": addr1, "addr2": per_uid("addr2", int),
        # dist1: 60% missing as in the real data; fraud has a heavier tail but mostly overlaps
        "dist1": np.where(rng.random(n) < 0.6, np.nan, np.where(np.isin(actor, ["legit", "first_party"]) | (rng.random(n) < 0.6), np.abs(rng.normal(15, 14, size=n)), np.abs(rng.normal(70, 60, size=n)))),
        "dist2": np.where(rng.random(n) < 0.93, np.nan, np.abs(rng.normal(18, 8, size=n))),
        "P_emaildomain": p_email, "R_emaildomain": np.where(rng.random(n) < 0.77, None, r_email),
        **c_cols, **d_cols, **m_cols,
        "uid_clean": uid_proxy,      # what v1 called uid_clean is now the observable proxy
        "uid_true": np.array([f"uid_{int(u):06d}" for u in uid_arr], dtype=object),
        "uid_proxy": uid_proxy, "actor_type": actor_col, "ring_id": df["ring_id"].to_numpy(), "true_p": true_p,
    })

    id_data: Dict[str, object] = {"TransactionID": tx_id, "DeviceType": device_type, "DeviceInfo": device_info}
    risk = np.where(np.isin(actor, ["legit", "first_party"]), 0.0, 1.0)
    for i in range(1, 39):
        col = f"id_{i:02d}"
        if col == "id_30":
            id_data[col] = id_30
        elif col == "id_31":
            id_data[col] = id_31
        elif col == "id_33":
            id_data[col] = np.where(has_identity, np.where(models[:, 1] == "mobile", "1334x750", "1920x1080"), None)
        elif col in {"id_36", "id_37", "id_38"}:
            id_data[col] = np.where(has_identity, np.where(rng.random(n) < 0.6, "T", "F"), None)
        elif col in {"id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_35"}:
            id_data[col] = np.where(has_identity, np.where(rng.random(n) < 0.8, "Found", "NotFound"), None)
        else:
            id_data[col] = np.where(has_identity, rng.normal(0.0, 1.0, size=n) + 0.05 * risk, np.nan)
    identity_df = pd.DataFrame(id_data)
    # the identity table only has rows for transactions with identity data (as in the real data)
    identity_df = identity_df[has_identity].reset_index(drop=True)

    transaction_df = transaction_df[TRANSACTION_COLUMNS + DIAGNOSTIC_COLUMNS]
    identity_df = identity_df[IDENTITY_COLUMNS]
    return transaction_df, identity_df
