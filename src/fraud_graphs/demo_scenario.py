from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class DemoScenarioBundle:
    transactions: pd.DataFrame
    seed_transaction_ids: List[int]


def _base_row(
    tx_id: int,
    *,
    is_fraud: int,
    pred_score: float,
    dt: int,
    amt: float,
    uid: str,
    product: str,
    card1: str,
    addr1: str,
    p_email: str,
    r_email: str,
    device: str,
    id30: str,
    id31: str,
    x: float,
    y: float,
) -> Dict[str, object]:
    return {
        "TransactionID": tx_id,
        "isFraud": int(is_fraud),
        "split": "demo",
        "TransactionDT": int(dt),
        "TransactionAmt": float(amt),
        "pred_raw": float(pred_score),
        "pred_uid_blended": float(pred_score),
        "pred_score": float(pred_score),
        "uid_clean": uid,
        "ProductCD": product,
        "card1": card1,
        "addr1": addr1,
        "P_emaildomain": p_email,
        "R_emaildomain": r_email,
        "DeviceInfo": device,
        "id_30": id30,
        "id_31": id31,
        "embed_x": float(x),
        "embed_y": float(y),
    }


def build_demo_scenario(random_state: int = 42) -> DemoScenarioBundle:
    """
    Build a compact, interpretable graph dataset with explicit motifs:
    - device farm star
    - shared-attribute ring
    - benign household cluster
    """
    rng = np.random.default_rng(random_state)
    rows: List[Dict[str, object]] = []
    tx_id = 9_000_000
    day = 10

    # 1) Device farm star (mostly fraud)
    star_seed_tx = tx_id
    for i in range(20):
        fraud = 1 if i < 16 else 0
        score = 0.92 - 0.015 * i if fraud else 0.22 + 0.02 * (i - 16)
        rows.append(
            _base_row(
                tx_id,
                is_fraud=fraud,
                pred_score=float(np.clip(score, 0.03, 0.99)),
                dt=(day * 86_400) + 35 * i,
                amt=float(19 + 6 * i + rng.normal(0, 2)),
                uid=f"uid_star_{i:03d}",
                product="W",
                card1=f"card_star_{i:03d}",
                addr1=f"addr_star_{i % 4}",
                p_email=f"throwaway{i:03d}.mail",
                r_email=f"throwaway{i:03d}.mail",
                device="DeviceFarm_Botnet_A",
                id30="Android 13",
                id31="mobile chrome 115",
                x=2.6 + rng.normal(0, 0.18),
                y=0.2 + rng.normal(0, 0.22),
            )
        )
        tx_id += 1

    # 2) Ring motif (attribute hand-off around the cycle)
    ring_seed_tx = tx_id
    ring_pairs: List[Tuple[str, str]] = [
        ("card_ring_a", "addr_ring_f"),
        ("card_ring_a", "addr_ring_b"),
        ("card_ring_b", "addr_ring_b"),
        ("card_ring_b", "addr_ring_c"),
        ("card_ring_c", "addr_ring_c"),
        ("card_ring_c", "addr_ring_d"),
        ("card_ring_d", "addr_ring_d"),
        ("card_ring_d", "addr_ring_f"),
    ]
    for i, (card, addr) in enumerate(ring_pairs):
        fraud = 1 if i in {0, 1, 4, 6} else 0
        score = 0.83 if fraud else 0.36
        rows.append(
            _base_row(
                tx_id,
                is_fraud=fraud,
                pred_score=score,
                dt=((day + 1) * 86_400) + i * 190,
                amt=float(95 + 18 * i + rng.normal(0, 4)),
                uid=f"uid_ring_{i:03d}",
                product="R",
                card1=card,
                addr1=addr,
                p_email=f"ring_{i % 4}@mail.net",
                r_email=f"ring_{(i + 1) % 4}@mail.net",
                device=f"RingPhone_{i % 3}",
                id30=f"iOS 17.{i % 3}",
                id31="mobile safari 16",
                x=-2.5 + np.cos(i * (2 * np.pi / len(ring_pairs))) * 1.15,
                y=0.0 + np.sin(i * (2 * np.pi / len(ring_pairs))) * 1.15,
            )
        )
        tx_id += 1

    # 3) Benign household cluster
    home_seed_tx = tx_id
    for i in range(14):
        rows.append(
            _base_row(
                tx_id,
                is_fraud=0,
                pred_score=float(0.07 + 0.02 * rng.random()),
                dt=((day + 2) * 86_400) + i * 640,
                amt=float(40 + 8 * (i % 4) + rng.normal(0, 3)),
                uid=f"uid_home_{i // 3:03d}",
                product="H",
                card1=f"card_home_{i // 4}",
                addr1="addr_home_001",
                p_email="family.home.com",
                r_email="family.home.com",
                device=f"HomeLaptop_{i % 2}",
                id30="Windows 11",
                id31="chrome 118",
                x=-0.4 + rng.normal(0, 0.4),
                y=2.3 + rng.normal(0, 0.35),
            )
        )
        tx_id += 1

    # 4) A few isolated baseline transactions
    for i in range(8):
        rows.append(
            _base_row(
                tx_id,
                is_fraud=0,
                pred_score=float(0.04 + 0.03 * rng.random()),
                dt=((day + 3) * 86_400) + i * 1300,
                amt=float(30 + rng.normal(0, 6)),
                uid=f"uid_iso_{i:03d}",
                product="S",
                card1=f"card_iso_{i:03d}",
                addr1=f"addr_iso_{i:03d}",
                p_email=f"iso_{i}@mail.org",
                r_email=f"iso_{i}@mail.org",
                device=f"IsoDevice_{i:02d}",
                id30="Windows 10",
                id31="firefox 115",
                x=0.5 + rng.normal(0, 0.45),
                y=-2.2 + rng.normal(0, 0.35),
            )
        )
        tx_id += 1

    df = pd.DataFrame(rows).sort_values("TransactionDT").reset_index(drop=True)
    return DemoScenarioBundle(
        transactions=df,
        seed_transaction_ids=[star_seed_tx, ring_seed_tx, home_seed_tx],
    )

