"""
dataset.py - seeded, deterministic order generator for the Nykaa support agent.

Design choices (also reproduced at the top of README.md):
  SEED            = 42
  N_RECORDS       = 60
  CATEGORY WEIGHTS: Beauty 0.30, Apparel 0.25, Footwear 0.15, Home 0.15, Electronics 0.15
  STATUS WEIGHTS:   Delivered 0.40, Shipped 0.20, Placed 0.15, Returned 0.15, Refunded 0.10
  P(delayed_shipment) = 0.18 (base rate, applied independently per record)
  AMOUNT RANGE (INR, per category - see PRICE_RANGES), overall Rs 199 - Rs 14,999.

Price reasoning (one sentence): ranges mirror typical Nykaa catalogue price points
(Beauty is cheap and high-volume, Electronics such as trimmers/hair dryers are the
priciest), so order values look like real basket sizes rather than a flat uniform draw.

If the delayed-shipment percentage falls outside 10-30%, change SEED (or the weights)
and regenerate. Individual records are NEVER hand-edited.
"""
import random
from collections import Counter

SEED = 42
N_RECORDS = 60

CATEGORIES = ["Apparel", "Electronics", "Home", "Footwear", "Beauty"]
STATUSES = ["Placed", "Shipped", "Delivered", "Returned", "Refunded"]

CATEGORY_WEIGHTS = {"Beauty": 0.30, "Apparel": 0.25, "Footwear": 0.15, "Home": 0.15, "Electronics": 0.15}
STATUS_WEIGHTS = {"Delivered": 0.40, "Shipped": 0.20, "Placed": 0.15, "Returned": 0.15, "Refunded": 0.10}

# Realistic INR price band per category (min, max)
PRICE_RANGES = {
    "Beauty": (199, 2999),
    "Apparel": (499, 3999),
    "Footwear": (799, 4999),
    "Home": (299, 5999),
    "Electronics": (999, 14999),
}

# Realistic age (days since order created) band per status, all inside 0-30
DAYS_RANGES = {
    "Placed": (0, 4),
    "Shipped": (1, 10),
    "Delivered": (3, 30),
    "Returned": (7, 30),
    "Refunded": (10, 30),
}

P_DELAYED = 0.18


def generate_orders(seed: int = SEED, n: int = N_RECORDS) -> list[dict]:
    rng = random.Random(seed)  # isolated RNG -> fully deterministic
    cats, cat_w = zip(*CATEGORY_WEIGHTS.items())
    stats, stat_w = zip(*STATUS_WEIGHTS.items())
    orders = []
    for i in range(1, n + 1):
        category = rng.choices(cats, weights=cat_w, k=1)[0]
        status = rng.choices(stats, weights=stat_w, k=1)[0]
        lo, hi = PRICE_RANGES[category]
        dlo, dhi = DAYS_RANGES[status]
        orders.append({
            "record_id": f"NYK-{1000 + i}",
            "category": category,
            "status": status,
            "order_value_inr": rng.randint(lo, hi),
            "days_since_created": rng.randint(dlo, dhi),
            "delayed_shipment": rng.random() < P_DELAYED,
        })
    return orders


def validate(orders: list[dict]) -> dict:
    """Assert every structural threshold from the brief; return the report."""
    assert len(orders) >= 40, "need >= 40 records"
    by_cat = Counter(o["category"] for o in orders)
    by_status = Counter(o["status"] for o in orders)
    for c in CATEGORIES:
        assert by_cat[c] >= 3, f"category {c} has only {by_cat[c]} records (need >= 3)"
    for s in STATUSES:
        assert by_status[s] >= 1, f"status {s} missing"
    for o in orders:
        assert 0 <= o["days_since_created"] <= 30
    pct = 100 * sum(o["delayed_shipment"] for o in orders) / len(orders)
    assert 10 <= pct <= 30, f"delayed_shipment {pct:.1f}% outside 10-30% band - change SEED"
    return {"n": len(orders), "by_category": dict(by_cat), "by_status": dict(by_status), "delayed_pct": pct}


ORDERS = generate_orders()
ORDERS_BY_ID = {o["record_id"]: o for o in ORDERS}

if __name__ == "__main__":
    report = validate(ORDERS)
    print(f"Seed={SEED}  Records={report['n']}")
    print("\nCount per category:")
    for c in CATEGORIES:
        print(f"  {c:<12}{report['by_category'][c]}")
    print("\nCount per status:")
    for s in STATUSES:
        print(f"  {s:<12}{report['by_status'][s]}")
    delayed = sum(o["delayed_shipment"] for o in ORDERS)
    print(f"\ndelayed_shipment=True: {delayed}/{report['n']} = {report['delayed_pct']:.1f}%  (band 10-30%: OK)")
    print("\nFirst 5 records:")
    for o in ORDERS[:5]:
        print(" ", o)