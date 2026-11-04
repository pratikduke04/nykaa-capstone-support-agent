"""
Task 6 - check_order_status tool with a DESIGNED escalation score.

FORMULA (all constants below):
    open_flag        = 1 if status in {Placed, Shipped, Returned} else 0
                       (Delivered / Refunded orders are resolved -> nothing to escalate)
    recency_norm     = days_since_created / 30          (dataset range is 0-30, so result is in [0, 1])
    escalation_score = open_flag * ( 0.5 * delayed_shipment + 0.5 * recency_norm )   -> always in [0, 1]
    escalate if escalation_score >= 0.40

WHY THESE NUMBERS (checked against the generated dataset, see __main__ output):
  * A delayed open order scores >= 0.50, so it always escalates (a late shipment is the strongest signal).
  * A NON-delayed open order needs recency_norm >= 0.80, i.e. >= 24 days old, to escalate.
    In this dataset only 1 of 24 open orders is that old, so ordinary in-flight orders are not flagged.
  * Result: 4 of 24 open orders (17%) escalate - just under the top-20% (80th-percentile) cut,
    whose score is ~0.30 in this data. Both signals matter; it is not a bare boolean OR.
"""
from dataset import ORDERS_BY_ID

W_DELAYED = 0.5
W_RECENCY = 0.5
MAX_DAYS = 30
ESCALATION_THRESHOLD = 0.40
OPEN_STATUSES = {"Placed", "Shipped", "Returned"}


def compute_escalation_score(order: dict) -> float:
    if order["status"] not in OPEN_STATUSES:
        return 0.0
    recency_norm = min(order["days_since_created"] / MAX_DAYS, 1.0)
    score = W_DELAYED * float(order["delayed_shipment"]) + W_RECENCY * recency_norm
    return round(score, 3)


def check_order_status(record_id: str) -> dict:
    """Look up one order. Returns status, order_value_inr and escalation_score in [0, 1]."""
    order = ORDERS_BY_ID.get(str(record_id).strip().upper())
    if order is None:
        return {"found": False, "record_id": record_id, "error": f"No order found with id {record_id!r}"}
    score = compute_escalation_score(order)
    return {
        "found": True,
        "record_id": order["record_id"],
        "status": order["status"],
        "order_value_inr": order["order_value_inr"],
        "escalation_score": score,
        "escalate_recommended": score >= ESCALATION_THRESHOLD,
        "delayed_shipment": order["delayed_shipment"],
        "days_since_created": order["days_since_created"],
    }


def _percentile(values, q):
    v = sorted(values)
    k = (len(v) - 1) * q
    f = int(k)
    c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)


if __name__ == "__main__":
    from dataset import ORDERS
    open_orders = [o for o in ORDERS if o["status"] in OPEN_STATUSES]
    scores = [compute_escalation_score(o) for o in open_orders]
    days = [o["days_since_created"] for o in open_orders]
    flagged = [o for o in open_orders if compute_escalation_score(o) >= ESCALATION_THRESHOLD]

    print("FORMULA: score = open_flag * (0.5*delayed + 0.5*days_since_created/30); escalate if >= 0.40\n")
    print(f"Open orders (Placed/Shipped/Returned): {len(open_orders)} of {len(ORDERS)}")
    print(f"  days_since_created among open orders: median={_percentile(days, .5):.1f}, "
          f"80th percentile={_percentile(days, .8):.1f}, max={max(days)}")
    print(f"  escalation_score among open orders  : median={_percentile(scores, .5):.3f}, "
          f"80th percentile={_percentile(scores, .8):.3f}, max={max(scores):.3f}")
    print(f"  threshold 0.40 flags {len(flagged)} of {len(open_orders)} open orders "
          f"({100 * len(flagged) / len(open_orders):.0f}%)")
    print(f"  non-delayed open order needs >= {(ESCALATION_THRESHOLD / W_RECENCY) * MAX_DAYS:.0f} days to escalate\n")

    picks = {
        "delayed + open (should escalate)": next(o for o in open_orders if o["delayed_shipment"]),
        "on-time + open, young (no escalation)": next(o for o in open_orders if not o["delayed_shipment"] and o["days_since_created"] <= 3),
        "on-time + open, very old (escalates on recency)": max((o for o in open_orders if not o["delayed_shipment"]), key=lambda o: o["days_since_created"]),
        "delayed but already Delivered (closed -> 0)": next(o for o in ORDERS if o["delayed_shipment"] and o["status"] == "Delivered"),
    }
    for label, o in picks.items():
        print(f"{label}:\n  {check_order_status(o['record_id'])}")
    print("unknown id:\n ", check_order_status("NYK-9999"))