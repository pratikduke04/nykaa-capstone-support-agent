"""
Query sets used by Task 4 (calibration + demo) and Task 5 (precision/recall).

CALIBRATION sets are used ONLY to choose the threshold.
DEMO set is a separate set (so the threshold is not tuned on the queries we show off),
with hand-labelled relevant parent documents for Task 5.
"""
D = {
    "return": "01_return_window_by_category",
    "cod": "02_cod_refund_timelines",
    "delivery": "03_delivery_slas",
    "pickup": "04_reverse_pickup_eligibility",
    "warranty": "05_warranty_terms_by_category",
    "cancel": "06_order_cancellation_policy",
    "loyalty": "07_loyalty_points_redemption",
    "payment": "08_payment_failure_retry_policy",
    "exchange": "09_size_exchange_policy",
    "damaged": "10_damaged_item_claim_process",
    "intl": "11_international_shipping_restrictions",
    "escalation": "12_customer_support_escalation_matrix",
}

CALIBRATION_IN_SCOPE = [
    "Are opened cosmetics returnable?",
    "How soon will my COD money come back?",
    "How many loyalty points do I earn per purchase?",
    "How long does standard delivery take in a metro city?",
    "Who handles complaints that nobody has resolved after three days?",
    "Can I ship perfume abroad?",
    "Can I pay again if my UPI payment failed?",
]

CALIBRATION_OUT_OF_SCOPE = [
    "What is the capital of France?",
    "Recommend a good laptop for gaming",
    "How do I bake a chocolate cake?",
    "Explain quantum computing in simple words",
    "Who won the cricket world cup?",
]

# Demo / evaluation queries with hand-labelled relevant documents (set of doc_ids)
DEMO_QUERIES = [
    {"q": "How many days do I have to return a hair dryer?", "relevant": {D["return"]}},
    {"q": "How long does a refund take for a cash on delivery order?", "relevant": {D["cod"]}},
    {"q": "Can I cancel my order after it has shipped?", "relevant": {D["cancel"]}},
    {"q": "What warranty comes with a trimmer?", "relevant": {D["warranty"]}},
    {"q": "My parcel arrived broken, what should I do?", "relevant": {D["damaged"]}},
    {"q": "Can I swap my sandals for a bigger size?", "relevant": {D["exchange"]}},
    {"q": "My shoes do not fit, can you pick them up and refund me?",
     "relevant": {D["return"], D["pickup"], D["exchange"]}},
]

DEMO_OUT_OF_SCOPE = "What is the best movie of the year?"