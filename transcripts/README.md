# Nykaa Domain Support Agent (CrewAI) — E-commerce & Retail track

I completed the **Nykaa (E-commerce & Retail)** track. Everything runs under MOCK_LLM with zero API keys.

## Part 1, Task 1: dataset design choices
- Seed: 42, 60 records
- Category weights: Beauty 0.30, Apparel 0.25, Footwear/Home/Electronics 0.15 each
- Status weights: Delivered 0.40, Shipped 0.20, Placed/Returned 0.15 each, Refunded 0.10
- Amount range: ₹199–₹14,999 (per-category bands)
- Delayed-shipment rate: 13/60 = 21.7%

## Part 1, Task 2: knowledge base
12 documents in `kb/`. The policies are illustrative, not Nykaa's official policy.

## Part 1, Task 4: threshold calibration
Measured top-1 cosine similarities (all-MiniLM-L6-v2):
- kb_fixed:    in-scope min 0.411, out-of-scope max 0.136 → threshold 0.27
- kb_sentence: in-scope min 0.386, out-of-scope max 0.138 → threshold 0.26