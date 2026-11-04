"""
Task 4 - empirical threshold calibration.

Measures top-1 cosine similarity for in-scope and deliberately out-of-scope queries
(separately for each collection) and suggests a threshold between the two clusters.

Run:  python -m rag.calibrate
"""
from rag.config import COLLECTION_FIXED, COLLECTION_SENTENCE
from rag.grounded import retrieve
from rag.queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE


def top1(query, collection):
    return retrieve(query, collection, k=1)[0]["similarity"]


if __name__ == "__main__":
    for name in (COLLECTION_FIXED, COLLECTION_SENTENCE):
        print("=" * 80)
        print(f"COLLECTION: {name}")
        print("=" * 80)
        ins = [(q, top1(q, name)) for q in CALIBRATION_IN_SCOPE]
        outs = [(q, top1(q, name)) for q in CALIBRATION_OUT_OF_SCOPE]
        print("\nIn-scope queries (top-1 similarity):")
        for q, s in ins:
            print(f"  {s:.3f}  {q}")
        print("\nOut-of-scope queries (top-1 similarity):")
        for q, s in outs:
            print(f"  {s:.3f}  {q}")
        lowest_in = min(s for _, s in ins)
        highest_out = max(s for _, s in outs)
        print(f"\nLowest in-scope  = {lowest_in:.3f}")
        print(f"Highest out-of-scope = {highest_out:.3f}")
        if lowest_in > highest_out:
            thr = round((lowest_in + highest_out) / 2, 2)
            print(f"Clusters separate (gap {lowest_in - highest_out:.3f}).")
            print(f"==> SUGGESTED THRESHOLD for {name}: {thr}")
        else:
            print("!! Clusters OVERLAP - no clean threshold. Send me this output.")
        print()