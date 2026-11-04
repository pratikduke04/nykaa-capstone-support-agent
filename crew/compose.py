"""
The Response Composer's logic (used by the MockLLM when an agent has no tools).

Reads the Retrieval Agent's output ([RETRIEVAL]{json}[/RETRIEVAL]) and the Lookup Agent's output
([LOOKUP]{json}[/LOOKUP]) from its task context, and writes ONE AnswerSchema JSON.

The policy part is EXTRACTIVE: it picks the sentences of the retrieved documents that best overlap the
question's keywords. Nothing is invented, so the draft is grounded by construction.
"""
import json
import math
import re

from crew.schemas import FALLBACK_ANSWER, AnswerSchema, OrderInfo

MAX_SENTENCES = 3
PRODUCT_WEIGHT = 0.25     # product words (hair dryer) count less than the category they imply (Electronics)
NO_NUMBER_FACTOR = 0.25  # ...and sentences WITHOUT a number are down-weighted for those questions
NUMBER_BONUS = 1.0        # "how long / how many / when" questions prefer sentences that contain numbers
STOP = {
    "the", "and", "for", "are", "you", "can", "how", "what", "does", "did", "was", "with", "this", "that",
    "have", "has", "from", "your", "our", "its", "not", "but", "all", "any", "get", "got", "will", "would",
    "should", "could", "about", "when", "where", "which", "into", "out", "who", "why", "their", "there",
    "than", "then", "them", "they", "these", "those", "per", "also", "may", "might", "must", "need", "want",
    "long", "take", "takes", "please", "tell", "know", "much", "many", "make", "being", "been", "were", "soon",
    "after", "again", "can't", "cant", "just", "like", "some", "does", "doing", "done", "use",
}
# Map everyday product words to the category words the policies use (the KB talks about categories).
CATEGORY_WORDS = {
    "electronics": ["hair dryer", "hairdryer", "dryer", "trimmer", "straightener", "curler", "epilator", "shaver",
                    "headphones", "earphones", "speaker", "charger", "phone", "mobile", "gadget"],
    "beauty": ["lipstick", "lip balm", "serum", "moisturizer", "moisturiser", "shampoo", "conditioner",
               "fragrance", "cream", "kajal", "mascara", "foundation", "makeup", "cosmetics", "skincare",
               "sunscreen", "nail polish"],
    "apparel": ["kurta", "dress", "jeans", "t-shirt", "tshirt", "shirt", "saree", "jacket", "hoodie",
                "leggings", "clothes", "clothing"],
    "footwear": ["shoes", "shoe", "sandals", "sandal", "sneakers", "heels", "boots", "slippers", "flip flops"],
    "home": ["lamp", "cushion", "bedsheet", "vase", "curtain", "towel", "candle", "mug"],
}
# Everyday wording -> the words the policies actually use (added to the question's keywords).
SYNONYMS = [
    (r"\b(broken|torn|defective|faulty|crushed|leaked|leaking|spilled|cracked|smashed|wrong item)\b", "damaged"),
    (r"\b(pay again|try again|retry)\b", "retry payment"),
    (r"\b(money back|get my money)\b", "refund"),
    (r"\b(late|delayed|not arrived|not received|hasn't arrived|has not arrived)\b", "delayed"),
    (r"\b(swap|bigger size|smaller size|change the size|change size)\b", "exchange"),
]
_KB_STATS = None


def extract_tagged(msgs: list[dict], tag: str):
    """JSON inside [TAG]...[/TAG] from system/user messages (never from the model's own turns)."""
    pat = re.compile(rf"\[{tag}\](.*?)\[/{tag}\]", re.DOTALL)
    for m in msgs:
        if m["role"] == "assistant":
            continue
        hit = pat.search(m["content"])
        if hit:
            try:
                return json.loads(hit.group(1))
            except json.JSONDecodeError:
                return None
    return None


def stem(word: str) -> str:
    for suf in ("ing", "ed", "es", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            word = word[: -len(suf)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word[:6]


def stem_set(text: str) -> set[str]:
    return {stem(w) for w in re.findall(r"[a-z]+", text.lower()) if len(w) >= 3 and w not in STOP}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _kb_idf() -> dict:
    """Inverse document frequency of each stem over every sentence in the knowledge base."""
    global _KB_STATS
    if _KB_STATS is None:
        from rag.chunking import load_documents
        sents = [x for d in load_documents() for x in split_sentences(d["text"])]
        df = {}
        for x in sents:
            for st in stem_set(x):
                df[st] = df.get(st, 0) + 1
        n = len(sents)
        _KB_STATS = {"n": n, "df": df}
    n, df = _KB_STATS["n"], _KB_STATS["df"]
    return {st: math.log((n + 1) / (c + 1)) + 1 for st, c in df.items()}


def weighted_question_stems(question: str) -> dict:
    """{stem: weight}. Product words are down-weighted and the category they imply is added at full weight."""
    q = re.sub(r"\bpick\w*\s+(?:(?:them|it|this|that|these|those|my|the|me)\s+)*up\b", "pickup", question.lower())
    weights = {}
    for category, words in CATEGORY_WORDS.items():
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", q):
                weights[stem(category)] = 1.0
                q = re.sub(rf"\b{re.escape(w)}\b", " ", q)
                for st in stem_set(w):
                    weights[st] = PRODUCT_WEIGHT
    for pattern, extra in SYNONYMS:
        if re.search(pattern, q):
            for st in stem_set(extra):
                weights[st] = 1.0
    for st in stem_set(q):
        weights.setdefault(st, 1.0)
    return weights


def score_sentence(sentence: str, question: str, q_weights: dict, idf: dict) -> float:
    sent_stems = stem_set(sentence)
    score = sum(w * idf.get(st, 1.0) for st, w in q_weights.items() if st in sent_stems)
    numeric_q = re.search(r"\b(how long|how many|how soon|how much|when|days?|time)\b", question.lower())
    if score > 0 and numeric_q:
        score = score + NUMBER_BONUS if re.search(r"\d", sentence) else score * NO_NUMBER_FACTOR
    return score


def select_sentences(question: str, docs: list[dict], limit: int = MAX_SENTENCES):
    """Best-scoring sentences across all retrieved docs. Returns [(doc_id, sentence)] in reading order."""
    idf, qw = _kb_idf(), weighted_question_stems(question)
    cands = []
    for rank, d in enumerate(docs):
        for i, s in enumerate(split_sentences(d["text"])):
            cands.append((score_sentence(s, question, qw, idf), d["similarity"], rank, i, d["doc_id"], s))
    ranked = sorted((c for c in cands if c[0] > 0), key=lambda c: (-c[0], -c[1], c[2], c[3]))
    best = ranked[:limit]
    if ranked:
        floor = 0.5 * ranked[0][0]
        for doc_id in dict.fromkeys(c[4] for c in ranked):          # best sentence of each strong document
            top_of_doc = next(c for c in ranked if c[4] == doc_id)
            if top_of_doc[0] >= floor and top_of_doc not in best:
                weakest = min(best, key=lambda c: c[0])
                if not any(c[4] == doc_id for c in best):
                    best = [c for c in best if c is not weakest] + [top_of_doc]
    if not best and docs:                         # vague question: fall back to the top document's first sentence
        top = docs[0]
        return [(top["doc_id"], split_sentences(top["text"])[0])]
    best.sort(key=lambda c: (c[2], c[3]))
    return [(c[4], c[5]) for c in best]


def compose_answer(question: str, retrieval, lookup) -> AnswerSchema:
    docs = (retrieval or {}).get("documents") or []
    policy_ok = bool(retrieval and retrieval.get("grounded") and docs)
    order = lookup if (lookup and lookup.get("found")) else None
    top_sim = float((retrieval or {}).get("top_similarity", 0.0))

    parts, sources = [], []
    if policy_ok:
        picked = select_sentences(question, docs)
        parts.append("According to our policy: " + " ".join(s for _, s in picked))
        sources = list(dict.fromkeys(doc_id for doc_id, _ in picked))

    order_info = None
    if order:
        order_info = OrderInfo(record_id=order["record_id"], status=order["status"],
                               order_value_inr=order["order_value_inr"],
                               escalation_score=order["escalation_score"],
                               escalate_recommended=order["escalate_recommended"])
        line = f"Order {order['record_id']} is currently {order['status']} (order value Rs {order['order_value_inr']:,})."
        if order["escalate_recommended"]:
            line += f" Our system recommends escalating it to the support team (escalation score {order['escalation_score']:.2f})."
        parts.append(line)
    elif lookup and lookup.get("error") and "No order id" not in lookup["error"]:
        parts.append("I could not find that order id in our system, so I cannot give its status.")

    if policy_ok and order:
        intent = "policy_and_order"
    elif policy_ok:
        intent = "policy"
    elif order:
        intent = "order"
    else:
        intent = "out_of_scope"

    if not parts or intent == "out_of_scope":
        answer = FALLBACK_ANSWER if not parts else " ".join(parts)
        return AnswerSchema(intent="out_of_scope", answer=answer, grounded=False, top_similarity=top_sim,
                            sources=[], order=None, escalate=False)
    return AnswerSchema(intent=intent, answer=" ".join(parts), grounded=True, top_similarity=top_sim,
                        sources=sources, order=order_info,
                        escalate=bool(order_info and order_info.escalate_recommended))


def compose_final_answer(msgs: list[dict], question: str) -> str:
    """Hook called by MockLLM for tool-less agents. Returns the AnswerSchema as a JSON string."""
    retrieval = extract_tagged(msgs, "RETRIEVAL")
    lookup = extract_tagged(msgs, "LOOKUP")
    return compose_answer(question, retrieval, lookup).model_dump_json()