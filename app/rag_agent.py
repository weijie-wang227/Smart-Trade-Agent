from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass
class Suggestion:
    suggested_hs_code: str | None
    confidence: float
    manual_review: bool
    cost:str
    reason: str
    retrieved: list[dict]


def _dist_to_sim(d: float) -> float:
    return 1.0 / (1.0 + d)


def _keyword_flags(text: str) -> set[str]:
    t = text.lower()
    flags = set()
    if any(k in t for k in ["solar", "panel", "photovoltaic"]):
        flags.add("renewables")
    if any(k in t for k in ["bluetooth", "headphone", "earbud", "wireless", "gpu", "processor", "usb", "power bank", "battery"]):
        flags.add("electronics")
    if any(k in t for k in ["cotton", "t-shirt", "fabric", "textile"]):
        flags.add("textiles")
    if any(k in t for k in ["coffee", "tea", "roasted", "beans"]):
        flags.add("food")
    return flags


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))

def _lexical_overlap(query: str, doc: str) -> float:
    q = _tokens(query)
    d = _tokens(doc)
    return len(q & d) / len(q) if q else 0.0

def _embed_sim(distance: float) -> float:
    return 1.0 / (1.0 + distance)

class HSCodeAgent:
    def __init__(self, vectorstore, ir_index, llm=None):
        self.vs = vectorstore
        self.ir = ir_index
        self.llm = llm

    def suggest(self, description: str, k: int = 3, use_hybrid: bool = False) -> Suggestion:
        results = self.vs.similarity_search_with_score(description, k=k)

        if not results:
            return Suggestion(None, 0.0, True, "No retrieval hits.", [])

        retrieved = [
            {
                "text": doc.page_content,
                "hs_code": doc.metadata.get("hs_code"),
                "category": doc.metadata.get("category"),
                "doc_id": doc.metadata.get("doc_id"),
                "score": score,
            }
            for doc, score in results
        ]

        # 1. IR scoring (once)
        if self.ir is not None:
            ir_scores = self.ir.score(description)
            for r in retrieved:
                r["ir_score"] = ir_scores[r["doc_id"]]
        else:
            for r in retrieved:
                r["ir_score"] = 0.0

        # 2. Similarity scoring
        if use_hybrid:
            for r in retrieved:
                dense = _embed_sim(r["score"])
                sparse = r["ir_score"]
                r["hybrid_score"] = 0.6 * dense + 0.4 * sparse

            retrieved.sort(key=lambda x: x["hybrid_score"], reverse=True)
            sims = [r["hybrid_score"] for r in retrieved]
        else:
            sims = [_dist_to_sim(r["score"]) for r in retrieved]

        # 3. Safety guard (recommended)
        if not sims:
            return Suggestion(None, 0.0, True, "No similarity scores computed.", retrieved)


        top_sim = sims[0]
        second_sim = sims[1] if len(sims) > 1 else 0.0

        confidence = top_sim / (top_sim + second_sim)

        reasons = []
        manual_review = False
        best_hs = retrieved[0]["hs_code"]
        cost = "$0"
        if self.llm is not None and 0.5 <= confidence <= 0.8:
            print("Using LLM to verify borderline confidence...")
            llm_score, llm_cost = self._llm_verify_score(
                description,
                retrieved,
            )
            cost = llm_cost
            confidence = min(1.0, confidence + 0.2 * llm_score)

        if confidence < 0.70:
            manual_review = True
            reasons.append(f"Low dominance confidence ({confidence:.2f})")

        if manual_review:
            return Suggestion(
                suggested_hs_code=best_hs,
                confidence=confidence,
                manual_review=True,
                cost=cost,
                reason="; ".join(reasons),
                retrieved=retrieved,
            )

        return Suggestion(
            suggested_hs_code=best_hs,
            confidence=confidence,
            manual_review=False,
            cost=cost,
            reason="High-confidence match",
            retrieved=retrieved,
        )
    
    def _llm_verify_score(
        self,
        description: str,
        retrieved: list[dict],
    ) -> float:
        context = "\n".join(
            f"HS: {r['hs_code']} | Category: {r['category']}" 
            for r in retrieved[:3]
        )

        prompt = f"""
            You are ranking HS-code candidates in a CLOSED SET.

            Product description:
            {description}

            Candidate HS codes (ONLY these are allowed):
            {context}

            Task:
            Assume the correct HS code MUST be chosen from the candidates above.
            You are NOT allowed to propose new HS codes or penalize missing options.

            Compare the TOP candidate against the other candidates from BEST to WORST.
            Judge ONLY which candidate is the best fit RELATIVE to the others.

            Scoring guidelines:
            • 1.0 = clearly the best among the given candidates
            • 0.7 = likely best, but with some uncertainty
            • 0.5 = comparable to others
            • 0.3 = weaker than at least one alternative
            • 0.0 = clearly the worst among the candidates

            Important rules:
            • Do NOT assess legal or technical correctness beyond comparison
            • Do NOT mention HS codes outside the candidate list
            • Do NOT explain your reasoning

            Return ONLY a single floating-point number between 0.0 and 1.0.
            """.strip()

        out, cost = self.llm.invoke(prompt)
        return self._parse_confidence(out), f"${cost:.7f}"
    
    def _parse_confidence(self, text: str) -> float:
        try:
            match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", text)
            if not match:
                return 0.0
            val = float(match.group())
            print(val)
            return max(0.0, min(1.0, val))
        except Exception:
            return 0.0



