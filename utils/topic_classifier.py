def classify_topic(text: str) -> str:
    """Heuristic topic classification for dashboard display.

    Tight output set for predictable UI filtering:
    - Politics
    - Economy
    - Sports
    - Other
    """

    t = (text or "").lower()
    if not t.strip():
        return "Other"

    topics: dict[str, list[str]] = {
        "Politics": [
            "anc",
            "da",
            "eff",
            "parliament",
            "cabinet",
            "president",
            "minister",
            "election",
            "elections",
            "coalition",
            "vote",
            "government",
        ],
        "Economy": [
            "rand",
            "inflation",
            "interest rate",
            "interest rates",
            "repo rate",
            "gdp",
            "economy",
            "economic",
            "jobs",
            "unemployment",
            "budget",
            "tax",
            "sars",
            "trade",
        ],
        "Sports": [
            "rugby",
            "soccer",
            "football",
            "cricket",
            "springboks",
            "bafana",
            "proteas",
            "kaizer chiefs",
            "orlando pirates",
            "premiership",
        ],
    }

    scores: dict[str, int] = {}
    for topic, keys in topics.items():
        score = 0
        for k in keys:
            if k in t:
                score += 1
        scores[topic] = score

    best_topic, best_score = max(scores.items(), key=lambda kv: kv[1])
    if best_score <= 0:
        return "Other"

    # If everything is extremely weak, avoid an overconfident label.
    if best_score == 1 and sum(scores.values()) == 1:
        return "Other"

    return best_topic
