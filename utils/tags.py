def assign_tags(text):
    tags = []
    keywords = {
        "sports": [
            "rugby", "soccer", "cricket", "swimming", "tennis", "athletics", "cycling",
            "springboks", "bafana bafana", "orlando pirates", "kaizer chiefs",
            "blue bulls", "proteas", "sharks", "stormers"
        ],
        "crime": [
            "murder", "robbery", "police", "cash in transit", "heist", "kidnapping",
            "fraud", "arson", "assault", "gang violence"
        ],
        "energy": [
            "eskom", "load shedding", "power outage", "electricity", "renewable energy",
            "solar", "wind power", "nuclear", "grid failure"
        ],
        "economy": [
            "rand", "inflation", "jobs", "unemployment", "usdzar", "interest rates",
            "gdp", "economy", "trade", "investment"
        ],
        "politics": [
            "anc", "eff", "da", "parliament", "cabinet", "president", "policy",
            "elections", "government", "coalition"
        ]
    }
    for tag, words in keywords.items():
        if any(word.lower() in text.lower() for word in words):
            tags.append(tag)
    return tags
