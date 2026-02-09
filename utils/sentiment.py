import os

from transformers import pipeline


_DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_MODEL_NAME = os.environ.get("SENTIMENT_MODEL_NAME", _DEFAULT_MODEL)

sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)


def analyze_sentiment(text):
    result = sentiment_analyzer(text)[0]
    return {"label": result["label"], "score": round(float(result["score"]), 2)}
