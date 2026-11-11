import os

import gradio as gr
from transformers import pipeline

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BASELINE_NAME = "Baseline (DistilBERT SST-2)"
FINETUNED_NAME = "Fine-tuned on IMDB"
MODEL_PATHS = {
    BASELINE_NAME: "distilbert-base-uncased-finetuned-sst-2-english",
}
finetuned_dir = os.path.join(BASE_DIR, "models", "finetuned-imdb")
if os.path.isdir(finetuned_dir):
    MODEL_PATHS[FINETUNED_NAME] = finetuned_dir

MAX_CHARS = 2000
_cache = {}


def get_classifier(model_choice):
    if model_choice not in _cache:
        _cache[model_choice] = pipeline(
            "sentiment-analysis", model=MODEL_PATHS[model_choice]
        )
    return _cache[model_choice]


def analyze_sentiment(text, model_choice):
    if not text or not text.strip():
        return "Please enter some text.", {}
    if len(text) > MAX_CHARS:
        return f"Text too long (max {MAX_CHARS} characters).", {}
    if model_choice not in MODEL_PATHS:
        return "Please choose a model.", {}
    try:
        clf = get_classifier(model_choice)
        result = clf(text, truncation=True, max_length=512)[0]
        label = result["label"].upper()
        label = "POSITIVE" if label in ("POSITIVE", "LABEL_1") else "NEGATIVE"
        score = float(result["score"])
        other = "NEGATIVE" if label == "POSITIVE" else "POSITIVE"
        confidences = {label: score, other: 1.0 - score}
        return f"{label} ({score:.1%} confidence)", confidences
    except Exception as e:
        return f"Error: {e}", {}


demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=[
        gr.Textbox(lines=5, label="Enter your text"),
        gr.Dropdown(
            choices=list(MODEL_PATHS.keys()),
            value=BASELINE_NAME,
            label="Model",
        ),
    ],
    outputs=[
        gr.Textbox(label="Predicted sentiment"),
        gr.Label(label="Confidence scores"),
    ],
    title="Sentiment Analysis with DistilBERT",
    description=(
        "Type a sentence or review to see if it is positive or negative. "
        "Note: the model only has two classes, so neutral text is forced "
        "into positive or negative, and confidence can be overstated."
    ),
    examples=[
        ["I absolutely loved this movie, it was fantastic!", BASELINE_NAME],
        ["The service was terrible and the food was cold.", BASELINE_NAME],
        ["It was okay, nothing special.", BASELINE_NAME],
    ],
)

if __name__ == "__main__":
    demo.launch()