import gradio as gr
from transformers import pipeline

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_CHARS = 2000

classifier = pipeline("sentiment-analysis", model=MODEL_NAME)


def analyze_sentiment(text):
    if not text or not text.strip():
        return "Please enter some text.", {}
    if len(text) > MAX_CHARS:
        return f"Text too long (max {MAX_CHARS} characters).", {}
    try:
        result = classifier(text, truncation=True, max_length=512)[0]
        label = result["label"]
        score = float(result["score"])
        other = "NEGATIVE" if label == "POSITIVE" else "POSITIVE"
        confidences = {label: score, other: 1.0 - score}
        return f"{label} ({score:.1%} confidence)", confidences
    except Exception as e:
        return f"Error: {e}", {}


demo = gr.Interface(
    fn=analyze_sentiment,
    inputs=gr.Textbox(lines=5, label="Enter your text"),
    outputs=[
        gr.Textbox(label="Predicted sentiment"),
        gr.Label(label="Confidence scores"),
    ],
    title="Sentiment Analysis with DistilBERT",
    description="Type a sentence or review to see if it is positive or negative.",
    examples=[
        ["I absolutely loved this movie, it was fantastic!"],
        ["The service was terrible and the food was cold."],
        ["It was okay, nothing special."],
    ],
)

if __name__ == "__main__":
    demo.launch()