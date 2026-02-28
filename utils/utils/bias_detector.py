from transformers import pipeline

# Load model once
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def detect_bias(text):

    labels = ["Left Bias", "Right Bias", "Neutral"]

    result = classifier(text, labels)

    best_label = result["labels"][0]
    confidence = round(result["scores"][0] * 100, 2)

    return best_label, confidence