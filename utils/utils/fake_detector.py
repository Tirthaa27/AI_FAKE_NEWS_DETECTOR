from transformers import pipeline

# Load model only once
classifier = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

def detect_fake_news(text):

    labels = ["Real News", "Fake News"]

    result = classifier(text, labels)

    best_label = result["labels"][0]
    confidence = round(result["scores"][0] * 100, 2)

    return best_label.upper(), confidence