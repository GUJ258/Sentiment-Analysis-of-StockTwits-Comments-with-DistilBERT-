from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from config import MODEL_NAME, NUM_LABELS

def load_tokenizer():
    return DistilBertTokenizer.from_pretrained(MODEL_NAME, clean_up_tokenization_spaces=False)

def load_model(num_labels=NUM_LABELS):
    return DistilBertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=num_labels)
