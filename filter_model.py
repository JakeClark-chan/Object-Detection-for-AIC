from transformers import pipeline
from sentence_transformers import SentenceTransformer, util
import torch

# Load BERT model for object extraction
def load_bert():
    return pipeline('ner')

# Load CLIP model for image-text similarity
def load_clip():
    return SentenceTransformer('clip-ViT-B-32')

# Extract object labels from text using BERT
def extract_objects_from_text_bert(text, bert_model):
    ner_results = bert_model(text)
    object_labels = [res['word'] for res in ner_results if res['entity'].startswith('B')]
    return list(set(object_labels))  # Unique object labels

# Extract object labels using CLIP (multimodal similarity)
def extract_objects_from_text_clip(text, clip_model, labels):
    text_embedding = clip_model.encode(text, convert_to_tensor=True)
    object_embeddings = clip_model.encode(labels, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(text_embedding, object_embeddings)
    return [labels[idx] for idx in similarities.argsort(descending=True)[0][:5]]  # Top 5 matches
