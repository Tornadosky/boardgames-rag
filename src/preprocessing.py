# src/preprocessing.py
import spacy
from sentence_transformers import SentenceTransformer

def preprocess_text(all_rules):
    """Split rules into sentences and chunk them into groups."""
    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")
    
    doc = nlp(all_rules)
    sentences = [str(sent) for sent in doc.sents]
    
    num_sentence_chunk_size = 10
    return [sentences[i:i + num_sentence_chunk_size] for i in range(0, len(sentences), num_sentence_chunk_size)]

def embed_text_chunks(sentence_chunks, embedding_model):
    """Embed sentence chunks using the specified model."""
    text_chunks = [" ".join(chunk) for chunk in sentence_chunks]
    return embedding_model.encode(text_chunks, batch_size=32, convert_to_tensor=True)
