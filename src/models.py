# src/models.py
import umap
import torch
import numpy as np
from sklearn.cluster import KMeans
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
from gpt4all import GPT4All

def load_models():
    """Load the embedding model, Gemma, and GPT4All models."""
    embedding_model = SentenceTransformer("all-mpnet-base-v2")
    
    # Load Gemma
    gemma_tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
    gemma_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-7b-it",
        device_map="cuda",
        torch_dtype=torch.bfloat16
    )
    
    # Load GPT4All
    gpt4all_model = GPT4All('mistral-7b-instruct-v0.1.Q4_0.gguf', device='cuda')
    
    return embedding_model, gemma_model, gemma_tokenizer, gpt4all_model

def apply_umap(embeddings, n_neighbors=15, n_components=2):
    """Reduce dimensions of embeddings using UMAP."""
    reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, metric='cosine', random_state=42)
    return reducer.fit_transform(embeddings.cpu().numpy())

def perform_clustering(umap_embeddings, n_clusters=5):
    """Perform KMeans clustering on reduced embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(umap_embeddings)

def generate_word_clouds(umap_df, n_clusters):
    """Generate word clouds for each cluster using KeyBERT for keyword extraction."""
    fig, axes = plt.subplots(1, n_clusters, figsize=(20, 5))
    kw_model = KeyBERT()
    
    for i in range(n_clusters):
        cluster_text = ' '.join(umap_df[umap_df['Cluster'] == i]['Sentence Chunk'].tolist())
        keywords = kw_model.extract_keywords(cluster_text, top_n=10, keyphrase_ngram_range=(1, 1))
        keyword_text = ' '.join([word for word, score in keywords])

        # Generate a word cloud for each cluster
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(keyword_text)
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'Cluster {i+1} Keywords')
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()

def ask_gemma(query, model, tokenizer, embeddings, temperature=0.7, max_new_tokens=150):
    """Query Gemma model with context from embeddings."""
    query_embedding = embeddings[0]
    input_ids = tokenizer(query, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids, max_new_tokens=max_new_tokens)
    return tokenizer.decode(outputs[0]), query_embedding

def ask_gpt4all(query, model, embeddings, n_resources_to_return=5):
    """Query GPT4All model with context from embeddings."""
    prompt = f"Answer the following query: {query}\nContext: {embeddings[:5]}"
    tokens = []
    with model.chat_session():
        for token in model.generate(prompt, streaming=True):
            tokens.append(token)
    return "".join(tokens), embeddings[:5]
