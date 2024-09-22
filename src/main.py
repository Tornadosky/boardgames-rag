# src/main.py
import os
import torch
from src.utils import load_rules, text_formatter
from src.preprocessing import preprocess_text, embed_text_chunks
from src.models import load_models, ask_gemma, ask_gpt4all, apply_umap, perform_clustering, generate_word_clouds
import plotly.express as px
import pandas as pd
import numpy as np

def main():
    # Load board game rules
    data_folder = "data/"
    rules = load_rules(data_folder)
    
    # Preprocess data
    formatted_rules = {name: text_formatter(text) for name, text in rules.items()}
    all_rules = " ".join(formatted_rules.values())
    sentence_chunks = preprocess_text(all_rules)
    
    # Load models
    embedding_model, gemma_model, gemma_tokenizer, gpt4all_model = load_models()

    # Embedding the text
    embeddings = embed_text_chunks(sentence_chunks, embedding_model)
    embeddings = torch.tensor(embeddings, dtype=torch.float32)
    
    # UMAP for dimensionality reduction
    umap_embeddings = apply_umap(embeddings)
    
    # KMeans Clustering
    n_clusters = 5
    clusters = perform_clustering(umap_embeddings, n_clusters=n_clusters)
    
    # Create a DataFrame for easy plotting
    umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP Dimension 1', 'UMAP Dimension 2'])
    umap_df['Sentence Chunk'] = [" ".join(chunk) for chunk in sentence_chunks]
    umap_df['Cluster'] = clusters

    # Visualization with plotly: UMAP scatter plot
    fig = px.scatter(
        umap_df,
        x='UMAP Dimension 1',
        y='UMAP Dimension 2',
        color='Cluster',
        hover_data=['Sentence Chunk'],
        title='UMAP 2D Projection of Sentence Embeddings'
    )
    fig.show()

    # Generate word clouds for each cluster
    generate_word_clouds(umap_df, n_clusters)

    # Define test queries and ground truth answers
    queries = [
        ("How much money does each player start with in Monopoly?", "Each player starts with $1500 in Monopoly."),
        ("How many agent cards are there for each team in Codenames?", "There are 8 red agent cards and 8 blue agent cards."),
        ("What happens if you contact the assassin in Codenames?", "If a field operative touches the assassin, the game ends immediately and that operative's team loses."),
        ("What is the objective of Ticket to Ride?", "The objective is to score the highest number of total points by claiming routes and completing tickets."),
        ("How do you win Battleship?", "You win Battleship by being the first to sink all 5 of your opponent’s ships."),
        ("What are the possible actions a player can take on their turn in Ticket to Ride?", "A player can draw train car cards, claim a route, or draw destination tickets."),
        ("How many Exploding Kittens should be inserted back into the deck for a game of five players?", "Insert 4 Exploding Kittens back into the deck for a game of five players."),
        ("What does a Defuse Card do when you draw an Exploding Kitten?", "A Defuse Card prevents you from exploding by allowing you to place the Exploding Kitten back into the draw pile anywhere you'd like."),
    ]

    # Test Gemma
    print("\nTesting Gemma model:")
    for query, truth in queries:
        answer, context = ask_gemma(query, gemma_model, gemma_tokenizer, embeddings)
        print(f"Query: {query}")
        print(f"Model's Answer: {answer.strip()}")
        print(f"Truth: {truth}")
        print(f"Context: {''.join([i['sentence_chunk'] for i in context])}")
        print("\n")

    # Test GPT4All
    print("\nTesting GPT4All model:")
    for query, truth in queries:
        answer, context = ask_gpt4all(query, gpt4all_model, embeddings)
        print(f"Query: {query}")
        print(f"Model's Answer: {answer.strip()}")
        print(f"Truth: {truth}")
        print(f"Context: {''.join([i['sentence_chunk'] for i in context])}")
        print("\n")

if __name__ == "__main__":
    main()
