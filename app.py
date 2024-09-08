import os
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import faiss

# Set environment variable
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Read the CSV file from the URL
url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"  # replace with the actual URL
df = pd.read_csv(url)

# Select the feature columns
df = df[["title", "description", "genres", "release_year", "imdb_score", "imdb_votes"]]

# Combine the columns into a single text column
df["text"] = df["title"] + " " + df["description"] + " " + df["genres"]

# Load the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)

# Generate embeddings for the text data
embeddings = []
for text in df["text"].values:  # Use .values to get a numpy array
    inputs = tokenizer(str(text), return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy()[0])

# Index the embeddings in a vector store
index = faiss.IndexFlatL2(len(embeddings[0]))
faiss.write_index(index, "movies_index.faiss")

def search_movies(query):
    # Generate an embedding for the user's query
    inputs = tokenizer(query, return_tensors="pt")
    outputs = model(**inputs)
    query_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]

    # Search for movies with embeddings closest to the user's query embedding
    index = faiss.read_index("movies_index.faiss")
    D, I = index.search(query_embedding.reshape(1, -1), k=10)

    # Display the search results
    print("Search Results:")
    for i, idx in enumerate(I[0]):
        print(f"{i+1}. {df.iloc[idx]['title']} ({df.iloc[idx]['release_year']})")

    # Add filters to sort results
    score_filter = input("Sort by IMDB Score? (yes/no): ")
    votes_filter = input("Sort by IMDB Votes? (yes/no): ")

    if score_filter.lower() == "yes":
        df_sorted = df.sort_values(by="imdb_score", ascending=False)
    elif votes_filter.lower() == "yes":
        df_sorted = df.sort_values(by="imdb_votes", ascending=False)
    else:
        df_sorted = df

    # Display the sorted results
    print("Sorted Results:")
    for i, idx in enumerate(I[0]):
        print(f"{i+1}. {df_sorted.iloc[idx]['title']} ({df_sorted.iloc[idx]['release_year']})")

def main():
    query = input("Enter your desired movie characteristics (e.g., 'heartfelt romantic comedy'): ")
    search_movies(query)

if __name__ == "__main__":
    main()