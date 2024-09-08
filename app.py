import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
import faiss
import streamlit as st

# Read the CSV file from the URL
url = "https://raw.githubusercontent.com/datum-oracle/netflix-movie-titles/main/titles.csv"  # replace with the actual URL
df = pd.read_csv(url)

# Select the feature columns
df = df[["title", "description", "genres"]]

# Combine the columns into a single text column
df["text"] = df["title"] + " " + df["description"] + " " + df["genres"]

# Load the pre-trained model and tokenizer
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Generate embeddings for the text data
embeddings = []
for text in df["text"].values:  # Use .values to get a numpy array
    inputs = tokenizer(str(text), return_tensors="pt", max_length=512, padding="max_length", truncation=True)
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state[:, 0, :].detach().numpy()[0])

# Index the embeddings in a vector store
index = faiss.IndexFlatL2(len(embeddings[0]))
faiss.write_index(index, "movies_index.faiss")

# Create a user interface for the application
st.title("Movie Search Engine")

# Get the user's query
query = st.text_input("Enter your desired movie characteristics (e.g., 'heartfelt romantic comedy')")

# Generate an embedding for the user's query
inputs = tokenizer(query, return_tensors="pt")
outputs = model(**inputs)
query_embedding = outputs.last_hidden_state[:, 0, :].detach().numpy()[0]

# Search for movies with embeddings closest to the user's query embedding
D, I = index.search(query_embedding.reshape(1, -1), k=10)

# Display the search results
st.write("Search Results:")
for i, idx in enumerate(I[0]):
    st.write(f"{i+1}. {df.iloc[idx]['title']} ({df.iloc[idx]['release_year']})")

# Add filters to sort results
st.write("Filters:")
score_filter = st.checkbox("Sort by IMDB Score")
votes_filter = st.checkbox("Sort by IMDB Votes")

if score_filter:
    df_sorted = df.sort_values(by="imdb_score", ascending=False)
elif votes_filter:
    df_sorted = df.sort_values(by="imdb_votes", ascending=False)
else:
    df_sorted = df

# Display the sorted results
st.write("Sorted Results:")
for i, idx in enumerate(I[0]):
    st.write(f"{i+1}. {df_sorted.iloc[idx]['title']} ({df_sorted.iloc[idx]['release_year']})")
