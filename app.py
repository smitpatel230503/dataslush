import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load the dataset from URL
url = "D:\0 MCA\Placement\data_slush_app\titles.csv"
df = pd.read_csv(url)

# Select and combine relevant columns for text embeddings
df = df[['Title', 'Description', 'Genres']]
df['combined_text'] = df['Title'] + " " + df['Description'] + " " + df['Genres']

# Load pre-trained model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# Generate embeddings for all movies
df['embeddings'] = df['combined_text'].apply(lambda x: model.encode(x))

# Convert embeddings to numpy array
embeddings = np.vstack(df['embeddings'].values)

# Initialize Faiss index
d = embeddings.shape[1]  # Dimension of embeddings
index = faiss.IndexFlatL2(d)  # Using L2 distance (Euclidean)
index.add(embeddings)

# Streamlit UI
st.title('Movie Search Engine')

# User input for movie search
user_input = st.text_input('Enter movie description (e.g., heartfelt romantic comedy)', '')

# Process search if input is provided
if user_input:
    # Generate query embedding
    query_embedding = model.encode(user_input).reshape(1, -1)
    
    # Search for similar movies
    _, I = index.search(query_embedding, k=5)
    
    # Display the results
    st.subheader('Search Results:')
    for idx in I[0]:
        st.write(f"Title: {df.iloc[idx]['Title']}")
        st.write(f"Description: {df.iloc[idx]['Description']}")
        st.write(f"Genres: {df.iloc[idx]['Genres']}")
        st.write('---')
