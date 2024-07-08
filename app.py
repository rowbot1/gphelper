import streamlit as st
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

# Pinecone settings
PINECONE_API_KEY = st.secrets["pinecone"]["api_key"]
PINECONE_ENVIRONMENT = st.secrets["pinecone"]["environment"]
PINECONE_INDEX_NAME = st.secrets["pinecone"]["index_name"]

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
index = pc.Index(PINECONE_INDEX_NAME)

# Initialize the base embedding model
base_model = SentenceTransformer('all-MiniLM-L6-v2')

# Custom layer to expand embeddings to 3072 dimensions
class EmbeddingExpander(nn.Module):
    def __init__(self, input_dim=384, output_dim=3072):
        super(EmbeddingExpander, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

# Initialize the embedding expander
embedding_expander = EmbeddingExpander()

@st.cache_resource
def get_models():
    return base_model, embedding_expander

def get_embedding(text):
    base_model, embedding_expander = get_models()
    # Get base embedding
    base_embedding = base_model.encode(text)
    # Expand embedding
    with torch.no_grad():
        expanded_embedding = embedding_expander(torch.tensor(base_embedding).float()).numpy()
    
    # Ensure the vector is exactly 3072 dimensions
    if len(expanded_embedding) < 3072:
        expanded_embedding = np.pad(expanded_embedding, (0, 3072 - len(expanded_embedding)))
    elif len(expanded_embedding) > 3072:
        expanded_embedding = expanded_embedding[:3072]
    
    return expanded_embedding.tolist()

def query_pinecone(embedding):
    try:
        results = index.query(
            vector=embedding,
            top_k=5,
            include_metadata=True
        )
        return results['matches']
    except Exception as e:
        st.error(f"Error querying Pinecone: {e}")
        return []

# ... rest of your Streamlit app code ...

if st.button("Generate Diagnosis and Treatment Plan"):
    if symptoms:
        try:
            embedding = get_embedding(symptoms)
            st.write(f"Embedding dimension: {len(embedding)}")  # Debug print
            similar_cases = query_pinecone(embedding)
            
            context = "Similar cases:\n" + "\n".join([case.get('metadata', {}).get('text', 'No text available') for case in similar_cases])
            prompt = f"Given the following patient symptoms:\n{symptoms}\n\nAnd considering these similar cases:\n{context}\n\nProvide a possible diagnosis and treatment plan."
            
            response = generate_response(prompt)
            
            if response:
                st.subheader("Diagnosis and Treatment Plan")
                st.write(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the patient's symptoms.")

# ... rest of your Streamlit app code ...
