import os
import streamlit as st
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
import numpy as np
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
index = pc.Index(os.getenv('PINECONE_INDEX_NAME'))

# Initialize Groq
client = Groq(api_key=os.getenv('GROQ_API_KEY'))

# Initialize the embedding model
model = SentenceTransformer('all-mpnet-base-v2')

def pad_embedding(embedding, target_dim=3072):
    """Pad the embedding to reach the target dimension."""
    current_dim = embedding.shape[0]
    if current_dim >= target_dim:
        return embedding[:target_dim].tolist()
    else:
        padding = np.zeros(target_dim - current_dim)
        return np.concatenate([embedding, padding]).tolist()

def get_embedding(text):
    embedding = model.encode(text)
    return pad_embedding(embedding)

def query_pinecone(embedding):
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    return results.matches

def generate_response(prompt):
    completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant for NHS GPs. Provide diagnoses and treatment plans based on patient symptoms.",
            },
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="mixtral-8x7b-32768",
        temperature=0.5,
        max_tokens=1000,
    )
    return completion.choices[0].message.content

st.title("NHS GP Assistant")

symptoms = st.text_area("Please enter the patient's symptoms:")

if st.button("Generate Diagnosis and Treatment Plan"):
    if symptoms:
        embedding = get_embedding(symptoms)
        similar_cases = query_pinecone(embedding)
        
        context = "Similar cases:\n" + "\n".join([case.metadata.get('text', 'No text available') for case in similar_cases])
        prompt = f"Given the following patient symptoms:\n{symptoms}\n\nAnd considering these similar cases:\n{context}\n\nProvide a possible diagnosis and treatment plan."
        
        response = generate_response(prompt)
        
        st.subheader("Diagnosis and Treatment Plan")
        st.write(response)
    else:
        st.warning("Please enter the patient's symptoms.")
