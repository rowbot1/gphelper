import streamlit as st
import numpy as np
from pinecone import Pinecone
from groq import Groq
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn

# Initialize Pinecone
try:
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index(st.secrets["pinecone"]["index_name"])
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Initialize Groq
try:
    client = Groq(api_key=st.secrets["groq"]["api_key"])
except Exception as e:
    st.error(f"Failed to initialize Groq: {e}")
    st.stop()

# Initialize the embedding model and expander
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_model = SentenceTransformer('all-MiniLM-L6-v2').to(device)

class EmbeddingExpander(nn.Module):
    def __init__(self, input_dim=384, output_dim=3072):
        super(EmbeddingExpander, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.fc(x)

embedding_expander = EmbeddingExpander().to(device)

@st.cache_resource
def load_models():
    return base_model, embedding_expander

base_model, embedding_expander = load_models()

@st.cache_data
def get_expanded_embedding(text):
    with torch.no_grad():
        base_embedding = base_model.encode(text, convert_to_tensor=True).to(device)
        expanded_embedding = embedding_expander(base_embedding).cpu().numpy()
    return expanded_embedding.tolist()

def query_pinecone(embedding):
    try:
        results = index.query(vector=embedding, top_k=5, include_metadata=True)
        return results['matches']
    except Exception as e:
        st.warning(f"Failed to query Pinecone: {e}. Proceeding without similar cases.")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_response(prompt):
    try:
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
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return None

st.title(st.secrets["app"]["name"])

if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

symptoms = st.text_area("Please enter the patient's symptoms:")

if st.button("Generate Diagnosis and Treatment Plan"):
    if symptoms:
        try:
            embedding = get_expanded_embedding(symptoms)
            similar_cases = query_pinecone(embedding)
            if not similar_cases:
                context = "No similar cases available."
            else:
                context = "Similar cases:\n" + "\n".join([case.get('metadata', {}).get('text', 'No text available') for case in similar_cases])
            prompt = f"Given the following patient symptoms:\n{symptoms}\n\nAnd considering these similar cases:\n{context}\n\nProvide a possible diagnosis and treatment plan."
            st.session_state.diagnosis = generate_response(prompt)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the patient's symptoms.")

if st.session_state.diagnosis:
    st.subheader("Diagnosis and Treatment Plan")
    st.write(st.session_state.diagnosis)

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
