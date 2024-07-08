import os
import streamlit as st
import numpy as np
import toml
from dotenv import load_dotenv

# Load environment variables (if you're still using .env for some configs)
load_dotenv()

# Load configuration from TOML file
try:
    config = toml.load("config.toml")
except Exception as e:
    st.error(f"Failed to load configuration: {e}")
    st.stop()

# Try to import required libraries
try:
    import pinecone
    from groq import Groq
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    st.error(f"Failed to import required library: {e}")
    st.stop()

# Initialize Pinecone
try:
    pinecone.init(
        api_key=config['pinecone']['api_key'],
        environment=config['pinecone']['environment']
    )
    index = pinecone.Index(config['pinecone']['index_name'])
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Initialize Groq
try:
    client = Groq(api_key=config['groq']['api_key'])
except Exception as e:
    st.error(f"Failed to initialize Groq: {e}")
    st.stop()

# Initialize the embedding model
try:
    model = SentenceTransformer('all-mpnet-base-v2')
except Exception as e:
    st.error(f"Failed to initialize SentenceTransformer: {e}")
    st.stop()

def get_embedding(text):
    return model.encode(text).tolist()

def query_pinecone(embedding):
    results = index.query(vector=embedding, top_k=5, include_metadata=True)
    return results['matches']

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

st.title(config['app']['name'])

symptoms = st.text_area("Please enter the patient's symptoms:")

if st.button("Generate Diagnosis and Treatment Plan"):
    if symptoms:
        try:
            embedding = get_embedding(symptoms)
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

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
