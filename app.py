import streamlit as st
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Wrap imports in try-except blocks
try:
    from pinecone import Pinecone
    logger.info("Pinecone imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Pinecone: {e}")
    st.error("Failed to import Pinecone. Please check your installation.")

try:
    from groq import Groq
    logger.info("Groq imported successfully")
except ImportError as e:
    logger.error(f"Failed to import Groq: {e}")
    st.error("Failed to import Groq. Please check your installation.")

try:
    from sentence_transformers import SentenceTransformer
    logger.info("SentenceTransformer imported successfully")
except ImportError as e:
    logger.error(f"Failed to import SentenceTransformer: {e}")
    st.error("Failed to import SentenceTransformer. Please check your installation.")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index(st.secrets["pinecone"]["index_name"])
    logger.info("Pinecone initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {e}")
    st.error(f"Failed to initialize Pinecone: {e}")

# Initialize Groq
try:
    client = Groq(api_key=st.secrets["groq"]["api_key"])
    logger.info("Groq initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq: {e}")
    st.error(f"Failed to initialize Groq: {e}")

# Initialize the embedding model
try:
    model = SentenceTransformer('all-mpnet-base-v2')
    logger.info("SentenceTransformer model initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer model: {e}")
    st.error(f"Failed to initialize SentenceTransformer model: {e}")

def get_embedding(text):
    try:
        return model.encode(text).tolist()
    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        st.error(f"Error in get_embedding: {e}")
        return []

def query_pinecone(embedding):
    try:
        results = index.query(
            vector=embedding, 
            top_k=5, 
            include_metadata=True
        )
        return results['matches']
    except Exception as e:
        logger.error(f"Error querying Pinecone: {e}")
        st.error(f"Error querying Pinecone: {e}")
        return []

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
        logger.error(f"Failed to generate response: {e}")
        st.error(f"Failed to generate response: {e}")
        return None

st.title(st.secrets["app"]["name"])

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
            logger.error(f"An error occurred: {e}")
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the patient's symptoms.")

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
