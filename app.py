import streamlit as st
from openai import OpenAI
import numpy as np
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import torch

# Set page config at the very beginning
st.set_page_config(page_title=st.secrets["app"]["name"], layout="wide")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=st.secrets["pinecone"]["api_key"])
    index = pc.Index(st.secrets["pinecone"]["index_name"])
except Exception as e:
    st.error(f"Failed to initialize Pinecone: {e}")
    st.stop()

# Initialize OpenAI
try:
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {e}")
    st.stop()

# Initialize the embedding model
@st.cache_resource
def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return SentenceTransformer('all-mpnet-base-v2').to(device)

base_model = load_models()

@st.cache_data
def get_embedding(text):
    with torch.no_grad():
        embedding = base_model.encode(text, convert_to_tensor=True).cpu().numpy()
    return embedding.tolist()

def query_pinecone(embedding, similarity_threshold=0.01):
    try:
        results = index.query(vector=embedding, top_k=20, include_metadata=True)
        filtered_results = [match['score'] for match in results['matches'] if match['score'] >= similarity_threshold]
        return filtered_results
    except Exception as e:
        st.sidebar.warning(f"Failed to query Pinecone: {e}. Proceeding without RAG scores.")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_response(patient_info, rag_scores):
    try:
        system_message = """You are an AI assistant for NHS GPs, designed to provide detailed analysis of patient conditions based on symptoms. Your responses should be structured, comprehensive, and medically accurate, while emphasizing the importance of clinical judgment and in-person examination."""

        user_prompt = f"""Please analyze the following patient case:

1. Patient Information:
{patient_info}

2. RAG Scores:
{rag_scores}

3. Analysis Structure:
   a) Possible Diagnosis: Provide potential diagnoses, explaining the reasoning behind each. Reference specific symptoms that support these diagnoses.
   b) Differential Diagnosis: Mention other conditions that might present similarly and explain why they are less likely
   c) Recommended Tests: Suggest diagnostic tests or examinations that would be helpful.
   d) Treatment Plan: Outline a treatment plan, including:
      - Medications (if applicable), with dosages and duration
      - Lifestyle modifications or self-care instructions
      - Follow-up recommendations
   e) Red Flags: Highlight any symptoms or factors that may indicate a more serious condition.
   f) Patient Education: Provide information about the condition(s) for patient education.
   g) ICD-10 Codes: Provide relevant ICD-10 codes for the potential diagnoses.

4. Important Notes:
   - If the symptoms are vague or insufficient for a confident diagnosis, clearly state this and recommend further evaluation.
   - Emphasize the importance of clinical judgment and the need for in-person examination.
   - If any critical information is missing, note what additional details would be helpful for a more accurate assessment.

Please provide your analysis in a clear, structured format, using medical terminology appropriately but also ensuring the content is understandable to GPs of varying experience levels."""

        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        return completion.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to generate response: {e}")
        return None

# Main app
st.title(st.secrets["app"]["name"])

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Information")
    symptoms = st.text_area("Please enter the patient's symptoms:")
    age = st.number_input("Patient's age:", min_value=0, max_value=120, value=30)
    gender = st.selectbox("Patient's gender:", ["Male", "Female", "Other"])
    duration = st.text_input("Duration of symptoms:")
    medical_history = st.text_area("Relevant medical history:")

    patient_info = f"""
Symptoms: {symptoms}
Age: {age}
Gender: {gender}
Duration of symptoms: {duration}
Medical history: {medical_history}
"""

    if st.button("Generate Diagnosis and Treatment Plan"):
        if symptoms:
            with st.spinner("Analyzing patient information..."):
                embedding = get_embedding(patient_info)
                rag_scores = query_pinecone(embedding)
                
                st.session_state.diagnosis = generate_response(patient_info, rag_scores)
                st.session_state.rag_scores = rag_scores
        else:
            st.warning("Please enter the patient's symptoms.")

with col2:
    if 'diagnosis' in st.session_state and st.session_state.diagnosis:
        st.subheader("Diagnosis and Treatment Plan")
        st.write(st.session_state.diagnosis)

        st.subheader("RAG Scores")
        if st.session_state.rag_scores:
            for i, score in enumerate(st.session_state.rag_scores, 1):
                st.write(f"Score {i}: {score:.4f}")
        else:
            st.write("No RAG scores available.")

        # Feedback mechanism
        feedback = st.radio("Was this diagnosis helpful?", ("Yes", "No"))
        if feedback == "No":
            improvement = st.text_area("Please provide feedback on how we can improve:")
            if st.button("Submit Feedback"):
                # Here you would typically save this feedback to a database
                st.success("Thank you for your feedback!")

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
