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

def query_pinecone(embedding, similarity_threshold=0.01):
    try:
        results = index.query(vector=embedding, top_k=20, include_metadata=True)
        filtered_results = [match for match in results['matches'] if match['score'] >= similarity_threshold]
        return filtered_results, results
    except Exception as e:
        st.sidebar.warning(f"Failed to query Pinecone: {e}. Proceeding without similar cases.")
        return [], None

def extract_relevant_info(similar_cases):
    relevant_info = ""
    for i, case in enumerate(similar_cases, 1):
        case_text = case.get('metadata', {}).get('text', 'No text available')
        relevant_info += f"Case {i} (Similarity: {case['score']:.4f}):\n{case_text}\n\n"
    return relevant_info

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_response(patient_info, similar_cases_info):
    try:
        system_message = """You are an AI assistant for NHS GPs, designed to provide detailed analysis of patient conditions based on symptoms and similar cases. Your responses should be structured, comprehensive, and medically accurate, while emphasizing the importance of clinical judgment and in-person examination. Always reference the provided similar cases in your analysis."""

        user_prompt = f"""Please analyze the following patient case and similar cases:

1. Patient Information:
{patient_info}

2. Similar Cases from Database:
{similar_cases_info}

3. Analysis Structure:
   a) Relevant Information from Similar Cases: Summarize key points from the similar cases that are relevant to this patient's symptoms. Reference specific case numbers.
   b) Possible Diagnosis: Provide potential diagnoses, explaining the reasoning behind each. Reference specific symptoms and similar cases that support these diagnoses.
   c) Differential Diagnosis: Mention other conditions that might present similarly and explain why they are less likely, referencing similar cases if applicable.
   d) Recommended Tests: Suggest diagnostic tests or examinations, referencing any tests mentioned in similar cases that were helpful.
   e) Treatment Plan: Outline a treatment plan, including:
      - Medications (if applicable), with dosages and duration
      - Lifestyle modifications or self-care instructions
      - Follow-up recommendations
   f) Red Flags: Highlight any symptoms or factors that may indicate a more serious condition, referencing similar cases if they showed any critical developments.
   g) Patient Education: Provide information about the condition(s) for patient education, incorporating any useful educational points from similar cases.
   h) ICD-10 Codes: Provide relevant ICD-10 codes for the potential diagnoses.

4. Important Notes:
   - Always reference the specific case numbers when using information from the similar cases.
   - If the symptoms are vague or insufficient for a confident diagnosis, clearly state this and recommend further evaluation.
   - Emphasize the importance of clinical judgment and the need for in-person examination.
   - If any critical information is missing, note what additional details would be helpful for a more accurate assessment.

Please provide your analysis in a clear, structured format, using medical terminology appropriately but also ensuring the content is understandable to GPs of varying experience levels."""

        completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": system_message,
                },
                {
                    "role": "user",
                    "content": user_prompt,
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

st.set_page_config(page_title="NHS GP Assistant", layout="wide")
st.title("NHS GP Assistant")

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
                embedding = get_expanded_embedding(patient_info)
                similar_cases, raw_results = query_pinecone(embedding)
                
                if not similar_cases:
                    similar_cases_info = "No similar cases available."
                else:
                    similar_cases_info = extract_relevant_info(similar_cases)
                
                st.session_state.diagnosis = generate_response(patient_info, similar_cases_info)
                st.session_state.debug_info = {
                    "similar_cases": similar_cases_info,
                    "raw_results": raw_results
                }
        else:
            st.warning("Please enter the patient's symptoms.")

with col2:
    if 'diagnosis' in st.session_state and st.session_state.diagnosis:
        st.subheader("Diagnosis and Treatment Plan")
        st.write(st.session_state.diagnosis)

        # Feedback mechanism
        feedback = st.radio("Was this diagnosis helpful?", ("Yes", "No"))
        if feedback == "No":
            improvement = st.text_area("Please provide feedback on how we can improve:")
            if st.button("Submit Feedback"):
                # Here you would typically save this feedback to a database
                st.success("Thank you for your feedback!")

# Debug information in an expander
with st.expander("Debug Information", expanded=False):
    if 'debug_info' in st.session_state:
        st.subheader("Similar Cases from Database:")
        st.write(st.session_state.debug_info["similar_cases"])
        
        st.subheader("Raw Pinecone Results:")
        st.write(st.session_state.debug_info["raw_results"])

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
