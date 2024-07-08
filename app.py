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
    st.write(f"Debug: Expanded embedding shape: {expanded_embedding.shape}")
    return expanded_embedding.tolist()

def query_pinecone(embedding, similarity_threshold=0.05):
    try:
        results = index.query(vector=embedding, top_k=10, include_metadata=True)
        st.write(f"Debug: Raw Pinecone results: {results}")
        filtered_results = [match for match in results['matches'] if match['score'] >= similarity_threshold]
        st.write(f"Debug: Filtered results: {filtered_results}")
        return filtered_results
    except Exception as e:
        st.warning(f"Failed to query Pinecone: {e}. Proceeding without similar cases.")
        st.error(f"Detailed error: {str(e)}")
        return []

@st.cache_data(ttl=3600)  # Cache for 1 hour
def generate_response(patient_info, context, use_rag):
    try:
        system_message = """You are an AI assistant for NHS GPs, designed to provide detailed analysis of patient conditions based on symptoms and similar cases. Your responses should be structured, comprehensive, and medically accurate, while emphasizing the importance of clinical judgment and in-person examination."""

        user_prompt = f"""Please analyze the following patient case:

1. Patient Information:
{patient_info}

2. Similar Cases from Database:
{context}

3. Analysis Structure:
   a) Possible Diagnosis: Provide one or more potential diagnoses, explaining the reasoning behind each. If using RAG, reference specific similar cases that support these diagnoses.
   b) Differential Diagnosis: Briefly mention other conditions that might present similarly and explain why they are less likely.
   c) Recommended Tests: Suggest any diagnostic tests or examinations that could confirm or rule out the proposed diagnoses.
   d) Treatment Plan: Outline a comprehensive treatment plan, including:
      - Medications (if applicable), with dosages and duration
      - Lifestyle modifications or self-care instructions
      - Follow-up recommendations
   e) Red Flags: Highlight any symptoms or factors that may indicate a more serious condition requiring immediate attention.
   f) Patient Education: Provide brief, clear information about the condition(s) that the GP can use to educate the patient.
   g) ICD-10 Codes: Provide relevant ICD-10 codes for the potential diagnoses.

4. Important Notes:
   - {'Use the similar cases provided to inform your analysis.' if use_rag else 'No similar cases were found in the database. Base your analysis on general medical knowledge.'}
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

def check_pinecone_index():
    try:
        stats = index.describe_index_stats()
        st.sidebar.write(f"Pinecone index stats: {stats}")
    except Exception as e:
        st.sidebar.error(f"Failed to get Pinecone index stats: {e}")

st.title(st.secrets["app"]["name"])

# Check Pinecone index at the start
check_pinecone_index()

if 'diagnosis' not in st.session_state:
    st.session_state.diagnosis = None

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
        try:
            embedding = get_expanded_embedding(patient_info)
            similar_cases = query_pinecone(embedding)
            
            st.subheader("Similar Cases from Database:")
            if not similar_cases:
                st.write("No sufficiently similar cases found in the database.")
                context = "No similar cases available."
                use_rag = False
            else:
                context = "Similar cases:\n"
                for i, case in enumerate(similar_cases, 1):
                    st.write(f"Case {i} (Similarity: {case['score']:.2f}):")
                    st.write(case.get('metadata', {}).get('text', 'No text available'))
                    st.write("---")
                    context += f"Case {i}: " + case.get('metadata', {}).get('text', 'No text available') + "\n\n"
                use_rag = True
            
            st.session_state.diagnosis = generate_response(patient_info, context, use_rag)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter the patient's symptoms.")

if st.session_state.diagnosis:
    st.subheader("Diagnosis and Treatment Plan")
    st.write(st.session_state.diagnosis)

    # Feedback mechanism
    feedback = st.radio("Was this diagnosis helpful?", ("Yes", "No"))
    if feedback == "No":
        improvement = st.text_area("Please provide feedback on how we can improve:")
        if st.button("Submit Feedback"):
            # Here you would typically save this feedback to a database
            st.success("Thank you for your feedback!")

st.sidebar.warning("Note: This app is for educational purposes only. Always consult with a qualified medical professional for accurate diagnoses and treatment plans.")
