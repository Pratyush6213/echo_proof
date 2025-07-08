import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import torch

@st.cache_resource
def load_counter_model():
    return pipeline(
        "text-generation",
        model="mistralai/Mistral-7B-Instruct-v0.3",
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

counter_llm = load_counter_model()

def get_counter_argument(convo_lines):
    prompt = f"""The following conversation may be an echo chamber:
{"\n".join(convo_lines)}

Suggest a balanced, respectful counter-argument to diversify the discussion."""
    result = counter_llm(prompt, max_new_tokens=100, temperature=0.7)
    return result[0]["generated_text"].strip()

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

st.title("🔁 EchoProof: Echo Chamber Detector")
st.write("Paste a group conversation to analyze for echo chamber patterns.")

user_input = st.text_area("🗣️ Paste the conversation (one message per line):", height=200)

if user_input:
    messages = [line.strip() for line in user_input.split("\n") if line.strip()]
    embeddings = model.encode(messages, convert_to_tensor=True)
    cosine_sim_matrix = util.pytorch_cos_sim(embeddings, embeddings).cpu().numpy()

    avg_similarity = np.mean([
        cosine_sim_matrix[i][j]
        for i in range(len(messages))
        for j in range(len(messages))
        if i != j
    ])

    st.write("### 🔍 Average Semantic Similarity:", round(avg_similarity, 3))

    if avg_similarity >= 0.8:
        st.error("⚠️ Echo Chamber Detected! Very similar viewpoints.")
        st.subheader("🗣️ Suggested Counter-Opinion (AI)")
        with st.spinner("Generating counter-opinion..."):
            counter = get_counter_argument(messages)
            st.markdown(counter)
    elif avg_similarity >= 0.6:
        st.warning("🟡 Mild Echo Detected. Consider including diverse opinions.")
    else:
        st.success("✅ Diverse Discussion! Multiple viewpoints present.")

    st.write("### 🔬 Similarity Matrix")
    st.dataframe(cosine_sim_matrix, use_container_width=True)