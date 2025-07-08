
# 🔁 EchoProof: Real-Time Echo Chamber Detector

![Made with Streamlit](https://img.shields.io/badge/Made%20with-Streamlit-1f6f8b.svg)
![License](https://img.shields.io/badge/license-MIT-green)

> **AI that detects whether a conversation is diverse or stuck in an echo chamber — live.**

---

## 🌐 Live App

🔗 [Click to Use EchoProof](https://p-echo-proof-snqqkhesa7dqe4dytuxjeh.streamlit.app/)

---

## 🚀 What It Does

EchoProof uses AI (sentence embeddings + cosine similarity) to analyze conversations and detect if everyone’s repeating similar opinions — a phenomenon known as an **echo chamber**.

### ✨ Features
- Paste any group chat, debate, or social media thread
- Detects semantic similarity using SentenceTransformers (MiniLM)
- Classifies the discussion as: Echo Chamber / Mild Echo / Diverse
- Clean, professional UI built with Streamlit
- Real-time similarity matrix

---

## 🧠 How It Works

1. Input conversation is split line-by-line  
2. Each line is embedded using `all-MiniLM-L6-v2`  
3. Cosine similarity matrix is computed  
4. Average similarity determines:
   - 🔴 Above 0.8 → **Echo Chamber**
   - 🟡 0.6–0.8 → **Mild Echo**
   - ✅ Below 0.6 → **Diverse Views**

---

## 🧪 Try Sample Input

```
AI is going to take all jobs.
Yes, machines are unstoppable now.
Exactly, no one can stop automation.
I agree, AI will dominate every field.
I saw GPT-4 winning a legal case!
```

Output: ⚠️ Echo Chamber Detected

---

## 📦 Tech Stack

- [Streamlit](https://streamlit.io/)
- [Sentence-Transformers](https://www.sbert.net/)
- [Transformers](https://huggingface.co/docs/transformers/)
- Python 3
- NumPy

---

## 🧾 License

This project is open source under the [MIT License](LICENSE).

---

> Built by [Pratyush Pateriya] | Patent-pending concept
