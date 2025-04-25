# BT5153_2025Grp13
# 🍽️ Yelp AI Restaurant Recommender (Scalable RAG Prototype)

A modular, multilingual Retrieval-Augmented Generation (RAG) system for recommending restaurants in Philadelphia using Yelp data. Built for **Google Colab**, with real-time conversational support, re-ranking options, and full traceability.

---

## 🧠 Objective

Build an LLM-powered system that:
- Retrieves meaningful content (reviews, tips, profiles)
- Ranks relevant restaurants
- Generates friendly and personalized recommendations

---

## 🔄 Retrieval Unit Design

Each row = 1 chunk of content, stored in a unified FAISS vector store.

**Chunk Types**:
- `profile`: Business metadata and description
- `tip`: Helpful user tips (filtered by compliments and length)
- `review`: Sampled user reviews (useful, recent, random)

**Each chunk includes**:
- `rag_text_chunk`
- `chunk_type`, `business_id`, `business_name`
- `stars`, `categories_list`, `price_range`, `latitude`, `longitude`, `detected_lang`
- `chunk_id = {business_id}_{chunk_type}_{n}`

---

## 🧹 Review & Tip Sampling

### Review Sampling
Up to **10 reviews** per business:
- 3 most useful
- 2 most recent
- 5 random (if available)

### Tip Sampling
Up to **5 tips** per business:
- Compliment count > 0
- Text length > 25 characters

> Sampling is done at ingestion for prototype; future versions may sample dynamically at retrieval.

---

## 📦 Chunking + Embedding

- Chunks are written in context-friendly phrasing:
  - `Profile`: "Craft Hall is a brewery with 4.0 stars..."
  - `Tip`: "Try the smoked wings, especially late-night."
  - `Review`: "The food was amazing and staff friendly..."
- **Embedding Model**: `bge-m3` ([Hugging Face link](https://huggingface.co/BAAI/bge-m3))
- Token cap: 512 tokens per chunk
- Vectors are normalized for cosine similarity

---

## 🧠 Vector Indexing

- **Indexing Framework**: FAISS `IndexFlatIP`
- **Embedding Dimensions**: 1024 (output of bge-m3)
- **Metadata** is stored separately for prompt construction, logging, and QA

Example metadata:
```
{
  "chunk_id": "abc123_review_0",
  "business_id": "abc123",
  "business_name": "Sakana",
  "chunk_type": "review",
  "stars": 4.5,
  "categories_list": ["Sushi", "Japanese"],
  "price_range": 2,
  "latitude": 39.95,
  "longitude": -75.16,
  "detected_lang": "en"
}
```

---

## 🔍 Retrieval + Re-ranking Flow

1. Embed user query using `bge-m3`
2. Retrieve top-K (default K=30) similar chunks from FAISS
3. *(Optional)* Re-rank with `bge-reranker-v2-m3`
4. Group by `business_id`
5. Score businesses using:
   - Number of matched chunks
   - Weighted chunk types (`review > tip > profile`)
   - Similarity or rerank score

---

## 🧩 Prompt Assembly

- Select **Top 3 restaurants**, up to **3 chunks each**
- Assemble into language-specific prompt (~3500 token cap)

Example format:
```
### Green Eggs Café
- Review: "Hands down the best French toast."
- Tip: Go early on weekends.
- Profile: Popular brunch cafe with vegan options.
```

---

## 💬 LLM Generation Layer

| Model                    | Parameters | Notes                                       |
|-------------------------|------------|---------------------------------------------|
| Qwen2.5-1.5B-Instruct   | 1.5B       | ✅ Primary model (friendly, multilingual)    |
| Qwen2.5-0.5B-Instruct   | 0.5B       | 🔁 Fallback (lightweight, fast)              |

Toggle between models:
```
USE_QWEN2_5_MAIN = True  # False to use Qwen2.5-0.5B
```

---

## 📤 Post-Generation Handling

- Use raw LLM output (no parsing)
- Soft filter: Remove hallucinated or duplicate restaurants
- Traceability Logging includes:
  - `user_query`, `user_query_lang`
  - `selected_business_ids`, `chunk_ids_in_prompt`
  - `llm_response_text`, `model_used`, `response_time_ms`

---

## ⚙️ Tools + Runtime

| Component      | Technology                         |
|----------------|-------------------------------------|
| Embedding      | `bge-m3` (Hugging Face)             |
| Vector Store   | FAISS `IndexFlatIP`                |
| Re-ranker      | `bge-reranker-v2-m3`               |
| LLMs           | Qwen2.5-1.5B and 0.5B Instruct     |
| Environment    | Google Colab (T4 GPU, 16GB VRAM)   |
| Dataset        | Yelp Open Dataset (Philadelphia)   |

---

## 🚀 Setup + Example Commands

```python
# Install dependencies
!pip install -q faiss-cpu sentence-transformers

# Load embedding model
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("BAAI/bge-m3", trust_remote_code=True).to("cuda")

# Encode chunks
embeddings = model.encode(texts, normalize_embeddings=True, batch_size=256, show_progress_bar=True)

# Build FAISS index
import faiss
index = faiss.IndexFlatIP(1024)
index.add(embeddings)

# Save/load index
faiss.write_index(index, "faiss_index_bge_m3.index")
index = faiss.read_index("faiss_index_bge_m3.index")
```

---

## 📂 Suggested Project Structure

```
project/
├── BT5153 Project Yelp.ipynb
├── chunk_embeddings.npy
├── faiss_index_bge_m3.index
├── philly_chunks.parquet
├── philly_restaurants.csv
├── philly_restaurants_cleaned.parquet
├── philly_reviews_sampled.csv
├── philly_tips_sampled.csv
├── philly_restaurants_heatmap.html
├── philly_restaurants_map.html
└── README.md
```
Due to file size limitations in GitHub, chunk_embeddings.npy and faiss_index_bge_m3.index could not be uploaded. These two files have to be regenerated in Colab prior to running the RAG.
---

## 📋 License & Attribution

- Yelp Open Dataset (https://www.yelp.com/dataset)
- Embedding models by [BAAI](https://huggingface.co/BAAI)
- Qwen models by [Qwen](https://huggingface.co/Qwen)
