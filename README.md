# 📄 Smart Doc Search

AI-powered document search system that lets users upload documents, chunk and embed their content, store it in a vector database (Qdrant), and perform semantic search to retrieve the most relevant sections for any query.

---

## 🚀 Features

- 📂 Upload documents (PDF, TXT, etc.)
- ✂️ Smart chunking with user-defined size & overlap
- 🔎 Embedding using OpenAI models
- 📦 Vector storage in [Qdrant](https://qdrant.tech/)
- 🔍 Semantic search on user queries to fetch relevant chunks

---

## 🛠️ Tech Stack

- **Backend**: Golang
- **Vector DB**: Qdrant (gRPC-based integration)
- **Embeddings**: OpenAI `text-embedding-ada-002` (or pluggable local models)
- **UUID**: For uniquely identifying document chunks
- **gRPC**: For high-performance vector operations

---

## 💡 Use Cases

- Documentation or Knowledge Base search
- Internal company Q&A bots
- AI-powered SOP/manual assistants
- Legal, financial, or medical document search

---

## ⚙️ Setup Instructions

1. **Install ZUP**
   brew tap devglyph1/zup
   brew install zup
2. **Configure your environment**
   Add your OpenAI_API_KEY to a .env file or directly in code
3. **Quick start in local**
   zup run
