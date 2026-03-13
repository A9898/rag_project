# 🔍 RAG Locale

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Ollama](https://img.shields.io/badge/LLM-Ollama-black)
![FastAPI](https://img.shields.io/badge/API-FastAPI-green)
![ChromaDB](https://img.shields.io/badge/VectorDB-ChromaDB-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**RAG (Retrieval-Augmented Generation) completamente locale** costruito con **Ollama + ChromaDB**.

Carica i tuoi documenti e fai domande usando un **LLM locale**.  
Tutto gira **sulla tua macchina** — **nessun dato lascia il tuo computer**.

---

# ✨ Features

- 🔒 **100% locale** — nessuna API esterna
- 📄 Supporto per più formati di documenti
- 🧠 **Embeddings locali** con Ollama
- 🔎 **Vector search** con ChromaDB
- 💬 Chat con contesto dai documenti
- ⚡ **Streaming delle risposte**
- 📚 Citazione delle **fonti rilevanti**
- 🧾 Database persistente

---

# 🧰 Prerequisiti

Assicurati di avere installato:

- **Python 3.10+**
- **Ollama**

Installa Ollama da:

https://ollama.ai

Avvia il server:

```bash
ollama serve
```

Scarica i modelli:

```bash
ollama pull mistral
ollama pull nomic-embed-text
```

---

# ⚡ Setup rapido

Clona il repository:

```bash
git clone https://github.com/tuo-username/rag-locale.git
cd rag-locale
```

Installa le dipendenze:

```bash
pip install -r requirements.txt
```

Assicurati che Ollama sia attivo:

```bash
ollama serve
```

Avvia il server:

```bash
python server.py
```

Apri il browser:

```
http://localhost:8000
```

---

# 🧠 Come funziona

1. **Carica un documento** dalla sidebar  
2. Il testo viene:
   - estratto
   - suddiviso in **chunk**
   - convertito in **embeddings**
3. Gli embeddings vengono salvati in **ChromaDB**
4. Quando fai una domanda:
   - il sistema trova i **chunk più rilevanti**
   - li passa come **contesto al modello LLM**
5. Il modello genera una risposta **basata sui documenti**

---

# ⚙️ Configurazione

Variabili d'ambiente opzionali:

| Variabile | Default | Descrizione |
|-----------|--------|-------------|
| `OLLAMA_BASE` | `http://localhost:11434` | URL server Ollama |
| `EMBED_MODEL` | `nomic-embed-text` | Modello embeddings |
| `CHAT_MODEL` | `mistral` | Modello LLM |
| `CHUNK_SIZE` | `800` | Dimensione chunk |
| `CHUNK_OVERLAP` | `200` | Sovrapposizione chunk |
| `TOP_K` | `5` | Chunk recuperati |

Esempio:

```bash
CHAT_MODEL=llama3 EMBED_MODEL=nomic-embed-text python server.py
```

---

# 📁 Struttura del progetto

```
rag-locale/
│
├── server.py
│
├── static/
│   └── index.html
│
├── requirements.txt
│
├── chroma_data/        # database vettoriale (auto-generato)
│
└── uploads/            # documenti caricati (auto-generato)
```

---

# 📄 Formati supportati

- 📄 PDF  
- 📝 DOCX  
- 📃 TXT / MD  
- 📊 CSV  
- 🌐 HTML / XML  
- 🔧 JSON  

---

# 📝 Note

- Il database **ChromaDB è persistente**
- I documenti restano indicizzati tra i riavvii
- La chat mantiene **gli ultimi 5 scambi** per il contesto
- Se `nomic-embed-text` non è disponibile puoi usare:

```bash
ollama pull all-minilm
```

---

# 🚀 Possibili miglioramenti

- Supporto **multi-document retrieval**
- **UI migliorata**
- **Reranking dei risultati**
- **Supporto immagini e tabelle**
- **Docker deployment**

---

# 📜 Licenza

MIT License

---

# ⭐ Se ti è utile

Lascia una **stella al repository** ⭐
