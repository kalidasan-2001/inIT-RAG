# InitRAG - Website Knowledge Base with Local LLM

A RAG (Retrieval-Augmented Generation) chatbot that crawls websites and lets you query the content using local AI models. Uses Ollama for the LLM and HuggingFace for embeddings - completely free, runs offline.

Currently indexes the inIT Institute website (init-owl.de): 4,622 pages, 16,405 text chunks, 214MB total storage.

## Features

- Crawls websites recursively with depth control
- Extracts clean text from HTML using trafilatura
- Processes PDF documents automatically
- Stores content as vector embeddings for semantic search
- Web UI with chat interface (Flask)
- Dark/light theme toggle
- Shows source URLs for every answer
- Runs entirely offline after initial model download

## What You Need

- **Python 3.11+** (tested on 3.11, might work on 3.8+)
- **8GB RAM minimum** (16GB recommended for larger datasets)
- **Ollama** installed and running
- **~2GB disk space** for models and data

## Installation

### 1. Install Ollama

Download from https://ollama.ai and install it.

```bash
# Pull the model (1.6GB download)
ollama pull gemma2:2b

# Verify it works
ollama list
```

### 2. Clone and Setup

```bash
git clone https://github.com/kalidasan-2001/inIT-RAG.git
cd inIT-RAG

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

First run downloads the HuggingFace embedding model (~90MB) and caches it locally.

### 3. Run the Application

```bash
python web_ui.py
```

**Startup takes 60-90 seconds** because it loads PyTorch, Transformers, and the 214MB storage JSON files. This is a one-time delay - subsequent queries are fast (2-5 seconds).

Open http://localhost:5000 in your browser.

## Usage

### Ask Questions

The web interface has 4 example questions to get started:
- "Tell me about research projects"
- "Who are the professors?"
- "What are the current news?"
- "How can I contact the institute?"

Or type your own question. The AI will search the indexed pages and generate an answer with source links.

### Crawl a Different Website

Edit `main_free.py` and change:

```python
config = CrawlConfig(
    base_url="https://your-website.com",  # Change this
    max_depth=3,        # How deep to crawl (1-5 recommended)
    max_pages=5000,     # Stop after this many pages
    delay=0.5           # Seconds between requests
)
```

Then run: `python main_free.py`

This will create a new `storage/` folder with the indexed content.

## How It Works

```
1. Web Crawler
   ↓ Fetches HTML pages recursively
2. Content Extractor (trafilatura)
   ↓ Cleans text, removes navigation/ads
3. Text Chunker
   ↓ Splits into 1024-token chunks with 200 overlap
4. Embedding Model (HuggingFace)
   ↓ Converts text to 384-dimensional vectors
5. Vector Store (JSON files)
   ↓ Stores embeddings + metadata (214MB)
6. Query Engine (LlamaIndex)
   ↓ Retrieves top-5 relevant chunks
7. LLM (Ollama Gemma2)
   ↓ Generates answer from context
8. Web UI (Flask)
   ↓ Displays response with sources
```

### Files

- `app.py` (453 lines) - Core RAG system, handles indexing and queries
- `web_ui.py` (526 lines) - Flask server and HTML/CSS/JS for the chat UI
- `main_free.py` (542 lines) - Standalone website crawler
- `requirements.txt` - Python dependencies
- `storage/` - Vector embeddings and documents (not in git, 214MB)

## Configuration

### Change the LLM Model

Edit `app.py` line 293:

```python
Settings.llm = Ollama(
    model="gemma2:2b",  # Change to: llama3.2:3b, gemma3:1b, etc.
    request_timeout=120.0,
    base_url="http://localhost:11434"
)
```

Available models: https://ollama.ai/library

### Change Crawler Settings

Edit `main_free.py` around line 500:

```python
config = CrawlConfig(
    base_url="https://www.init-owl.de",
    max_depth=3,        # How many link levels deep to go
    max_pages=5000,     # Stop after crawling this many pages
    delay=0.5,          # Seconds to wait between requests
)
```

### Change Chunk Settings

Edit `app.py` line 320:

```python
Settings.node_parser = SentenceSplitter(
    chunk_size=1024,     # Tokens per chunk
    chunk_overlap=200    # Overlap between chunks
)
```

Smaller chunks = more precise retrieval but less context. Larger chunks = opposite.

## Common Problems

### "Empty Response" from the chatbot

Fixed in the latest version. If you still see it, the issue is in `app.py` line 413-432. The response object needs `.response` attribute accessed, not just `str(response)`.

### Server won't start / Port 5000 in use

```bash
# Find what's using port 5000
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # Linux/Mac

# Kill it or change the port in web_ui.py line 540
app.run(host='0.0.0.0', port=5001)
```

### Ollama not connecting

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# If not, start it
ollama serve

# Make sure you pulled the model
ollama pull gemma2:2b
```

### ModuleNotFoundError

Make sure you activated the virtual environment:

```bash
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

pip install -r requirements.txt
```

### Slow startup (60-90 seconds)

This is normal. The system loads:
- PyTorch and Transformers libraries (~30-40 seconds)
- 214MB of JSON storage files (~30-40 seconds)

This happens once when the server starts. Queries after that are 2-5 seconds.

## Performance Notes

**Current Dataset (init-owl.de):**
- 4,622 web pages crawled
- 16,405 text chunks (1024 tokens each)
- 214MB total storage (141MB vectors + 72MB text)

**Timing:**
- Startup: 60-90 seconds (one time)
- Query: 2-5 seconds (vector search + LLM generation)
- Retrieves top 5 most relevant chunks per query

**Resource Usage:**
- RAM: ~2-3GB when idle, ~4GB during queries
- CPU: Spikes to 60-90% during answer generation
- GPU: Not used (runs on CPU with Q4 quantization)

## Known Limitations

- Not suitable for production - uses Flask development server
- Startup is slow due to PyTorch imports and large JSON files
- No incremental updates - must re-crawl entire site for new content
- UI is basic - no markdown rendering, syntax highlighting, or math support
- Query responses can be slow (2-5 seconds)
- No conversation history or follow-up questions
- Gemma2:2b sometimes gives generic answers for complex questions

## License

MIT License - see the repository for full text.

## Credits

Built with:
- LlamaIndex for RAG orchestration
- Ollama for local LLM inference
- HuggingFace for embeddings
- Flask for the web server
- Trafilatura for content extraction