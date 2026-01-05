# 🤖 InitRAG - Intelligent Website Knowledge Base

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.x-green.svg)](https://www.llamaindex.ai/)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)

A production-ready RAG (Retrieval-Augmented Generation) chatbot that crawls and indexes entire websites, providing an intelligent chat interface powered by **100% free, local models** (Ollama + HuggingFace). Built for the **inIT Institute for Industrial IT** with 4,622 pages and 16,405 document chunks.

## ✨ Features

### 🚀 Core Capabilities
- **Recursive Website Crawling**: Intelligent domain exploration with depth control
- **Clean Content Extraction**: Uses `trafilatura` for noise-free text extraction
- **PDF Support**: Automatic PDF document processing and indexing
- **Local & Free**: Runs entirely on your machine using Ollama (no API costs!)
- **Fast Retrieval**: Vector-based similarity search with top-5 results
- **Source Citations**: Every answer includes clickable source URLs
- **Modern UI**: Clean ChatGPT-style interface with dark/light themes

### 🎨 Web Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Theme Switcher**: Beautiful dark and light modes with localStorage persistence
- **Real-time Chat**: Instant question answering with streaming-like feel
- **Suggestion Chips**: Quick-start questions for common queries
- **Copy to Clipboard**: Easy sharing of AI responses
- **Auto-scroll**: Smooth scrolling to latest messages

### ⚡ Performance
- **16,405 Document Chunks**: Comprehensive knowledge base from 4,622 pages
- **214MB Storage**: Efficient vector embeddings (141MB) + metadata (72MB)
- **Top-5 Retrieval**: Similarity cutoff at 0.5 for quality results
- **60-90s Startup**: One-time loading for instant subsequent queries

## 📋 Requirements

### System Requirements
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 500MB for models + 214MB for data
- **Ollama**: For local LLM inference

### Models Used
- **LLM**: Ollama Gemma2:2b (2.6B parameters, Q4_0 quantized)
- **Embeddings**: HuggingFace `all-MiniLM-L6-v2` (cached locally)
- **Alternatives**: Supports llama3.2:3b, gemma3:1b, or any Ollama model

## 🛠️ Installation

### 1. Install Ollama

Download and install Ollama from [ollama.ai](https://ollama.ai/):

```bash
# Pull the Gemma2 model (2.6B parameters, ~1.6GB)
ollama pull gemma2:2b

# Verify Ollama is running
ollama list
```

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/initrag.git
cd initrag
```

### 3. Create Virtual Environment

```bash
# Windows
python -m venv venv_new
venv_new\Scripts\activate

# Linux/Mac
python3 -m venv venv_new
source venv_new/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: Initial installation downloads the HuggingFace embedding model (~90MB) and caches it locally. Subsequent runs use the cached version.

### 5. Verify Installation

```bash
# Check Python packages
pip list | grep llama-index

# Verify Ollama connection
curl http://localhost:11434/api/tags
```

## 🚀 Quick Start

### Option 1: Use Existing Knowledge Base (Recommended)

If you have the pre-crawled `storage/` folder (214MB):

```bash
python web_ui.py
```

**Startup time**: 60-90 seconds (PyTorch/Transformers loading + 214MB JSON parsing)

### Option 2: Crawl a New Website

To crawl your own website:

```bash
# Edit main_free.py to change the target URL
python main_free.py
```

Default configuration:
- **Base URL**: `https://www.init-owl.de`
- **Max Depth**: 3 levels
- **Chunk Size**: 1024 tokens
- **Chunk Overlap**: 200 tokens

### Access the Web Interface

Once started, open your browser:

```
http://localhost:5000
```

You'll see a clean ChatGPT-style interface with:
- Welcome screen with suggestion chips
- Dark/Light theme toggle
- Real-time chat with source citations

## 💡 Usage Examples

### Example Questions

```
💬 Tell me about research projects at inIT
💬 Who are the professors at the institute?
💬 What are the current news and updates?
💬 How can I contact the inIT Institute?
💬 What publications are available from the research team?
```

### Sample Response

```
Question: Tell me about research projects

Answer: The inIT Institute focuses on Industrial IoT, 
Industry 4.0, and smart manufacturing systems. Current 
projects include intelligent production systems, 
cybersecurity in industrial environments, and data 
analytics for predictive maintenance...

Sources:
1. https://www.init-owl.de/research/projects/iot
2. https://www.init-owl.de/team/prof-lemgo
3. https://www.init-owl.de/publications/2025
```

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Web Crawler   │ ───> │  Content Extract │ ───> │  Vector Store   │
│  (Trafilatura)  │      │   (Clean Text)   │      │ (SimpleVector)  │
└─────────────────┘      └──────────────────┘      └─────────────────┘
                                                              │
                                                              ▼
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Flask Web UI  │ <─── │  Query Engine    │ <─── │  LlamaIndex     │
│  (ChatGPT-like) │      │  (Top-5 Retrieval│      │  (Orchestrator) │
└─────────────────┘      └──────────────────┘      └─────────────────┘
         │                         │
         ▼                         ▼
┌─────────────────┐      ┌──────────────────┐
│      User       │      │  Ollama LLM      │
│   (Browser)     │      │  (Gemma2:2b)     │
└─────────────────┘      └──────────────────┘
```

### Core Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Web Crawler** | `requests` + `BeautifulSoup` | Recursive domain crawling with depth control |
| **Content Extraction** | `trafilatura` | Clean text extraction from HTML |
| **PDF Processing** | `PDFReader` | Extract text from PDF documents |
| **Vector Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Convert text to 384-dim vectors |
| **Vector Store** | `SimpleVectorStore` (JSON) | Persist embeddings + metadata (214MB) |
| **RAG Orchestration** | `LlamaIndex 0.x` | Query engine with retrieval + generation |
| **LLM** | Ollama `gemma2:2b` | Local language model for answers |
| **Web Framework** | Flask 3.1.2 | REST API + HTML frontend |

### Data Flow

1. **Crawling Phase** (One-time):
   ```
   URL → Fetch HTML → Extract Text → Create Documents → Generate Embeddings → Store
   ```

2. **Query Phase** (Real-time):
   ```
   User Question → Embed Query → Vector Search (Top-5) → 
   Retrieve Context → LLM Generation → Format + Sources → Display
   ```

### Storage Structure

```
storage/
├── docstore.json              # 72MB - Document metadata + text
├── default__vector_store.json # 141MB - Vector embeddings
├── index_store.json           # 1.4MB - Index metadata
└── graph_store.json           # Empty (future use)
```

**Document Metadata**:
```json
{
  "id": "uuid-here",
  "text": "Full document text...",
  "metadata": {
    "url": "https://www.init-owl.de/page",
    "title": "Page Title",
    "content_type": "text/html",
    "depth": 2
  }
}
```

## 📁 Project Structure

```
initrag/
├── app.py                      # Core RAG system (InitRAGSystem class)
├── web_ui.py                   # Flask web server + ChatGPT-style UI
├── main_free.py                # Original crawler implementation
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── .gitignore                  # Git ignore patterns
│
├── .github/
│   └── copilot-instructions.md # GitHub Copilot custom instructions
│
├── storage/                    # Knowledge base (214MB, gitignored)
│   ├── docstore.json          # Document text + metadata (72MB)
│   ├── default__vector_store.json  # Vector embeddings (141MB)
│   ├── index_store.json       # Index metadata (1.4MB)
│   └── graph_store.json       # Graph relationships
│
├── model_cache/               # HuggingFace models (gitignored)
│   └── models--sentence-transformers--all-MiniLM-L6-v2/
│
├── venv_new/                  # Virtual environment (gitignored)
└── templates/                 # (Optional) Additional HTML templates
```

### Key Files

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | 453 | RAG system core: crawling, indexing, querying |
| `web_ui.py` | 526 | Flask server with clean ChatGPT-style interface |
| `main_free.py` | 542 | Standalone crawler with CLI interface |

## ⚙️ Configuration

### Model Selection

Edit `app.py` to change models:

```python
# Line 25 - Toggle between FREE (Ollama) or PAID (OpenAI)
USE_FREE_MODELS = True  # Set to False for OpenAI

# Line 293 - Choose Ollama model
Settings.llm = Ollama(
    model="gemma2:2b",        # Options: gemma3:1b, llama3.2:3b
    request_timeout=120.0,
    base_url="http://localhost:11434"
)

# Line 297 - Embedding model (cached locally)
Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    cache_folder="./model_cache"
)
```

### Crawler Settings

Customize in `main_free.py`:

```python
# Line 500+ - CrawlConfig
config = CrawlConfig(
    base_url="https://www.init-owl.de",
    max_depth=3,                    # Crawl depth (1-5)
    max_pages=5000,                 # Max pages to crawl
    delay=0.5,                      # Politeness delay (seconds)
    user_agent="InitRAG/1.0"        # User-Agent string
)

# Line 320+ - NodeParser settings
Settings.node_parser = SentenceSplitter(
    chunk_size=1024,                # Tokens per chunk
    chunk_overlap=200               # Overlap for context
)
```

### Web UI Customization

Edit `web_ui.py` for UI changes:

```python
# Line 30-447 - HTML_TEMPLATE (CSS + JavaScript)
# Modify colors, fonts, layout in the <style> section

# Line 300-330 - Dark theme colors
[data-theme="dark"] body {
    background: #343541;           # Change background
}

# Line 290-295 - Suggestion chips
<div class="suggestion" onclick="askQuestion('Your question')">
    <div class="suggestion-title">🔬 Your Topic</div>
</div>
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Empty Responses from AI

**Symptom**: Chatbot returns "Empty Response"

**Solution**: This was fixed in commit `09e9a76`. If you still see it:

```python
# Verify the fix in app.py line 413-432
if hasattr(response, 'response'):
    answer_text = response.response  # Correct way
else:
    answer_text = str(response)      # Fallback
```

#### 2. Slow Startup (60-90 seconds)

**Symptom**: Server takes a long time to start

**Cause**: PyTorch/Transformers imports + 214MB JSON loading

**Workarounds**:
- ✅ **Accept it**: One-time cost, subsequent queries are instant
- ⚠️ **Use smaller storage**: Reduce `max_pages` when crawling
- ❌ **ChromaDB migration**: Not compatible with current LlamaIndex version

#### 3. Ollama Connection Failed

**Symptom**: `❌ Ollama not available`

**Solutions**:
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama (if not running)
ollama serve

# Verify model is installed
ollama list
ollama pull gemma2:2b
```

#### 4. Port 5000 Already in Use

**Symptom**: `Address already in use`

**Solutions**:
```bash
# Find process using port 5000
netstat -ano | findstr :5000    # Windows
lsof -i :5000                   # Linux/Mac

# Kill the process
taskkill /PID <PID> /F          # Windows
kill -9 <PID>                   # Linux/Mac

# Or change port in web_ui.py line 540
app.run(host='0.0.0.0', port=5001)  # Use different port
```

#### 5. Module Not Found Errors

**Symptom**: `ModuleNotFoundError: No module named 'llama_index'`

**Solution**:
```bash
# Activate virtual environment first!
source venv_new/bin/activate    # Linux/Mac
venv_new\Scripts\activate       # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

## 📊 Performance Metrics

### Storage Statistics

| Metric | Value | Description |
|--------|-------|-------------|
| **Pages Crawled** | 4,622 | Unique URLs from init-owl.de |
| **Document Chunks** | 16,405 | 1024-token chunks with 200 overlap |
| **Total Storage** | 214MB | Vector store + metadata |
| **Vector Embeddings** | 141MB | 384-dim vectors per chunk |
| **Metadata** | 72MB | Document text + URL info |
| **Categories** | 4 | Research (32%), Other (65%), News (2%), Team (1%) |

### Query Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **Startup Time** | 60-90s | One-time PyTorch loading |
| **Query Time** | 2-5s | Vector search + LLM generation |
| **Retrieval** | Top-5 | Similarity cutoff 0.5 |
| **Context Window** | ~5,000 tokens | From top-5 chunks |
| **LLM Speed** | ~15 tokens/sec | Gemma2:2b on CPU |

### Resource Usage

| Resource | Idle | Query | Notes |
|----------|------|-------|-------|
| **RAM** | 2GB | 3-4GB | Includes model weights |
| **CPU** | <5% | 60-90% | During generation |
| **GPU** | N/A | N/A | CPU-only (Q4_0 quantization) |
| **Disk I/O** | Minimal | Minimal | Loaded into RAM |

## 🔒 Best Practices & Ethics

### Responsible Crawling

1. **Respect robots.txt**: Always check `/robots.txt` before crawling
2. **Rate Limiting**: Use delays (0.5-1s) to avoid overwhelming servers
3. **User-Agent**: Identify your bot clearly in requests
4. **Content Filtering**: Set minimum content length to avoid noise
5. **Domain Boundaries**: Strict domain filtering prevents external crawling

### Privacy & Security

- **No Personal Data**: Crawler avoids forms, login pages, and user data
- **Public Content Only**: Indexes publicly accessible pages
- **Local Processing**: All data stays on your machine (no cloud upload)
- **API Keys**: Never commit API keys to version control

### Legal Compliance

- ✅ **Terms of Service**: Review target website's TOS before crawling
- ✅ **Copyright**: Respect intellectual property rights
- ✅ **Fair Use**: Use for research, education, or personal projects
- ❌ **Commercial Use**: Requires explicit permission from content owners

## 🚀 Deployment

### Production Considerations

**Not recommended for production as-is**. This is a development server. For production:

1. **Use WSGI Server**:
   ```bash
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 web_ui:app
   ```

2. **Add SSL/TLS**:
   ```bash
   gunicorn --certfile cert.pem --keyfile key.pem web_ui:app
   ```

3. **Use Nginx Reverse Proxy**:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Optimize Storage**:
   - Consider PostgreSQL + pgvector for better scalability
   - Implement caching layer (Redis) for frequent queries
   - Use CDN for static assets

## 🤝 Contributing

Contributions are welcome! Here's how:

### Development Setup

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/initrag.git
cd initrag

# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
python web_ui.py

# Commit with descriptive message
git commit -m "Add: Description of your feature"

# Push and create PR
git push origin feature/your-feature-name
```

### Code Style

- **Python**: Follow PEP 8 guidelines
- **Comments**: Docstrings for all functions and classes
- **Type Hints**: Use type annotations where possible
- **Logging**: Use appropriate log levels (INFO, WARNING, ERROR)

### Pull Request Guidelines

1. **One feature per PR**: Keep changes focused
2. **Tests**: Add tests for new functionality (if applicable)
3. **Documentation**: Update README if adding features
4. **Clean commits**: Squash commits before submitting

## 📈 Roadmap

### Planned Features

- [ ] **Loading Progress Indicator**: Show "Loading 45%..." during startup
- [ ] **Export Chat History**: Download conversations as JSON/TXT
- [ ] **Query Statistics**: Display search time and source count
- [ ] **Mobile Responsive**: Better mobile UI experience
- [ ] **Markdown Rendering**: Support for formatted responses
- [ ] **Syntax Highlighting**: Code blocks with Prism.js
- [ ] **Math Rendering**: LaTeX support with KaTeX
- [ ] **Multi-language**: Support for non-English content
- [ ] **Incremental Crawling**: Update index with new pages only
- [ ] **Advanced Filters**: Filter by date, category, or content type

### Future Enhancements

- [ ] **Streaming Responses**: Real-time token generation
- [ ] **Multi-model Support**: Switch between Ollama models in UI
- [ ] **RAG Metrics**: Track hallucination, relevance, and accuracy
- [ ] **A/B Testing**: Compare different embedding models
- [ ] **Docker Support**: Containerized deployment
- [ ] **API Documentation**: OpenAPI/Swagger specs

## 📄 License

This project is licensed under the **MIT License** - see below for details:

```
MIT License

Copyright (c) 2026 InitRAG Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## 🙏 Acknowledgments

### Technologies

- **[LlamaIndex](https://www.llamaindex.ai/)**: RAG orchestration framework
- **[Ollama](https://ollama.ai/)**: Local LLM inference engine
- **[HuggingFace](https://huggingface.co/)**: Sentence transformers for embeddings
- **[Flask](https://flask.palletsprojects.com/)**: Web framework
- **[Trafilatura](https://trafilatura.readthedocs.io/)**: Content extraction
- **[BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)**: HTML parsing

### Models

- **Gemma2:2b**: Google's efficient small language model
- **all-MiniLM-L6-v2**: Sentence-transformers embedding model

### Inspiration

- **ChatGPT**: UI design inspiration
- **init-owl.de**: Test dataset (inIT Institute for Industrial IT)

## 📞 Support & Contact

### Getting Help

1. **Check Documentation**: Read this README thoroughly
2. **Search Issues**: Look for similar issues on GitHub
3. **Enable Debug Logging**: Set `logging.DEBUG` in code
4. **Check Logs**: Review console output and error messages

### Reporting Issues

When reporting bugs, include:
- Python version (`python --version`)
- OS and architecture (Windows/Linux/Mac, x64/ARM)
- Ollama version (`ollama --version`)
- Error messages and stack traces
- Steps to reproduce

### Feature Requests

Open a GitHub issue with:
- Clear description of the feature
- Use case and benefits
- Proposed implementation (optional)

---

<div align="center">

**Built with ❤️ using 100% free, local AI models**

**No API costs • No cloud dependencies • Full privacy**

[⭐ Star on GitHub](https://github.com/YOUR_USERNAME/initrag) • [🐛 Report Bug](https://github.com/YOUR_USERNAME/initrag/issues) • [💡 Request Feature](https://github.com/YOUR_USERNAME/initrag/issues)

</div>