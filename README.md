# InitRAG - Advanced LlamaIndex Website Crawler

A sophisticated RAG (Retrieval-Augmented Generation) application that recursively crawls and indexes the entire **init-owl.de** website using LlamaIndex, providing an intelligent chat interface for querying website content.

## 🚀 Features

- **Recursive Website Crawling**: Custom crawler that intelligently explores entire domains
- **Clean Content Extraction**: Uses `trafilatura` for noise-free text extraction
- **PDF Processing**: Handles PDF documents with LlamaParse/PDFReader
- **Persistent Vector Storage**: ChromaDB-backed storage for fast retrieval
- **Interactive Chat**: AI-powered chat interface with source citations
- **Domain Filtering**: Smart filtering to stay within target domains
- **Politeness Controls**: Respects server resources with configurable delays

## 📋 Requirements

- Python 3.8 or higher
- OpenAI API key
- (Optional) LlamaCloud API key for enhanced PDF processing

## 🛠️ Installation

1. **Clone or navigate to the project directory**:
   ```bash
   cd initrag
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   LLAMA_CLOUD_API_KEY=your_llama_cloud_api_key_here  # Optional
   ```

## 🚀 Usage

### Basic Usage

Run the application:
```bash
python main.py
```

The application will:
1. Check for existing index in `./storage_init_full`
2. If no index exists, crawl the entire init-owl.de website
3. Create a persistent vector index
4. Launch an interactive chat interface

### Configuration Options

You can customize the crawler behavior by editing the environment variables in `.env`:

```env
# Crawler settings
MAX_DEPTH=3                    # Maximum crawl depth
CRAWL_DELAY=0.5               # Delay between requests (seconds)
MAX_PAGES_PER_DOMAIN=1000     # Maximum pages to crawl

# Storage settings
STORAGE_DIR=./storage_init_full
CHUNK_SIZE=1024               # Text chunk size for indexing
CHUNK_OVERLAP=200             # Overlap between chunks
```

### Example Chat Session

```
🤖 InitRAG - inIT Institute AI Assistant
============================================================
Ask questions about the inIT Institute in Lemgo!
Type 'quit' or 'exit' to end the conversation.
============================================================

💬 You: What research areas does the inIT Institute focus on?

🤖 AI Assistant: Based on the information from the init-owl.de website, 
the inIT Institute focuses on several key research areas including:

1. Industrial IoT and Industry 4.0 technologies
2. Software engineering and system integration
3. Data analytics and artificial intelligence
4. Cybersecurity for industrial systems

This information was found at: https://www.init-owl.de/research/areas
```

## 🏗️ Architecture

### Core Components

1. **WebsiteCrawler**: Handles recursive crawling with domain filtering
2. **Content Extraction**: Uses trafilatura for clean text extraction
3. **PDF Processing**: LlamaParse/PDFReader for PDF content
4. **Vector Indexing**: LlamaIndex with OpenAI embeddings
5. **Chat Engine**: Context-aware conversational interface

### Crawling Strategy

```python
def crawl_domain(base_url: str, max_depth: int = 3) -> List[Document]:
    """
    Recursively crawls domain with:
    - Domain filtering (only init-owl.de)
    - File type filtering (HTML, PDF allowed)
    - Politeness delays
    - Content extraction
    """
```

### Content Processing Pipeline

```
URL → Fetch → Extract Clean Text → Create Document → Index → Store
```

## 📁 Project Structure

```
initrag/
├── main.py                 # Main application entry point
├── requirements.txt        # Python dependencies
├── .env.example           # Environment variable template
├── .env                   # Your environment variables (create this)
├── .github/
│   └── copilot-instructions.md
├── storage_init_full/     # Persistent vector storage (created on first run)
└── crawler.log           # Crawler activity logs
```

## 🔧 Advanced Configuration

### Custom Domain Crawling

To crawl a different domain, modify the configuration in `main.py`:

```python
config = CrawlConfig(
    base_url="https://your-domain.com",
    max_depth=3,
    max_pages=1000,
    delay=0.5
)
```

### File Type Filtering

Customize allowed/blocked file extensions:

```python
config.allowed_extensions = {'.html', '.htm', '.php', '.pdf'}
config.blocked_extensions = {'.jpg', '.css', '.js', '.zip'}
```

## 📊 Monitoring and Logging

The application provides comprehensive logging:

- **Console Output**: Real-time crawling progress
- **Log File**: Detailed logs in `crawler.log`
- **Statistics**: Pages crawled, documents created, failed URLs

## 🚨 Error Handling

The application includes robust error handling for:
- Network timeouts and connection errors
- Invalid URLs and broken links
- Content extraction failures
- PDF processing errors
- Storage and indexing issues

## 🔒 Best Practices

1. **Respect Robots.txt**: The crawler respects server policies
2. **Rate Limiting**: Configurable delays prevent server overload
3. **Content Quality**: Minimum content length filtering
4. **Domain Security**: Strict domain filtering prevents crawling external sites
5. **Resource Management**: Automatic cleanup of temporary files

## 🛠️ Troubleshooting

### Common Issues

1. **OpenAI API Key Error**:
   ```
   Error: OPENAI_API_KEY environment variable is required
   ```
   Solution: Add your OpenAI API key to the `.env` file

2. **No Content Extracted**:
   - Check if the target website blocks crawlers
   - Verify the domain is accessible
   - Check crawler logs for specific errors

3. **PDF Processing Issues**:
   - Ensure LlamaCloud API key is set for enhanced PDF processing
   - Check if PDFs are accessible and not password-protected

### Debug Mode

Enable verbose logging by modifying the logging level:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Optimization

- **Chunk Size**: Adjust `CHUNK_SIZE` for your content type
- **Batch Size**: Modify crawl delay for faster/slower crawling
- **Storage**: Use SSD storage for better index performance
- **Memory**: Larger RAM helps with processing many documents

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add comprehensive tests
4. Submit a pull request

## 📄 License

This project is for educational and research purposes. Please respect website terms of service when crawling.

## 🆘 Support

For issues and questions:
1. Check the logs in `crawler.log`
2. Review the troubleshooting section
3. Ensure all dependencies are correctly installed
4. Verify API keys are properly configured

---

**Note**: This application is designed to crawl public websites responsibly. Always respect robots.txt files and website terms of service.