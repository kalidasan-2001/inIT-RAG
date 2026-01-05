"""
InitRAG - Core RAG System for inIT Institute

RAG (Retrieval-Augmented Generation) system for the inIT Institute website.
Built with LlamaIndex for document indexing and retrieval.

Author: Senior Python AI Engineer
Created: November 24, 2025
"""

import os
import time
import logging
import requests
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Set, Optional, Dict
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# LlamaIndex imports
from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext, 
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.postprocessor import SimilarityPostprocessor

try:
    from llama_index.vector_stores.chroma import ChromaVectorStore
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False

# Choose your models - FREE (Ollama + HuggingFace) or PAID (OpenAI)
USE_FREE_MODELS = True  # Set to False to use OpenAI models

if USE_FREE_MODELS:
    from llama_index.llms.ollama import Ollama
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
else:
    from llama_index.llms.openai import OpenAI
    from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index.readers.file import PDFReader

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs for faster loading
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress verbose library logs
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('llama_index').setLevel(logging.WARNING)
logging.getLogger('transformers').setLevel(logging.ERROR)

# Load environment variables
load_dotenv()

@dataclass
class CrawlConfig:
    """Configuration for the web crawler."""
    base_url: str = "https://www.init-owl.de"
    max_depth: int = 3
    max_pages: int = 1000
    delay: float = 0.5
    timeout: int = 30
    user_agent: str = "InitRAG Bot 1.0 (Educational Research)"
    allowed_extensions: Set[str] = None
    blocked_extensions: Set[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'.html', '.htm', '.php', '.asp', '.aspx', '.jsp', '.pdf'}
        if self.blocked_extensions is None:
            self.blocked_extensions = {
                '.jpg', '.jpeg', '.png', '.gif', '.svg', '.ico', 
                '.css', '.js', '.zip', '.rar', '.tar', '.gz',
                '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.flv',
                '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx'
            }

class WebsiteCrawler:
    """Advanced website crawler with domain filtering and content extraction."""
    
    def __init__(self, config: CrawlConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': config.user_agent,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })
        
        self.visited_urls: Set[str] = set()
        self.failed_urls: Set[str] = set()
        self.documents: List[Document] = []
        self.pdf_reader = PDFReader()
        
        # Parse base domain
        parsed_url = urlparse(config.base_url)
        self.base_domain = parsed_url.netloc.lower()
        self.allowed_domains = {self.base_domain, f"www.{self.base_domain.replace('www.', '')}"}
        
        logger.info(f"Initialized crawler for domain: {self.base_domain}")
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled based on domain and file extension."""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if not any(allowed in domain for allowed in self.allowed_domains):
                return False
            
            path = parsed.path.lower()
            if path.endswith('/'):
                return True
                
            ext = os.path.splitext(path)[1]
            if ext and ext in self.config.blocked_extensions:
                return False
                
            if ext and ext not in self.config.allowed_extensions:
                return False
                
            return True
        except Exception:
            return False
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments."""
        parsed = urlparse(url)
        return urlunparse((
            parsed.scheme, parsed.netloc, parsed.path,
            parsed.params, parsed.query, ''
        ))
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all valid links from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href or href.startswith('#'):
                    continue
                    
                absolute_url = urljoin(base_url, href)
                normalized_url = self.normalize_url(absolute_url)
                
                if self.is_valid_url(normalized_url):
                    links.append(normalized_url)
            
            return list(set(links))
        except Exception:
            return []
    
    def extract_content(self, html_content: str, url: str) -> Optional[str]:
        """Extract clean text content using trafilatura."""
        try:
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                url=url
            )
            
            if extracted and len(extracted.strip()) > 100:
                return extracted.strip()
            return None
        except Exception:
            return None
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch and return page content."""
        try:
            response = self.session.get(url, timeout=self.config.timeout, allow_redirects=True)
            response.raise_for_status()
            return response.text
        except Exception:
            self.failed_urls.add(url)
            return None
    
    def extract_title(self, html_content: str) -> str:
        """Extract page title from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            title_tag = soup.find('title')
            if title_tag and title_tag.string:
                return title_tag.string.strip()
        except:
            pass
        return "Unknown Title"
    
    def crawl_page(self, url: str, depth: int = 0) -> List[str]:
        """Crawl a single page and return found links."""
        if (url in self.visited_urls or 
            depth > self.config.max_depth or 
            len(self.visited_urls) >= self.config.max_pages):
            return []
        
        self.visited_urls.add(url)
        logger.info(f"Crawling (depth {depth}): {url}")
        
        time.sleep(self.config.delay)
        
        html_content = self.fetch_page(url)
        if not html_content:
            return []
        
        text_content = self.extract_content(html_content, url)
        if text_content:
            document = Document(
                text=text_content,
                metadata={
                    'url': url,
                    'title': self.extract_title(html_content),
                    'content_type': 'html',
                    'depth': depth
                }
            )
            self.documents.append(document)
            logger.info(f"Successfully extracted content from: {url}")
        
        return self.extract_links(html_content, url)
    
    
    def crawl_domain(self, base_url: str, max_depth: int = 3) -> List[Document]:
        """Crawl domain starting from base_url."""
        logger.info(f"Starting domain crawl from: {base_url}")
        
        self.config.max_depth = max_depth
        urls_to_visit = [base_url]
        current_depth = 0
        
        while urls_to_visit and current_depth <= max_depth:
            logger.info(f"Depth Level {current_depth}: Processing {len(urls_to_visit)} URLs...")
            
            next_level_urls = []
            
            for url in urls_to_visit:
                if len(self.visited_urls) >= self.config.max_pages:
                    break
                    
                found_links = self.crawl_page(url, current_depth)
                
                for link in found_links:
                    if link not in self.visited_urls:
                        next_level_urls.append(link)
            
            urls_to_visit = list(set(next_level_urls))
            current_depth += 1
        
        logger.info(f"Crawling completed! Documents: {len(self.documents)}, Pages: {len(self.visited_urls)}, Failed: {len(self.failed_urls)}")
        return self.documents



class InitRAGSystem:
    """Main RAG system - uses simple JSON storage (compatible)."""
    
    def __init__(self, storage_dir: str = "./storage"):
        self.storage_dir = Path(storage_dir)
        self.index = None
        self.query_engine = None
        self.setup_llama_index()
    
    def setup_llama_index(self):
        """Configure LlamaIndex settings with model selection."""
        if USE_FREE_MODELS:
            # FREE: Ollama + HuggingFace
            Settings.llm = Ollama(
                model="gemma2:2b",
                request_timeout=120.0,
                base_url="http://localhost:11434"
            )
            # Use cached embedding model (no internet download)
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                trust_remote_code=True,
                cache_folder="./model_cache"
            )
            logger.warning("✅ Using FREE models (Ollama + HuggingFace)")
        else:
            # PAID: OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY required for paid models")
            
            Settings.llm = OpenAI(
                model="gpt-3.5-turbo",
                temperature=0.1,
                api_key=api_key
            )
            Settings.embed_model = OpenAIEmbedding(
                model="text-embedding-ada-002",
                api_key=api_key
            )
            logger.warning("✅ Using PAID models (OpenAI)")
        
        # Configure node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=1024,
            chunk_overlap=200
        )
    
    def check_ollama_status(self) -> bool:
        """Check if Ollama is available (for free models)."""
        if not USE_FREE_MODELS:
            return True
            
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get("models", [])
            model_names = [model["name"] for model in models]
            
            if "gemma2:2b" in model_names or "llama3.2:3b" in model_names:
                return True
            else:
                logger.error("Required Ollama model not found")
                return False
        except Exception as e:
            logger.error(f"Ollama not available: {e}")
            return False
    
    def load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index - streaming JSON parsing for speed."""
        if self.storage_dir.exists() and (self.storage_dir / "docstore.json").exists():
            try:
                logger.warning("📖 Loading existing knowledge base (this may take 30-60 seconds)...")
                
                # Use fast loading with streaming
                storage_context = StorageContext.from_defaults(
                    persist_dir=str(self.storage_dir)
                )
                self.index = load_index_from_storage(storage_context)
                
                logger.warning("✅ Loaded successfully!")
                return self.index
            except Exception as e:
                logger.error(f"Error loading index: {e}")
                import traceback
                traceback.print_exc()
        
        logger.warning("Creating new index by crawling...")
        return self.create_new_index()
    def create_new_index(self) -> VectorStoreIndex:
        """Create new index by crawling the website."""
        config = CrawlConfig(
            base_url="https://www.init-owl.de",
            max_depth=3,
            max_pages=1000,
            delay=0.5
        )
        
        crawler = WebsiteCrawler(config)
        documents = crawler.crawl_domain(config.base_url, config.max_depth)
        
        if not documents:
            raise ValueError("No documents were extracted from the website")
        
        logger.warning("Building ChromaDB vector index...")
        
        # Create ChromaDB storage
        chroma_collection = self.chroma_client.get_or_create_collection("init_docs")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        self.index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        
        logger.warning(f"✅ Index created with {len(documents)} documents!")
        return self.index
    
    def create_query_engine(self):
        """Create query engine with source citation."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5
        )
        
        self.query_engine = RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=[
                SimilarityPostprocessor(similarity_cutoff=0.5)
            ]
        )
        
        logger.warning("Query engine created successfully")
    
    def query(self, question: str) -> Dict[str, any]:
        """Execute query and return response with sources."""
        if not self.query_engine:
            raise ValueError("Query engine not initialized")
        
        response = self.query_engine.query(question)
        
        # Extract answer text (handle both str and Response objects)
        if hasattr(response, 'response'):
            answer_text = response.response
        else:
            answer_text = str(response)
        
        # Extract source URLs
        source_urls = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                if 'url' in node.metadata:
                    url = node.metadata['url']
                    if url not in source_urls:
                        source_urls.append(url)
        
        return {
            'answer': answer_text if answer_text else "No answer generated",
            'sources': source_urls
        }


# Main execution
if __name__ == "__main__":
    logger.warning("🚀 InitRAG Core System")
    logger.warning(f"📊 Using {'FREE' if USE_FREE_MODELS else 'PAID'} models")
    
    # Initialize system
    rag_system = InitRAGSystem()
    
    # Check Ollama if using free models
    if USE_FREE_MODELS and not rag_system.check_ollama_status():
        logger.error("Ollama not available. Please start Ollama and pull gemma2:2b")
        exit(1)
    
    # Load or create index
    rag_system.load_or_create_index()
    rag_system.create_query_engine()
    
    logger.info("✅ System ready!")
    logger.info("Add your UI layer (Flask, Streamlit, Gradio, etc.) to interact with this system")
