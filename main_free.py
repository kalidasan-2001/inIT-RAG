"""
Free InitRAG - LlamaIndex RAG Application with Local Models

This version uses completely free, local alternatives:
- Ollama for local LLM (instead of OpenAI)
- HuggingFace embeddings (instead of OpenAI embeddings)
- No API keys or costs required!
"""

import os
import time
import logging
import requests
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from typing import List, Set, Dict, Optional
from dataclasses import dataclass
from pathlib import Path

import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from llama_index.core import (
    VectorStoreIndex, 
    Document, 
    StorageContext, 
    Settings,
    load_index_from_storage
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.readers.file import PDFReader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('crawler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
        logger.info(f"Allowed domains: {self.allowed_domains}")
    
    def is_valid_url(self, url: str) -> bool:
        """Check if URL should be crawled based on domain and file extension."""
        try:
            parsed = urlparse(url)
            
            # Check domain
            domain = parsed.netloc.lower()
            if not any(allowed in domain for allowed in self.allowed_domains):
                return False
            
            # Check file extension
            path = parsed.path.lower()
            if path.endswith('/'):
                return True  # Directory URLs are valid
                
            # Get file extension
            ext = os.path.splitext(path)[1]
            if ext and ext in self.config.blocked_extensions:
                return False
                
            # If has extension, must be in allowed list
            if ext and ext not in self.config.allowed_extensions:
                return False
                
            return True
            
        except Exception as e:
            logger.warning(f"Error validating URL {url}: {e}")
            return False
    
    def normalize_url(self, url: str) -> str:
        """Normalize URL by removing fragments and query parameters (optional)."""
        parsed = urlparse(url)
        # Keep query parameters for dynamic content, but remove fragments
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            ''  # Remove fragment
        ))
        return normalized
    
    def extract_links(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            links = []
            
            for link in soup.find_all('a', href=True):
                href = link['href'].strip()
                if not href or href.startswith('#'):
                    continue
                    
                # Convert relative URLs to absolute
                absolute_url = urljoin(base_url, href)
                normalized_url = self.normalize_url(absolute_url)
                
                if self.is_valid_url(normalized_url):
                    links.append(normalized_url)
            
            return list(set(links))  # Remove duplicates
            
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
            return []
    
    def extract_content(self, html_content: str, url: str) -> Optional[str]:
        """Extract clean text content using trafilatura."""
        try:
            # Use trafilatura for clean content extraction
            extracted = trafilatura.extract(
                html_content,
                include_comments=False,
                include_tables=True,
                include_formatting=True,
                url=url
            )
            
            if extracted and len(extracted.strip()) > 100:  # Minimum content length
                return extracted.strip()
            else:
                logger.warning(f"No substantial content extracted from {url}")
                return None
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def fetch_page(self, url: str) -> Optional[str]:
        """Fetch and return page content."""
        try:
            response = self.session.get(
                url, 
                timeout=self.config.timeout,
                allow_redirects=True
            )
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            if 'text/html' not in content_type and 'application/pdf' not in content_type:
                logger.debug(f"Skipping non-HTML/PDF content: {url} ({content_type})")
                return None
            
            return response.text if 'text/html' in content_type else response.content
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            self.failed_urls.add(url)
            return None
    
    def process_pdf(self, url: str) -> Optional[Document]:
        """Process PDF document and extract text."""
        try:
            response = self.session.get(url, timeout=self.config.timeout)
            response.raise_for_status()
            
            # Save PDF temporarily
            temp_path = f"temp_{hash(url)}.pdf"
            with open(temp_path, 'wb') as f:
                f.write(response.content)
            
            try:
                # Use basic PDFReader (free)
                documents = self.pdf_reader.load_data(temp_path)
                content = documents[0].text if documents else None
                
                if content and len(content.strip()) > 100:
                    return Document(
                        text=content.strip(),
                        metadata={
                            'url': url,
                            'content_type': 'pdf',
                            'title': os.path.basename(urlparse(url).path)
                        }
                    )
                    
            finally:
                # Clean up temp file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Error processing PDF {url}: {e}")
            
        return None
    
    def crawl_page(self, url: str, depth: int = 0) -> List[str]:
        """Crawl a single page and return found links."""
        if (url in self.visited_urls or 
            depth > self.config.max_depth or 
            len(self.visited_urls) >= self.config.max_pages):
            return []
        
        self.visited_urls.add(url)
        logger.info(f"Crawling (depth {depth}): {url}")
        
        # Respect politeness delay
        time.sleep(self.config.delay)
        
        # Handle PDF files separately
        if url.lower().endswith('.pdf'):
            document = self.process_pdf(url)
            if document:
                self.documents.append(document)
                logger.info(f"Successfully processed PDF: {url}")
            return []
        
        # Fetch HTML content
        html_content = self.fetch_page(url)
        if not html_content:
            return []
        
        # Extract clean text content
        text_content = self.extract_content(html_content, url)
        if text_content:
            # Create document with metadata
            document = Document(
                text=text_content,
                metadata={
                    'url': url,
                    'content_type': 'html',
                    'depth': depth,
                    'title': self.extract_title(html_content)
                }
            )
            self.documents.append(document)
            logger.info(f"Successfully extracted content from: {url} ({len(text_content)} chars)")
        
        # Extract links for further crawling
        return self.extract_links(html_content, url)
    
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
    
    def crawl_domain(self, base_url: str, max_depth: int = 3) -> List[Document]:
        """Recursively crawl domain starting from base_url."""
        logger.info(f"Starting domain crawl from: {base_url}")
        logger.info(f"Max depth: {max_depth}, Max pages: {self.config.max_pages}")
        
        self.config.max_depth = max_depth
        urls_to_visit = [base_url]
        current_depth = 0
        
        while urls_to_visit and current_depth <= max_depth:
            logger.info(f"Processing depth level {current_depth} with {len(urls_to_visit)} URLs")
            next_level_urls = []
            
            for url in urls_to_visit:
                if len(self.visited_urls) >= self.config.max_pages:
                    logger.info(f"Reached maximum pages limit ({self.config.max_pages})")
                    break
                    
                found_links = self.crawl_page(url, current_depth)
                
                # Add new links for next depth level
                for link in found_links:
                    if link not in self.visited_urls:
                        next_level_urls.append(link)
            
            urls_to_visit = list(set(next_level_urls))  # Remove duplicates
            current_depth += 1
        
        logger.info(f"Crawling completed!")
        logger.info(f"Total pages visited: {len(self.visited_urls)}")
        logger.info(f"Total documents created: {len(self.documents)}")
        logger.info(f"Failed URLs: {len(self.failed_urls)}")
        
        return self.documents


class FreeInitRAGApplication:
    """Free RAG application using local models."""
    
    def __init__(self, storage_dir: str = "./storage_init_full"):
        self.storage_dir = Path(storage_dir)
        self.setup_llama_index()
        self.index = None
        self.chat_engine = None
        
    def setup_llama_index(self):
        """Configure LlamaIndex with free local models."""
        logger.info("Setting up free local models...")
        
        try:
            # Use free local Ollama LLM
            Settings.llm = Ollama(
                model="gemma2:2b",  # Smaller model that works with available memory
                request_timeout=120.0,
                base_url="http://localhost:11434"
            )
            logger.info("✅ Ollama LLM configured")
        except Exception as e:
            logger.error(f"❌ Ollama setup failed: {e}")
            logger.info("🔧 Please install Ollama from https://ollama.com/download")
            raise
        
        try:
            # Use free HuggingFace embeddings
            Settings.embed_model = HuggingFaceEmbedding(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            logger.info("✅ HuggingFace embeddings configured")
        except Exception as e:
            logger.error(f"❌ HuggingFace embeddings setup failed: {e}")
            raise
        
        # Configure node parser
        Settings.node_parser = SentenceSplitter(
            chunk_size=int(os.getenv("CHUNK_SIZE", "1024")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "200"))
        )
        
        logger.info("🎉 Free LlamaIndex setup completed!")
    
    def check_ollama_status(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            models = response.json().get("models", [])
            
            # Check if required model is available
            model_names = [model["name"] for model in models]
            if "gemma2:2b" not in model_names and "llama3.2:3b" not in model_names:
                logger.info("📥 Downloading gemma2:2b model (this may take a few minutes)...")
                os.system("ollama pull gemma2:2b")
            
            logger.info("✅ Ollama is running and model is available")
            return True
        except:
            logger.error("❌ Ollama is not running!")
            logger.info("🚀 Please start Ollama: 'ollama serve'")
            return False
    
    def load_or_create_index(self) -> VectorStoreIndex:
        """Load existing index or create new one by crawling."""
        if self.storage_dir.exists() and any(self.storage_dir.iterdir()):
            logger.info("Loading existing index from storage...")
            try:
                storage_context = StorageContext.from_defaults(persist_dir=str(self.storage_dir))
                self.index = load_index_from_storage(storage_context)
                logger.info("✅ Successfully loaded existing index")
                return self.index
            except Exception as e:
                logger.error(f"❌ Error loading index: {e}")
                logger.info("Creating new index...")
        
        logger.info("🕷️ Creating new index by crawling website...")
        return self.create_new_index()
    
    def create_new_index(self) -> VectorStoreIndex:
        """Create new index by crawling the website."""
        config = CrawlConfig(
            base_url="https://www.init-owl.de",
            max_depth=int(os.getenv("MAX_DEPTH", "3")),
            max_pages=int(os.getenv("MAX_PAGES_PER_DOMAIN", "1000")),
            delay=float(os.getenv("CRAWL_DELAY", "0.5"))
        )
        
        crawler = WebsiteCrawler(config)
        documents = crawler.crawl_domain(config.base_url, config.max_depth)
        
        if not documents:
            raise ValueError("No documents were extracted from the website")
        
        logger.info(f"📚 Creating index from {len(documents)} documents...")
        
        # Create index
        self.index = VectorStoreIndex.from_documents(documents)
        
        # Persist to storage
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index.storage_context.persist(persist_dir=str(self.storage_dir))
        
        logger.info(f"💾 Index created and persisted to {self.storage_dir}")
        return self.index
    
    def setup_chat_engine(self):
        """Setup the chat engine with custom system prompt."""
        if not self.index:
            raise ValueError("Index not initialized")
        
        system_prompt = (
            "You are the AI expert for the inIT Institute in Lemgo. "
            "Answer users' questions based on the retrieved context from the init-owl.de website. "
            "Always cite the URL where you found the information. "
            "If the information is not available in the context, clearly state that. "
            "Provide accurate, helpful, and detailed responses."
        )
        
        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            system_prompt=system_prompt,
            verbose=True
        )
        
        logger.info("💬 Chat engine initialized successfully")
    
    def chat(self, message: str) -> str:
        """Send a message to the chat engine and get response."""
        if not self.chat_engine:
            self.setup_chat_engine()
        
        response = self.chat_engine.chat(message)
        return str(response)
    
    def run_interactive_chat(self):
        """Run interactive chat loop."""
        print("\n" + "="*60)
        print("🆓 FREE InitRAG - inIT Institute AI Assistant")
        print("="*60)
        print("💡 Powered by FREE local models (Ollama + HuggingFace)")
        print("Ask questions about the inIT Institute in Lemgo!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("\n👋 Goodbye! Thank you for using FREE InitRAG!")
                    break
                
                if not user_input:
                    continue
                
                print("\n🤖 AI Assistant: ", end="", flush=True)
                response = self.chat(user_input)
                print(response)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye! Thank you for using FREE InitRAG!")
                break
            except Exception as e:
                logger.error(f"Error in chat: {e}")
                print(f"\n❌ Sorry, an error occurred: {e}")


def main():
    """Main application entry point."""
    print("🆓 Starting FREE InitRAG Application...")
    print("💡 Using completely free local models - no API keys needed!")
    
    try:
        # Initialize application
        app = FreeInitRAGApplication()
        
        # Check Ollama status
        if not app.check_ollama_status():
            print("\n❌ Ollama setup required. Please:")
            print("1. Install Ollama: https://ollama.com/download")
            print("2. Start Ollama: 'ollama serve'")
            print("3. Run this script again")
            return
        
        # Load or create index
        app.load_or_create_index()
        
        # Setup chat engine
        app.setup_chat_engine()
        
        # Run interactive chat
        app.run_interactive_chat()
        
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"\n❌ Application error: {e}")


if __name__ == "__main__":
    main()