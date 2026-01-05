<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# InitRAG - LlamaIndex Website Crawler Project

This project implements a sophisticated RAG (Retrieval-Augmented Generation) application that crawls and indexes the entire init-owl.de website using LlamaIndex.

## Project Overview
- **Language**: Python 3.8+
- **Framework**: LlamaIndex for RAG functionality
- **Web Crawling**: Custom recursive crawler with requests and BeautifulSoup
- **Content Extraction**: Trafilatura for clean text extraction
- **PDF Processing**: LlamaParse/PDFReader for PDF content
- **Vector Store**: ChromaDB for document indexing
- **Chat Interface**: LlamaIndex chat engine

## Key Features
1. Recursive website crawling with domain filtering
2. Clean content extraction using trafilatura
3. PDF document processing
4. Persistent vector storage
5. Interactive chat interface with source citations

## Development Guidelines
- Follow Python best practices and PEP 8
- Implement proper error handling and logging
- Use type hints for better code clarity
- Add comprehensive docstrings
- Respect website politeness with request delays