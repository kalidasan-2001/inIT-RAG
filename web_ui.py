"""
InitRAG - Clean Modern Flask Web UI
Standard chat interface for inIT Institute RAG chatbot.
Clean, minimal design with essential features.
"""

import logging
from flask import Flask, render_template_string, request, jsonify
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress verbose logging
logging.getLogger('transformers').setLevel(logging.WARNING)
logging.getLogger('sentence_transformers').setLevel(logging.WARNING)
logging.getLogger('llama_index').setLevel(logging.WARNING)

# Flask app
app = Flask(__name__)
CORS(app)

# Global state
system_ready = False
rag_system = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>inIT Bot</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: #f7f7f8;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }

        .header {
            background: white;
            border-bottom: 1px solid #e5e5e5;
            padding: 16px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }

        .logo {
            font-size: 20px;
            font-weight: 600;
            color: #202123;
        }

        .theme-btn {
            background: none;
            border: 1px solid #d9d9e3;
            border-radius: 6px;
            padding: 8px 12px;
            cursor: pointer;
            font-size: 14px;
        }

        .theme-btn:hover {
            background: #f7f7f8;
        }

        .chat-wrapper {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
        }

        .chat-messages {
            max-width: 768px;
            margin: 0 auto;
        }

        .message {
            margin-bottom: 24px;
            display: flex;
            gap: 16px;
        }

        .avatar {
            width: 32px;
            height: 32px;
            border-radius: 4px;
            flex-shrink: 0;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
        }

        .user-avatar {
            background: #10a37f;
            color: white;
        }

        .bot-avatar {
            background: #19c37d;
            color: white;
        }

        .message-content {
            flex: 1;
            line-height: 1.6;
        }

        .user .message-content {
            color: #202123;
        }

        .bot .message-content {
            color: #353740;
        }

        .sources {
            margin-top: 12px;
            padding-top: 12px;
            border-top: 1px solid #e5e5e5;
            font-size: 14px;
        }

        .source-link {
            color: #10a37f;
            text-decoration: none;
            display: block;
            padding: 4px 0;
        }

        .source-link:hover {
            text-decoration: underline;
        }

        .input-area {
            background: white;
            border-top: 1px solid #e5e5e5;
            padding: 16px 24px;
        }

        .input-container {
            max-width: 768px;
            margin: 0 auto;
            position: relative;
        }

        #userInput {
            width: 100%;
            border: 1px solid #d9d9e3;
            border-radius: 8px;
            padding: 12px 48px 12px 16px;
            font-size: 15px;
            resize: none;
            font-family: inherit;
            outline: none;
        }

        #userInput:focus {
            border-color: #10a37f;
        }

        #sendBtn {
            position: absolute;
            right: 8px;
            bottom: 8px;
            background: #10a37f;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 8px 16px;
            cursor: pointer;
            font-size: 14px;
        }

        #sendBtn:hover:not(:disabled) {
            background: #0d8c6f;
        }

        #sendBtn:disabled {
            background: #d9d9e3;
            cursor: not-allowed;
        }

        .loading {
            display: none;
            text-align: center;
            color: #8e8ea0;
            padding: 16px;
        }

        .loading.active {
            display: block;
        }

        .welcome {
            max-width: 768px;
            margin: 80px auto;
            text-align: center;
        }

        .welcome h1 {
            font-size: 32px;
            margin-bottom: 16px;
            color: #202123;
        }

        .welcome p {
            font-size: 16px;
            color: #6e6e80;
            margin-bottom: 32px;
        }

        .suggestions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            max-width: 600px;
            margin: 0 auto;
        }

        .suggestion {
            background: white;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 16px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s;
        }

        .suggestion:hover {
            border-color: #10a37f;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }

        .suggestion-title {
            font-weight: 500;
            margin-bottom: 4px;
        }

        /* Dark theme */
        [data-theme="dark"] body {
            background: #343541;
        }

        [data-theme="dark"] .header {
            background: #202123;
            border-color: #444654;
        }

        [data-theme="dark"] .logo,
        [data-theme="dark"] .user .message-content,
        [data-theme="dark"] .welcome h1 {
            color: #ececf1;
        }

        [data-theme="dark"] .bot .message-content {
            color: #d1d5db;
        }

        [data-theme="dark"] .welcome p {
            color: #8e8ea0;
        }

        [data-theme="dark"] .input-area {
            background: #40414f;
            border-color: #444654;
        }

        [data-theme="dark"] #userInput {
            background: #40414f;
            color: #ececf1;
            border-color: #565869;
        }

        [data-theme="dark"] .suggestion {
            background: #444654;
            border-color: #565869;
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">🤖 inIT Bot</div>
        <button class="theme-btn" onclick="toggleTheme()" id="themeBtn">🌙 Dark</button>
    </div>

    <div class="chat-wrapper" id="chatWrapper">
        <div class="chat-messages" id="chatMessages">
            <div class="welcome" id="welcome">
                <h1>Welcome to inIT Bot</h1>
                <p>Ask me anything about the inIT Institute for Industrial IT</p>
                <div class="suggestions">
                    <div class="suggestion" onclick="askQuestion('Tell me about research projects')">
                        <div class="suggestion-title">🔬 Research Projects</div>
                    </div>
                    <div class="suggestion" onclick="askQuestion('Who are the professors?')">
                        <div class="suggestion-title">👨‍🏫 Professors</div>
                    </div>
                    <div class="suggestion" onclick="askQuestion('What are the current news?')">
                        <div class="suggestion-title">📰 Latest News</div>
                    </div>
                    <div class="suggestion" onclick="askQuestion('How can I contact the institute?')">
                        <div class="suggestion-title">📞 Contact Info</div>
                    </div>
                </div>
            </div>
        </div>
        <div class="loading" id="loading">Thinking...</div>
    </div>

    <div class="input-area">
        <div class="input-container">
            <textarea 
                id="userInput" 
                placeholder="Ask a question..." 
                rows="1"
                onkeydown="handleKeyDown(event)"
            ></textarea>
            <button id="sendBtn" onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        // Theme management
        function toggleTheme() {
            const html = document.documentElement;
            const current = html.getAttribute('data-theme') || 'light';
            const newTheme = current === 'dark' ? 'light' : 'dark';
            html.setAttribute('data-theme', newTheme);
            localStorage.setItem('theme', newTheme);
            document.getElementById('themeBtn').textContent = newTheme === 'dark' ? '☀️ Light' : '🌙 Dark';
        }

        // Load saved theme
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        document.getElementById('themeBtn').textContent = savedTheme === 'dark' ? '☀️ Light' : '🌙 Dark';

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function askQuestion(question) {
            document.getElementById('userInput').value = question;
            sendMessage();
        }

        async function sendMessage() {
            const input = document.getElementById('userInput');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Hide welcome
            document.getElementById('welcome').style.display = 'none';
            
            // Add user message
            addMessage(message, 'user');
            input.value = '';
            
            // Show loading
            document.getElementById('loading').classList.add('active');
            document.getElementById('sendBtn').disabled = true;
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ question: message })
                });
                
                const data = await response.json();
                
                if (data.error) {
                    addMessage('Error: ' + data.error, 'bot');
                } else {
                    addMessage(data.answer, 'bot', data.sources);
                }
            } catch (error) {
                addMessage('Connection error. Please try again.', 'bot');
            } finally {
                document.getElementById('loading').classList.remove('active');
                document.getElementById('sendBtn').disabled = false;
                input.focus();
            }
        }

        function addMessage(text, type, sources = []) {
            const container = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${type}`;
            
            const avatar = document.createElement('div');
            avatar.className = `avatar ${type}-avatar`;
            avatar.textContent = type === 'user' ? '👤' : '🤖';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            content.textContent = text;
            
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<strong>Sources:</strong>';
                sources.forEach((url, i) => {
                    const link = document.createElement('a');
                    link.href = url;
                    link.target = '_blank';
                    link.className = 'source-link';
                    link.textContent = `${i + 1}. ${url}`;
                    sourcesDiv.appendChild(link);
                });
                content.appendChild(sourcesDiv);
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            container.appendChild(messageDiv);
            
            // Scroll to bottom
            document.getElementById('chatWrapper').scrollTop = document.getElementById('chatWrapper').scrollHeight;
        }

        // Auto-resize textarea
        const textarea = document.getElementById('userInput');
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 200) + 'px';
        });
    </script>
</body>
</html>
"""


def initialize_system():
    """Initialize RAG system - preload everything before Flask starts."""
    global system_ready, rag_system
    try:
        logger.info("🔧 Loading RAG system...")
        
        # Import here to avoid slow startup
        from app import InitRAGSystem
        rag_system = InitRAGSystem()
        
        # Check Ollama
        if not rag_system.check_ollama_status():
            logger.error("❌ Ollama not available")
            return
        
        # Load index (this is fast since index already exists)
        rag_system.load_or_create_index()
        rag_system.create_query_engine()
        
        system_ready = True
        logger.info("✅ RAG system ready!")
    except Exception as e:
        logger.error(f"❌ Initialization error: {e}")
        import traceback
        traceback.print_exc()


@app.route('/')
def index():
    """Serve main page."""
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/status')
def status():
    """Check system status."""
    return jsonify({
        'ready': system_ready,
        'message': 'System ready' if system_ready else 'Initializing...'
    })


@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    if not system_ready:
        return jsonify({'error': 'System not ready. Please wait...'}), 503
    
    try:
        data = request.json
        question = data.get('question', '').strip()
        
        if not question:
            return jsonify({'error': 'No question provided'}), 400
        
        logger.info(f"Question: {question}")
        
        # Query RAG system
        result = rag_system.query(question)
        
        return jsonify({
            'answer': result['answer'],
            'sources': result['sources']
        })
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    logger.info("🚀 Starting inIT RAG System...")
    
    # Initialize system BEFORE starting Flask (blocks until ready)
    initialize_system()
    
    if not system_ready:
        logger.error("❌ Failed to initialize. Please check Ollama and try again.")
        exit(1)
    
    logger.info("🌐 Starting web server...")
    logger.info("📍 Open http://localhost:5000 in your browser")
    
    # Run Flask app
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
