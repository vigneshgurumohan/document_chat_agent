# Document Chat Agent with RAG

A production-grade document chat agent with Retrieval-Augmented Generation (RAG) capabilities, built with LangChain and OpenAI.

## 🚀 Features

- **Multi-format Document Support**: PDF documents (easily extensible to other formats)
- **Advanced RAG**: Uses GPT-4 with text-embedding-3-large for high-quality responses
- **Detailed Citations**: Source documents with page numbers and content previews
- **Conversation Memory**: Maintains chat history for contextual conversations
- **Production-Ready**: ChromaDB vector store with persistent storage
- **Comprehensive Logging**: Detailed logging for debugging and monitoring

## 🛠️ Tech Stack

| Component | Technology | Choice |
|-----------|------------|---------|
| **LLM** | OpenAI GPT-4 | Best quality responses |
| **Embeddings** | text-embedding-3-large | Best quality embeddings |
| **Vector DB** | ChromaDB | Production-ready with metadata filtering |
| **Document Processing** | PyPDFLoader | Reliable PDF parsing |
| **Framework** | LangChain | Industry-standard orchestration |

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- PDF documents to process

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up Your OpenAI API Key

Set your OpenAI API key as an environment variable:

**On Windows:**
```cmd
set OPENAI_API_KEY=your-actual-openai-api-key-here
```

**On macOS/Linux:**
```bash
export OPENAI_API_KEY=your-actual-openai-api-key-here
```

**⚠️ Security Note**: Never hardcode API keys in your code. Always use environment variables for production use.

### 3. Add Your Documents

Create a `documents` folder and place your PDF files there:

```
your-project/
├── documents/
│   ├── document1.pdf
│   ├── document2.pdf
│   └── ...
├── document_agent.py
├── requirements.txt
└── README.md
```

### 4. Run the Agent

#### Option A: Interactive Chat Interface
```bash
python example_usage.py
```

#### Option B: Programmatic Usage
```python
import os
from document_agent import DocumentChatAgent

# Initialize agent
agent = DocumentChatAgent(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    documents_folder="documents"
)

# Process documents
agent.process_documents()

# Ask questions
result = agent.ask_question("What is the main topic of the documents?")
print(result["answer"])
```

## 📖 Usage Examples

### Basic Question Answering

```python
# Ask a simple question
result = agent.ask_question("What are the key findings in the research paper?")

# Get answer and sources
print(f"Answer: {result['answer']}")
print(f"Sources: {len(result['sources'])} found")
```

### Chat History Management

```python
# Get conversation history
history = agent.get_chat_history()
for conv in history:
    print(f"Q: {conv['question']}")
    print(f"A: {conv['answer']}")

# Clear memory
agent.clear_memory()
```

### Advanced Configuration

```python
agent = DocumentChatAgent(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    documents_folder="custom_docs",
    chunk_size=1500,        # Larger chunks for more context
    chunk_overlap=300,      # More overlap for better continuity
    temperature=0.5         # Lower temperature for more focused answers
)
```

## 🔧 Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 1000 | Size of text chunks for processing |
| `chunk_overlap` | 200 | Overlap between chunks |
| `temperature` | 0.7 | LLM creativity (0.0-1.0) |
| `documents_folder` | "documents" | Folder containing PDF files |

## 📊 Response Format

The agent returns structured responses:

```python
{
    "answer": "The main finding is...",
    "sources": [
        {
            "content": "Preview of source text...",
            "metadata": {
                "source": "document.pdf",
                "page": 5,
                "file_path": "/path/to/document.pdf"
            }
        }
    ],
    "error": None
}
```

## 🗂️ File Structure

```
project/
├── documents/           # Place your PDF files here
├── vector_db/          # Auto-generated vector database
├── document_agent.py   # Main agent implementation
├── example_usage.py    # Interactive usage example
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## 🔍 Troubleshooting

### Common Issues

1. **"No PDF files found"**
   - Ensure PDF files are in the `documents` folder
   - Check file extensions are `.pdf` (lowercase)

2. **"OpenAI API key error"**
   - Verify your API key is correct
   - Check your OpenAI account has sufficient credits

3. **"Import errors"**
   - Run `pip install -r requirements.txt`
   - Ensure Python 3.8+ is installed

4. **"Memory issues with large documents"**
   - Reduce `chunk_size` parameter
   - Process documents in smaller batches

### Performance Tips

- **Large Documents**: Increase `chunk_size` for better context
- **Many Documents**: Process in batches to avoid memory issues
- **Fast Responses**: Lower `temperature` for more focused answers
- **Better Citations**: Increase `chunk_overlap` for more context

## 🔒 Security Considerations

- Never commit API keys to version control
- Use environment variables for production deployments
- Consider rate limiting for high-volume usage
- Monitor API usage and costs

## 🚀 Production Deployment

For production use, consider:

1. **Environment Variables**: Use `.env` files or system environment variables
2. **Logging**: Configure proper logging levels and outputs
3. **Error Handling**: Implement retry logic and graceful error handling
4. **Monitoring**: Add metrics and health checks
5. **Scaling**: Consider using cloud vector databases for large datasets

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests. 