#pip install chromadb
#pip install pypdf

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import shutil

# LangChain imports
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChatAgent:
    """
    Production-grade Document Chat Agent with RAG capabilities
    """
    
    def __init__(self, 
                 openai_api_key: str,
                 documents_folder: str = "documents",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 temperature: float = 0.7):
        """
        Initialize the Document Chat Agent
        
        Args:
            openai_api_key: OpenAI API key
            documents_folder: Path to folder containing PDF documents
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            temperature: LLM temperature for responses
        """
        self.openai_api_key = openai_api_key
        self.documents_folder = Path(documents_folder)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temperature = temperature
        
        # Initialize components
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = None
        
        # Create documents folder if it doesn't exist
        self.documents_folder.mkdir(exist_ok=True)
        
        # Initialize the system
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize all LangChain components"""
        try:
            # Initialize embeddings with text-embedding-3-large (best quality)
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=self.openai_api_key
            )
            
            # Initialize LLM with GPT-4
            self.llm = ChatOpenAI(
                model="gpt-4",
                temperature=self.temperature,
                openai_api_key=self.openai_api_key
            )
            
            # Initialize memory for conversation context
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            
            # Initialize text splitter
            self.text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            logger.info("Components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def load_documents(self) -> List[Document]:
        """
        Load all PDF documents from the documents folder
        
        Returns:
            List of processed documents
        """
        documents = []
        
        try:
            # Find all PDF files in the documents folder
            pdf_files = list(self.documents_folder.glob("*.pdf"))
            
            if not pdf_files:
                logger.warning(f"No PDF files found in {self.documents_folder}")
                return documents
            
            logger.info(f"Found {len(pdf_files)} PDF files")
            
            for pdf_path in pdf_files:
                try:
                    logger.info(f"Processing: {pdf_path.name}")
                    
                    # Load PDF using PyPDFLoader
                    loader = PyPDFLoader(str(pdf_path))
                    pdf_docs = loader.load()
                    
                    # Add metadata to each document
                    for doc in pdf_docs:
                        doc.metadata.update({
                            "source": pdf_path.name,
                            "file_path": str(pdf_path),
                            "document_type": "pdf"
                        })
                    
                    documents.extend(pdf_docs)
                    logger.info(f"Successfully loaded {len(pdf_docs)} pages from {pdf_path.name}")
                    
                except Exception as e:
                    logger.error(f"Error processing {pdf_path.name}: {e}")
                    continue
            
            logger.info(f"Total documents loaded: {len(documents)}")
            return documents
            
        except Exception as e:
            logger.error(f"Error loading documents: {e}")
            raise
    
    def create_vectorstore(self, documents: List[Document]) -> Chroma:
        """
        Create and populate the vector store with document chunks
        
        Args:
            documents: List of documents to process
            
        Returns:
            Chroma vector store
        """
        try:
            if not documents:
                logger.warning("No documents to process")
                return None
            
            # Split documents into chunks
            logger.info("Splitting documents into chunks...")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Created {len(chunks)} chunks")
            
            # Create vector store with ChromaDB (best for production)
            # Use a direct persistent directory
            db_path = Path("vector_db")
            if db_path.exists():
                shutil.rmtree(db_path)
            
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embeddings,
                persist_directory=str(db_path)
            )
            
            logger.info("Vector store created and populated successfully")
            return self.vectorstore
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def setup_qa_chain(self):
        """Setup the question-answering chain with detailed citations"""
        
        # Custom prompt template for detailed citations
        template = """You are a helpful AI assistant that answers questions based on the provided context. 
        Always provide detailed citations including the source document name and page number when available.
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Human: {question}
        
        Assistant: Please provide a comprehensive answer based on the context above. 
        When citing information, use the format: [Source: document_name, Page: X] or [Source: document_name] if page number is not available.
        If the context doesn't contain enough information to answer the question, say so clearly.
        
        Answer:"""
        
        prompt = PromptTemplate(
            input_variables=["context", "chat_history", "question"],
            template=template
        )
        
        # Create the conversational retrieval chain without memory to avoid output_key conflict
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            ),
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("QA chain setup completed")
    
    def process_documents(self):
        """Process all documents and setup the chat system"""
        try:
            logger.info("Starting document processing...")
            
            # Load documents
            documents = self.load_documents()
            
            if not documents:
                logger.warning("No documents to process")
                return False
            
            # Create vector store
            self.create_vectorstore(documents)
            
            # Setup QA chain
            self.setup_qa_chain()
            
            logger.info("Document processing completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Ask a question and get an answer with citations
        
        Args:
            question: The question to ask
            
        Returns:
            Dictionary containing answer and source documents
        """
        try:
            if not self.qa_chain:
                return {
                    "answer": "Please process documents first using process_documents()",
                    "sources": [],
                    "error": "QA chain not initialized"
                }
            
            # Get chat history for context
            chat_history = []
            if self.memory:
                for i in range(0, len(self.memory.chat_memory.messages), 2):
                    if i + 1 < len(self.memory.chat_memory.messages):
                        chat_history.append((
                            self.memory.chat_memory.messages[i].content,
                            self.memory.chat_memory.messages[i + 1].content
                        ))
            
            # Get answer with sources
            result = self.qa_chain({
                "question": question,
                "chat_history": chat_history
            })
            
            # Store in memory manually
            if self.memory:
                self.memory.chat_memory.add_user_message(question)
                self.memory.chat_memory.add_ai_message(result["answer"])
            
            # Extract source documents
            source_docs = result.get("source_documents", [])
            sources = []
            
            for doc in source_docs:
                source_info = {
                    "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return {
                "answer": result["answer"],
                "sources": sources,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error asking question: {e}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "sources": [],
                "error": str(e)
            }
    
    def get_chat_history(self) -> List[Dict[str, str]]:
        """Get the current chat history"""
        try:
            if not self.memory:
                return []
            
            # Extract chat history from memory
            history = []
            for i in range(0, len(self.memory.chat_memory.messages), 2):
                if i + 1 < len(self.memory.chat_memory.messages):
                    history.append({
                        "question": self.memory.chat_memory.messages[i].content,
                        "answer": self.memory.chat_memory.messages[i + 1].content
                    })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting chat history: {e}")
            return []
    
    def clear_memory(self):
        """Clear the conversation memory"""
        if self.memory:
            self.memory.clear()
            logger.info("Chat memory cleared")


def main():
    """
    Main function to demonstrate usage
    """
    # Get OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize the agent
    agent = DocumentChatAgent(
        openai_api_key=OPENAI_API_KEY,
        documents_folder="documents",
        chunk_size=1000,
        chunk_overlap=200,
        temperature=0.7
    )
    
    # Process documents
    success = agent.process_documents()
    
    if not success:
        print("Failed to process documents. Please check the logs.")
        return
    
    print("Document Chat Agent is ready!")
    print("Place your PDF documents in the 'documents' folder and restart to process them.")
    print("Use agent.ask_question('your question') to ask questions.")
    print("Use agent.get_chat_history() to see conversation history.")
    print("Use agent.clear_memory() to clear conversation history.")


if __name__ == "__main__":
    main()
