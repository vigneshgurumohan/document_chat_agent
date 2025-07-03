#!/usr/bin/env python3
"""
Example usage of the Document Chat Agent
"""

import os
from document_agent import DocumentChatAgent

def main():
    # Get OpenAI API key from environment variable
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # Initialize the agent
    print("Initializing Document Chat Agent...")
    agent = DocumentChatAgent(
        openai_api_key=OPENAI_API_KEY,
        documents_folder="documents",
        chunk_size=1000,
        chunk_overlap=200,
        temperature=0.7
    )
    
    # Process documents
    print("Processing documents...")
    success = agent.process_documents()
    
    if not success:
        print("❌ Failed to process documents. Please check:")
        print("   - OpenAI API key is correct")
        print("   - PDF files are in the 'documents' folder")
        print("   - All dependencies are installed")
        return
    
    print("✅ Document Chat Agent is ready!")
    print("\n" + "="*50)
    
    # Interactive chat loop
    print("💬 Chat with your documents (type 'quit' to exit, 'history' to see chat history)")
    print("="*50)
    
    while True:
        try:
            # Get user question
            question = input("\n🤔 Your question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if question.lower() == 'history':
                history = agent.get_chat_history()
                if history:
                    print("\n📚 Chat History:")
                    for i, conv in enumerate(history, 1):
                        print(f"\n{i}. Q: {conv['question']}")
                        print(f"   A: {conv['answer'][:200]}...")
                else:
                    print("No chat history yet.")
                continue
            
            if not question:
                continue
            
            # Get answer
            print("🤖 Thinking...")
            result = agent.ask_question(question)
            
            if result["error"]:
                print(f"❌ Error: {result['error']}")
                continue
            
            # Display answer
            print(f"\n✅ Answer: {result['answer']}")
            
            # Display sources if available
            if result["sources"]:
                print(f"\n📖 Sources ({len(result['sources'])} found):")
                for i, source in enumerate(result["sources"], 1):
                    metadata = source["metadata"]
                    source_name = metadata.get("source", "Unknown")
                    page = metadata.get("page", "N/A")
                    print(f"   {i}. {source_name} (Page: {page})")
                    print(f"      Preview: {source['content']}")
            
            print("\n" + "-"*50)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main() 