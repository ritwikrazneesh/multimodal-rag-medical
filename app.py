"""
Smart Healthcare Assistant with Citations
Dual-mode: RAG (with citations) + Generative (with disclaimers)
Uses LLM-based query rewriting for context-aware retrieval
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import streamlit as st
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from typing import List, Dict

# Load environment
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
VECTORSTORE_DIR = os.path.join(working_dir, "vector_db_dir")

# Configuration
SIMILARITY_THRESHOLD = 1.2  # Lower score = better match in ChromaDB
MIN_SOURCES = 2
MAX_SOURCES = 5


# ============================================================================
# VECTOR STORE AND EMBEDDINGS
# ============================================================================

@st.cache_resource
def load_vectorstore():
    """Load the vector database"""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    if not os.path.exists(VECTORSTORE_DIR):
        st.error("❌ Vector database not found! Run: `python vectorize_documents.py`")
        st.stop()
    
    return Chroma(
        persist_directory=VECTORSTORE_DIR,
        embedding_function=embeddings,
        collection_name="medical_knowledge"
    )


# ============================================================================
# LLM AND PROMPTS
# ============================================================================

@st.cache_resource
def get_llm():
    """Get the language model"""
    return ChatGoogleGenerativeAI(
        model="models/gemini-flash-latest",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )


def get_condense_question_prompt():
    """Prompt for rewriting follow-up questions with context"""
    template = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that includes all necessary context from the conversation history.

If the follow-up question references previous topics (like "it", "this", "complications", "treatment", "diet"), incorporate the specific medical condition or topic from the conversation history.

Chat History:
{chat_history}

Follow-Up Question: {question}

Standalone Question with Context:"""
    
    return PromptTemplate(
        input_variables=["chat_history", "question"],
        template=template
    )


def get_rag_prompt():
    """Prompt for RAG mode (with sources)"""
    template = """You are a helpful medical information assistant.

Use ONLY the following verified medical sources to answer the question.
Do not add information from outside these sources.

VERIFIED SOURCES:
{context}

IMPORTANT INSTRUCTIONS:
- Provide accurate information based ONLY on the sources above
- Be clear, concise, and helpful
- If the sources don't fully answer the question, acknowledge what's missing
- Do not make up or infer information not in the sources
- Use medical terminology but explain it clearly
- If appropriate, mention when to consult a healthcare professional

QUESTION: {question}

ANSWER:"""
    
    return PromptTemplate(
        input_variables=["context", "question"],
        template=template
    )


def get_generative_prompt():
    """Prompt for generative mode (no sources)"""
    template = """You are a helpful medical information assistant.

⚠️ IMPORTANT: No verified sources are available in the database for this specific query.
Provide a helpful but cautious response based on your general medical knowledge.

GUIDELINES:
- Be helpful but emphasize this is general information
- Clearly recommend consulting healthcare professionals for medical advice
- Avoid making definitive medical diagnoses or treatment claims
- Be clear about uncertainty
- Suggest what type of healthcare professional to consult if relevant
- Keep the response concise but informative

QUESTION: {question}

ANSWER:"""
    
    return PromptTemplate(
        input_variables=["question"],
        template=template
    )


# ============================================================================
# CONVERSATION CHAIN (WITH MEMORY)
# ============================================================================

def build_rag_chain(vectorstore, memory):
    """Build conversational RAG chain with memory and query rewriting"""
    llm = get_llm()
    
    # Get prompts
    condense_question_prompt = get_condense_question_prompt()
    rag_prompt = get_rag_prompt()
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": MAX_SOURCES}
    )
    
    # Build conversational chain with query rewriting
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        verbose=True,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": rag_prompt},
        condense_question_prompt=condense_question_prompt,
        condense_question_llm=llm,  # Use LLM to rewrite follow-up questions
    )
    
    return chain


# ============================================================================
# RESPONSE GENERATION
# ============================================================================

def generate_response(question: str, vectorstore, memory) -> Dict:
    """
    Generate response with appropriate mode and citations
    Uses LLM-based query rewriting for context-aware retrieval
    """
    
    print(f"\n{'='*70}")
    print(f"🔍 QUERY: {question}")
    print(f"{'='*70}")
    print(f"   Using conversational RAG chain with automatic query rewriting")
    
    llm = get_llm()
    
    # Always use conversational chain (it handles context internally)
    try:
        print(f"\n🤖 INVOKING RAG CHAIN...")
        
        chain = build_rag_chain(vectorstore, memory)
        response = chain.invoke({"question": question})
        answer = response["answer"]
        
        # Extract sources
        sources = []
        seen_urls = set()
        source_documents = response.get("source_documents", [])
        
        print(f"\n📊 DOCUMENTS RETRIEVED: {len(source_documents)}")
        
        for idx, doc in enumerate(source_documents[:MAX_SOURCES], 1):
            metadata = doc.metadata
            title = metadata.get('title', 'Unknown')
            url = metadata.get('url', metadata.get('source', 'Unknown'))
            
            print(f"  {idx}. {title}")
            
            if url not in seen_urls:
                sources.append({
                    'title': title,
                    'url': url,
                    'source': metadata.get('source', 'Unknown'),
                    'organization': metadata.get('organization', ''),
                    'date_updated': metadata.get('date_updated', ''),
                    'confidence': 0.0,
                    'doc_type': metadata.get('doc_type', 'unknown')
                })
                seen_urls.add(url)
        
        print(f"\n✅ RESPONSE GENERATED: {len(answer)} chars")
        print(f"   Unique sources: {len(sources)}")
        
        # Decide mode based on sources found
        if len(sources) >= MIN_SOURCES:
            print(f"   ✓ RAG MODE (enough sources)")
            return {
                'answer': answer,
                'sources': sources,
                'mode': 'rag',
                'metadata': {
                    'num_sources': len(sources),
                    'avg_confidence': 0.0
                }
            }
        else:
            print(f"   ⚠️ GENERATIVE MODE (only {len(sources)} sources)")
            return {
                'answer': answer,
                'sources': [],
                'mode': 'generative',
                'metadata': {}
            }
            
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        
        # Fallback
        print(f"\n🤖 FALLBACK: Pure generative mode")
        prompt = get_generative_prompt()
        answer = llm.invoke(prompt.format(question=question)).content
        
        return {
            'answer': answer,
            'sources': [],
            'mode': 'generative',
            'metadata': {}
        }


def format_response_with_citations(response_data: Dict) -> str:
    """Format response with appropriate citations or disclaimers"""
    answer = response_data['answer']
    sources = response_data['sources']
    mode = response_data['mode']
    
    formatted_response = answer
    
    if mode == 'rag' and sources:
        # Add citations section
        formatted_response += "\n\n---\n\n### 📚 Sources\n\n"
        
        for idx, source in enumerate(sources, 1):
            formatted_response += f"**{idx}. {source['title']}**\n"
            formatted_response += f"   - Source: {source['source']}"
            
            if source.get('organization'):
                formatted_response += f" ({source['organization']})"
            
            formatted_response += "\n"
            
            if source.get('date_updated'):
                formatted_response += f"   - Updated: {source['date_updated']}\n"
            
            formatted_response += f"   - [View Source]({source['url']})\n\n"
        
        formatted_response += "\n✅ *Information verified from trusted medical sources*"
        
    else:
        # Add AI-generated disclaimer
        formatted_response += "\n\n---\n\n"
        formatted_response += "### ⚠️ Disclaimer\n\n"
        formatted_response += "This response is **AI-generated** and not based on specific verified sources in our database. "
        formatted_response += "This is general information only. **Always consult qualified healthcare professionals** "
        formatted_response += "for medical advice, diagnosis, or treatment.\n\n"
        formatted_response += "❌ *No verified citations available*"
    
    return formatted_response


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main application"""
    
    st.set_page_config(
        page_title="Smart Healthcare Assistant",
        page_icon="🩺",
        layout="centered"
    )
    
    st.title("🩺 Smart Healthcare Assistant")
    st.caption("AI-powered medical information with verified citations")
    
    # Disclaimer banner
    st.info(
        "⚠️ **Medical Disclaimer:** This assistant provides general health information and is NOT a substitute for "
        "professional medical advice, diagnosis, or treatment. Always consult your physician or qualified health provider."
    )
    
    # Initialize session state with memory
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
    
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = load_vectorstore()
    
    # Clear button at the top
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.rerun()
    
    st.divider()
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    user_input = st.chat_input("Ask about symptoms, conditions, medications, treatments...")
    
    if user_input:
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching medical knowledge base..."):
                try:
                    # Generate response
                    response_data = generate_response(
                        user_input,
                        st.session_state.vectorstore,
                        st.session_state.memory
                    )
                    
                    # Format with citations
                    formatted_response = format_response_with_citations(response_data)
                    
                    # Display response
                    st.markdown(formatted_response)
                    
                    # Show mode badge
                    if response_data['mode'] == 'rag':
                        st.success(f"✅ Verified sources used ({response_data['metadata']['num_sources']})")
                    else:
                        st.warning("⚠️ AI-generated (no verified sources)")
                    
                    # Save to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": formatted_response
                    })
                    
                except Exception as e:
                    error_msg = f"⚠️ Error: {str(e)}"
                    st.error(error_msg)
                    print(f"\n❌ ERROR: {e}")
                    import traceback
                    traceback.print_exc()


if __name__ == "__main__":
    main()