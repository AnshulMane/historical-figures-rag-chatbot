import os
from dotenv import load_dotenv
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_core.chat_history import InMemoryChatMessageHistory

# ---------------------------------------------------------
# 1. LangSmith Configuration
# ---------------------------------------------------------
# Ensure LangSmith tracing is enabled.
# NOTE: Set your actual API key in your terminal before running,
# e.g., export LANGCHAIN_API_KEY="your_api_key_here"
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# # os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key" # Best set in terminal
# if "LANGCHAIN_PROJECT" not in os.environ:
#     os.environ["LANGCHAIN_PROJECT"] = "History_Figures_Chatbot"

# ---------------------------------------------------------
# 2. PDF Ingestion & Splitting
# ---------------------------------------------------------
print("Loading and splitting PDF...")
loader = PyPDFLoader("history_figures.pdf")
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=200, 
    chunk_overlap=30,
    separator="\n"
)
docs = text_splitter.split_documents(documents)

# ---------------------------------------------------------
# 3. Vector Store Initialization
# ---------------------------------------------------------
print("Initializing Vector Store...")
embeddings = OllamaEmbeddings(model="granite-embedding:latest")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="history_figures"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# ---------------------------------------------------------
# 4. LLM Integration & Chat Logic
# ---------------------------------------------------------
print("Setting up LLM and QA Chain...")
# Using local Ollama model (llama3)
llm = Ollama(model="llama3")

# Custom Prompt Template
prompt_template = """Use the following pieces of retrieved context to answer the question about historical figures. 
If you don't know the answer, just say that you don't know. 

Context: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Initialize RetrievalQA
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Initialize Chat History
chat_history = InMemoryChatMessageHistory()

# ---------------------------------------------------------
# 5. UI Functions
# ---------------------------------------------------------
def answer_question(user_message, history_state):
    # Add user message to history
    chat_history.add_user_message(user_message)
    
    # Get response from the RAG pipeline
    response = qa_chain.invoke({"query": user_message})
    bot_reply = response.get("result", "I'm sorry, I couldn't generate an answer.")
    
    # Add AI response to history
    chat_history.add_ai_message(bot_reply)
    
    # Update Gradio history state for display
    history_state.append({"role": "user", "content": user_message})
    history_state.append({"role": "assistant", "content": bot_reply})
    return "", history_state

def clear_conversation():
    chat_history.clear()
    return []

# ---------------------------------------------------------
# 6. Gradio UI Layout
# ---------------------------------------------------------
with gr.Blocks(title="Historical Figures Chatbot") as demo:
    gr.Markdown("### Hello, I am HistoryBot, your expert on historical figures. How can I assist you today?")
    
    chatbot = gr.Chatbot(label="Conversation History")
    
    with gr.Row():
        txt_input = gr.Textbox(
            show_label=False, 
            placeholder="Ask a question about a historical figure...",
            scale=4
        )
        submit_btn = gr.Button("Submit", scale=1)
    
    clear_btn = gr.Button("Clear History")
    
    # Event wiring
    submit_btn.click(
        fn=answer_question, 
        inputs=[txt_input, chatbot], 
        outputs=[txt_input, chatbot]
    )
    # Also allow pressing Enter to submit
    txt_input.submit(
        fn=answer_question, 
        inputs=[txt_input, chatbot], 
        outputs=[txt_input, chatbot]
    )
    
    clear_btn.click(
        fn=clear_conversation, 
        inputs=None, 
        outputs=chatbot
    )

if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch() 
