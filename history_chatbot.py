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


# Load PDF
print("Loading and splitting PDF...")
loader = PyPDFLoader("history_figures.pdf")
documents = loader.load()

# Split text
text_splitter = CharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30,
    separator="\n"
)
docs = text_splitter.split_documents(documents)

# Vector store
print("Initializing Vector Store...")
embeddings = OllamaEmbeddings(model="granite-embedding:latest")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    collection_name="history_figures"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

# LLM setup
print("Setting up LLM and QA Chain...")
llm = Ollama(model="llama3")

# Prompt template
prompt_template = """Use the following pieces of retrieved context to answer the question about historical figures. 
If you don't know the answer, just say that you don't know. 

Context: {context}

Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT}
)

# Chat memory
chat_history = InMemoryChatMessageHistory()

# Answer function
def answer_question(user_message, history_state):

    # Save user message
    chat_history.add_user_message(user_message)

    # Get response
    response = qa_chain.invoke({"query": user_message})
    bot_reply = response.get("result", "I'm sorry, I couldn't generate an answer.")

    # Save bot reply
    chat_history.add_ai_message(bot_reply)

    # Update UI history
    history_state.append({"role": "user", "content": user_message})
    history_state.append({"role": "assistant", "content": bot_reply})

    return "", history_state


# Clear chat
def clear_conversation():
    chat_history.clear()
    return []


# Gradio UI
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

    # Submit button
    submit_btn.click(
        fn=answer_question,
        inputs=[txt_input, chatbot],
        outputs=[txt_input, chatbot]
    )

    # Enter key submit
    txt_input.submit(
        fn=answer_question,
        inputs=[txt_input, chatbot],
        outputs=[txt_input, chatbot]
    )

    # Clear history
    clear_btn.click(
        fn=clear_conversation,
        inputs=None,
        outputs=chatbot
    )


if __name__ == "__main__":
    print("Starting Gradio Server...")
    demo.launch()