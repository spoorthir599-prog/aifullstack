import gradio as gr
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

print("Gradio Version:", gr.__version__)
print("Gradio Path:", gr.__file__)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

Settings.llm = HuggingFaceLLM(
    model_name="gpt2",
    tokenizer_name="gpt2",
    context_window=1024,
    max_new_tokens=100,
)

Settings.text_splitter = SentenceSplitter(
    chunk_size=256,
    chunk_overlap=20,
)

docs = SimpleDirectoryReader(
    input_files=[r"document/day 4.pdf"]
).load_data()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection("education_docs")

vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    docs,
    storage_context=storage_context,
)

query_engine = index.as_query_engine(similarity_top_k=1)

memory = ChatMemoryBuffer.from_defaults(token_limit=1000)

chat_engine = CondenseQuestionChatEngine.from_defaults(
    query_engine=query_engine,
    memory=memory,
)

questions = 0


def chat(message):
    global questions
    questions += 1

    response = chat_engine.chat(message)
 
    dashboard = f"""Questions Asked : {questions}

Retrieved Chunks : 1

Memory Enabled : Yes

RAG : Enabled

Embedding Model:
sentence-transformers/all-MiniLM-L6-v2

LLM Model:
GPT-2"""

    return str(response), dashboard


with gr.Blocks() as demo:
    gr.Markdown("""
# Educational RAG Assistant

Ask questions from your PDF document.
""")

    chatbot = gr.Chatbot(height=500)

    msg = gr.Textbox(
        placeholder="Ask a question from your PDF..."
    )

    dashboard = gr.Textbox(
        label="Evaluation Dashboard",
        lines=8,
    )

    def respond(message, history):
        answer, stats = chat(message)

        history = history or []

        history.append({
            "role": "user",
            "content": message,
        })

        history.append({
            "role": "assistant",
            "content": answer,
        })

        return "", history, stats

    msg.submit(
        respond,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot, dashboard],
    )

demo.queue()
demo.launch(share=True)