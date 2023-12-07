from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import HuggingFaceHub
from .ingest import create_vector_db
from pathlib import Path
import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
huggingface_token = os.getenv('HUGGINGFACE_TOKEN')

base_dir = Path(__file__).resolve().parent

DB_FAISS_PATH = index_path = os.path.join(base_dir, 'vectorstore', 'db_faiss')


_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a 
                standalone question without changing the content in given question.

                Chat History:
                {chat_history}
                Follow Up Input: {question}
                Standalone question:
            """
                
condense_question_prompt_template = PromptTemplate.from_template(_template)

prompt_template = """You are Humberside Fire Service Chatbot, providing fire safety and prevention tips and make sure you don't answer anything 
                    not related to following context. Ensure you answer with brief answers except told otherwise. You are to always provide useful information & details available in the given context. Use the following pieces of context to answer the question at the end. 
                    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
                    {context}
                    Question: {question}
                    Helpful Answer:
                """

qa_prompt = PromptTemplate(
                        template=prompt_template, input_variables=["context", "question"]
                        )


def get_vectorstore():
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
    return vectorstore


def get_conversation_chain_openai(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    # memory = ConversationBufferWindowMemory (memory_key='chat_history',k=2, return_messages=True)
    
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        get_chat_history=lambda h: h
    )
    
    return conversation_chain

def get_conversation_chain_flan(vectorstore):
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.8, "max_length":512})
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory
    )

    return conversation_chain

def get_conversation_chain_bloke(vectorstore):
    
    llm = HuggingFaceHub(repo_id="TheBloke/Llama-2-7B-Chat-GGML", model_kwargs={"temperature":0.5}, use_auth_token=huggingface_token)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory
    )
    return conversation_chain

def get_conversation_chain_llama(vectorstore):
    llm = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature":0.5}, use_auth_token=huggingface_token)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory
    )
    return conversation_chain

def get_conversation_chain_falcon(vectorstore):
    llm = HuggingFaceHub(repo_id="tiiuae/falcon-40b", model_kwargs={"temperature":0.5}, use_auth_token=huggingface_token)
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory
    )
    return conversation_chain