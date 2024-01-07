from dotenv import load_dotenv
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI, ChatCohere
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.llms import HuggingFaceHub
from .ingest import create_vector_db
from pathlib import Path
import os
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from aiocache import cached


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

prompt_template = """
                    You are a concise Humberside Fire Service Chatbot. Provide short, to-the-point fire safety and prevention tips. Keep answers brief unless further detail is requested. 
                    Do not elaborate unnecessarily. Use the context to answer the question succinctly.
                    Ensure that every answer is directly relevant to the following context, and if unsure about an answer, simply state that you don't know. 
                    Do not attempt to provide a response based on guesswork. Utilize the provided context to answer the question at the end concisely.
                    {context}
                    Question: {question}
                    Brief Answer:
                    """
                
                          
qa_prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])

@cached()
async def get_vectorstore():
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
    embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
    return vectorstore


def get_conversation_chain_openai(vectorstore):
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.5)

    memory = ConversationBufferMemory(memory_key='chat_history',input_key="question", output_key='answer', return_messages=True)

    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        question_generator=question_generator,
        combine_docs_chain=doc_chain
    )
    
    return conversation_chain

def get_conversation_chain_ChatCohere(vectorstore):
    llm = ChatCohere()

    memory = ConversationBufferMemory(memory_key='chat_history',input_key="question", output_key='answer', return_messages=True)
    
    
    question_generator = LLMChain(llm=llm, prompt=condense_question_prompt_template, memory=memory)
    doc_chain = load_qa_chain(llm, chain_type="stuff", prompt=qa_prompt)

    conversation_chain = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        memory=memory,
        question_generator=question_generator,
        combine_docs_chain=doc_chain
    )
    
    return conversation_chain

def get_conversation_chain_flan(vectorstore):
    
    llm=HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5})

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        verbose=True,
        memory=memory
    )

    return conversation_chain