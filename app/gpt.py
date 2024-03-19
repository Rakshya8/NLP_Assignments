import torch
import os
import re
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from transformers import AutoTokenizer, AutoModelForCausalLM,pipeline
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain import HuggingFacePipeline

def answer_question(query):
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Define model and tokenizer names
    model_name = 'hkunlp/instructor-base'
    model_id = 'gpt2'

    # Initialize embeddings
    embedding_model = HuggingFaceInstructEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

    # Define the prompt template
    prompt_template = """
        I'm your friendly AIT chatbot named ChakyBot, here to assist Chaky and Gun with any questions they have about AIT. 
        If you're curious about anything about AIT, feel free to ask any questions you may have. 
        Whether it's about general, or specific topics. 
        I'm here to help break down complex concepts into easy-to-understand explanations.
        Just let me know what you're wondering about, and I'll do my best to guide you through it!
        {context}
        Question: {question}
        Answer:
        """.strip()

    # Create a PromptTemplate
    PROMPT = PromptTemplate.from_template(template=prompt_template)

    # Load vector database
    vector_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/vector-store'
    db_file_name = 'nlp_stanford'

    vectordb = FAISS.load_local(
        folder_path=os.path.join(vector_path, db_file_name),
        embeddings=embedding_model,
        index_name='nlp'
    )
    retriever = vectordb.as_retriever()

    # Load tokenizer and model for text generation
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    # Create a pipeline for text generation
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device
    )

    llm = HuggingFacePipeline(pipeline=pipe)
    doc_chain = load_qa_chain(
        llm=llm,
        chain_type='stuff',
        prompt=PROMPT,
        verbose=True
    )
    question_generator = LLMChain(
        llm=llm,
        prompt=CONDENSE_QUESTION_PROMPT,
        verbose=True
    )
    memory = ConversationBufferWindowMemory(
        k=3,
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    chain = ConversationalRetrievalChain(
        retriever=retriever,
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        memory=memory,
        verbose=True,
        get_chat_history=lambda h: h
    )

    # Get the answer
    answer = chain({"question": query})
    answer_text = answer['answer']

    # Extract unique source titles and links
    unique_sources = set()
    unique_links = set()

    for doc in answer['source_documents']:
        source_title = doc.metadata['source']
        unique_sources.add(source_title)

        links = re.findall(r'(https?://\S+)', doc.page_content)
        unique_links.update(links)

    return {
        'answer': answer_text,
        'unique_sources': list(unique_sources),
        'unique_links': list(unique_links)
    }

