import torch
from langchain.embeddings import HuggingFaceInstructEmbeddings
import os
import torch
from langchain import PromptTemplate
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain
from transformers import AutoTokenizer, pipeline, AutoModelForSeq2SeqLM
from transformers import BitsAndBytesConfig
from langchain import HuggingFacePipeline
import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device


model_name = 'hkunlp/instructor-base'

embedding_model = HuggingFaceInstructEmbeddings(
    model_name = model_name,
    model_kwargs = {"device" : device}
)

# Reading the prompt template from the file
with open('D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/data/prompt_template.txt', 'r') as file:
    prompt_template = file.read().strip()

# Now you can use the prompt template in your code
PROMPT = PromptTemplate.from_template(template=prompt_template)


#calling vector from local
vector_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/vector-store'
db_file_name = 'nlp_stanford'

vectordb = FAISS.load_local(
    folder_path = os.path.join(vector_path, db_file_name),
    embeddings = embedding_model,
    index_name = 'nlp' #default index
)   
retriever = vectordb.as_retriever()

model_id = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/models_fast_chat/fastchat-t5-3b-v1.0'

tokenizer = AutoTokenizer.from_pretrained(
    model_id)

tokenizer.pad_token_id = tokenizer.eos_token_id

offload_folder = 'offload'
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_id,
    device_map = 'auto',
    offload_folder=offload_folder
)

pipe = pipeline(
    task="text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens = 256,
    model_kwargs = {
        "temperature" : 0,
        "repetition_penalty": 1.5
    }
)

llm = HuggingFacePipeline(pipeline = pipe)

doc_chain = load_qa_chain(
    llm = llm,
    chain_type = 'stuff',
    prompt = PROMPT,
    verbose = True
)

query = "What is AIT?"
input_document = retriever.get_relevant_documents(query)

doc_chain({'input_documents':input_document, 'question':query})

