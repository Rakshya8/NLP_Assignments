# Import necessary modules
from flask import Flask, render_template, request
import pickle
import numpy as np
from heapq import nlargest
import torch, torchtext
from lstm import LSTMLanguageModel
from Seq2Seq import Seq2SeqTransformer 
from Encoder import Encoder
from Decoder import Decoder, DecoderLayer
from Encoder_layer import EncoderLayer
from Feed_foreward import PositionwiseFeedforwardLayer
from Additive_attention import AdditiveAttention
from Mutihead_attention import MultiHeadAttentionLayer
from Bert import BERT, calculate_similarity
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece
from torch.nn.utils.rnn import pad_sequence
from PyPDF2 import PdfReader
# import fitz 
import io
import pandas as pd
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.matcher import Matcher
from sklearn.metrics.pairwise import cosine_similarity
import csv
from transformers import BertTokenizer

# Initialize Flask app
app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

#make our work comparable if restarted the kernel
SEED = 1234
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Load all embedding dictionaries from pickled files
embedding_dicts = {}

# Use GPU if available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize a basic English tokenizer from torchtext
tokenizer = torchtext.data.utils.get_tokenizer('basic_english')

# Load embeddings from pickled files for different embedding types
for embedding_type in ['glove', 'skipgram_positive', 'skipgram_negative']:
    file_path = f'../Jupyter Files//model//embed_{embedding_type}.pkl'
    
    with open(file_path, 'rb') as pickle_file:
        embedding_dicts[embedding_type] = pickle.load(pickle_file)

# Load the vocabulary from the saved file
with open('../Jupyter Files//model/vocab_lm.pkl', 'rb') as f:
    loaded_vocab = pickle.load(f)

load_path = '../Jupyter Files/model/additive_Seq2SeqTransformer.pt'
params, state = torch.load(load_path)
model3 = Seq2SeqTransformer(**params, device=device).to(device)
model3.load_state_dict(state)

# Load the trained LSTM language model
model_path_2 = '../Jupyter Files/model/best-val-lstm_lm.pt'
vocab_size = len(loaded_vocab)
emb_dim = 1024
hid_dim = 1024
num_layers = 2
dropout_rate = 0.65             
lr = 1e-3   
lstm_model = LSTMLanguageModel(vocab_size, emb_dim, hid_dim, num_layers, dropout_rate).to(device)
lstm_model.load_state_dict(torch.load(model_path_2, map_location=device))

# List to store previous queries
previous_queries = []
token_transform = {}

# # Specify the file path from where you want to load the vocab_transform
# vocab_transform_path = '../Jupyter Files/model/vocab'

# # Load the vocab_transform using pickle
# vocab_transform = torch.load(vocab_transform_path)

# nlp = spacy.load('en_core_web_md')
# skill_path = '../Jupyter Files/data/skills.jsonl'
# ruler = nlp.add_pipe("entity_ruler")
# ruler.from_disk(skill_path)

# load the model and all its hyperparameters
save_path = '../Jupyter Files/model/bert.pt'
params, state = torch.load(save_path)
model_bert = BERT(**params, device=device).to(device)
model_bert.load_state_dict(state)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# define mean pooling function
def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool

def configurations(u,v):
    # build the |u-v| tensor
    uv = torch.sub(u, v)   # batch_size,hidden_dim
    uv_abs = torch.abs(uv) # batch_size,hidden_dim

    # concatenate u, v, |u-v|
    x = torch.cat([u, v, uv_abs], dim=-1) # batch_size, 3*hidden_dim
    return x

def cosine_similarity(u, v):
    dot_product = np.dot(u, v)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    similarity = dot_product / (norm_u * norm_v)
    return similarity




def preprocessing(sentence):
    stopwords    = list(STOP_WORDS)
    doc          = nlp(sentence)
    clean_tokens = []
    
    for token in doc:
        if token.text not in stopwords and token.pos_ != 'PUNCT' and token.pos_ != 'SYM' and \
            token.pos_ != 'SPACE':
                clean_tokens.append(token.lemma_.lower().strip())
                
    return " ".join(clean_tokens)

def get_entities(resume):
    
    doc = nlp(resume)

    entities={}
    
    for entity in doc.ents:
        if entity.label_ in entities:
            entities[entity.label_].append(entity.text)
        else:
            entities[entity.label_] = [entity.text]
    for ent_type in entities.keys():
        entities[ent_type]=', '.join(unique_entities(entities[ent_type]))
    return entities

def unique_entities(x):
    return list(set(x))


# Function to calculate cosine similarity between two vectors
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

# Function to find the top N words most similar to a target word using cosine similarity
def find_next_10_cosine_words_for_word(target_word, embeddings, top_n=10):
    if target_word not in embeddings:
        return ["Word not in Corpus"]

    target_vector = embeddings[target_word]
    cosine_similarities = [(word, cosine_similarity(target_vector, embeddings[word])) for word in embeddings.keys()]
    top_n_words = nlargest(top_n + 1, cosine_similarities, key=lambda x: x[1])

    # Exclude the target word itself
    top_n_words = [word for word, _ in top_n_words if word != target_word]

    return top_n_words[:10]

# Function to generate text based on a given prompt
def generate_text(prompt, max_seq_len, temperature, model, tokenizer, vocab, device, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    model.eval()
    tokens = tokenizer(prompt)
    indices = [vocab[t] for t in tokens]
    batch_size = 1
    hidden = model.init_hidden(batch_size, device)
    with torch.no_grad():
        for i in range(max_seq_len):
            src = torch.LongTensor([indices]).to(device)
            prediction, hidden = model(src, hidden)
            
            # prediction: [batch size, seq len, vocab size]
            # prediction[:, -1]: [batch size, vocab size] # probability of last vocab
            
            probs = torch.softmax(prediction[:, -1] / temperature, dim=-1)  
            prediction = torch.multinomial(probs, num_samples=1).item()    
            
            while prediction == vocab['<unk>']:  # if it is unk, we sample again
                prediction = torch.multinomial(probs, num_samples=1).item()

            if prediction == vocab['<eos>']:  # if it is eos, we stop
                break

            indices.append(prediction)  # autoregressive, thus output becomes input

    itos = vocab.get_itos()
    tokens = [itos[i] for i in indices]
    return tokens

TRG_LANGUAGE ='ne'
SRC_LANGUAGE='en'

# Function to preprocess a source sentence (tokenization, normalization, etc.)
def preprocess_src_sentence(sentence, lang):
    token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
    return {lang: token_transform[lang](sentence.lower())}

# Function to preprocess a target sentence (tokenization, normalization, etc.)
def preprocess_trg_sentence(sentence, lang):
    token_transform["ne"] = WordPiece()
    return {lang: token_transform[lang].encode(sentence.lower()).tokens}

# helper function to club together sequential operations
def sequential_transforms(*transforms):
    """
    Combine multiple text transformations into a single function.

    Args:
    *transforms: Variable number of text transformation functions.

    Returns:
    func: A function that applies each transformation sequentially.
    """
    def func(txt_input):
        """
        Apply sequential transformations to the input text.

        Args:
        txt_input: Input text to be transformed.

        Returns:
        Transformed text after applying each transformation.
        """
        for transform in transforms:
            try:
                txt_input = transform(txt_input)
            except:
                # If an exception occurs, assume it's an encoding and use encode function
                txt_input = transform.encode(txt_input).tokens
        return txt_input
    return func

def tensor_transform(token_ids):
    """
    Add start and end tokens to a sequence of token IDs.

    Args:
    token_ids: Sequence of token IDs.

    Returns:
    Tensor with start token (2), token_ids, and end token (2).
    """
    return torch.cat((torch.tensor([2]), torch.tensor(token_ids), torch.tensor([2])))


# # src and trg language text transforms to convert raw strings into tensors indices
# text_transform = {}
# token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
# token_transform["ne"] = WordPiece()
# for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
#     text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
#                                                vocab_transform[ln], #Numericalization
#                                                tensor_transform)

max_seq_length = 1024

# Route for the home page
@app.route('/')
def home():
    return render_template('pages/index.html', previous_queries=previous_queries)

# Route for the first functionality (a1) page
@app.route('/a1', methods=['POST','GET'])
def a1():
    similar_words = []

    if request.method == "POST":
        search_query = request.form['search_query']
        selected_embedding = request.form['embedding_type']
        
        # Use the selected embedding
        embedding_dict = embedding_dicts.get(selected_embedding, {})
        
        if search_query:
            # Find the most similar words using chosen model
            similar_words = find_next_10_cosine_words_for_word(search_query, embedding_dict, top_n=10)
            
            # Save the query to the list of previous queries
            previous_queries.append(search_query)

    return render_template('pages/a1.html', results=similar_words, previous_queries=previous_queries)

# Route for the second functionality (a2) page
@app.route('/a2', methods=['POST', 'GET'])
def a2():
    generated_texts = []

    if request.method == "POST":
        prompt = request.form['prompt_a2']  # Change the ID for clarity
        max_seq_len = 30
        seed = 0

        # Load the model using pickle
        with open('../Jupyter Files/model/lstm_model.pkl', 'rb') as f:
            lstm_model = pickle.load(f)
        lstm_model.load_state_dict(torch.load(model_path_2, map_location=device))
        lstm_model.to(device)
        lstm_model.eval()

        if prompt:
            # Define temperature values
            temperatures = [0.5, 0.7, 0.75, 0.8, 1.0]

            # Generate text for each temperature
            for temperature in temperatures:
                generated_text = generate_text(prompt, max_seq_len, temperature, lstm_model, tokenizer, loaded_vocab, device, seed)
                generated_texts.append({
                    'temperature': temperature,
                    'generated_text': ' '.join(generated_text)
                })

            # Save the query to the list of previous queries
            previous_queries.append(prompt)

    return render_template('pages/a2.html', generated_texts=generated_texts, previous_queries=previous_queries)

# Route for the third functionality (a3) page
# Route for the translation page (a3)
# ...

# Route for the translation page (a3)
@app.route('/a3', methods=['POST', 'GET'])
def a3():
    # Initialize translation result
    translation_result = None

    # Check if the request method is POST
    if request.method == "POST":
        # Get the input sentence from the form
        input_sentence = request.form['input_sentence']

        # Check if the input sentence is not empty
        if input_sentence:
            # Load the pre-trained Seq2Seq model
            load_path = '../Jupyter Files/model/additive_Seq2SeqTransformer.pt'
            params, state = torch.load(load_path)
            model3 = Seq2SeqTransformer(**params, device=device).to(device)
            model3.load_state_dict(state)
            model3.eval()

            # Print the input sentence
            print(input_sentence)

            # Tokenize and transform the input sentence to tensors
            input = text_transform[SRC_LANGUAGE](input_sentence).to(device)
            print("==",input)
            output = text_transform[TRG_LANGUAGE]("").to(device)
            input = input.reshape(1,-1)
            output = output.reshape(1,-1)

            # Perform model inference
            with torch.no_grad():
                output, _ = model3(input, output)

            # Process the model output
            output = output.squeeze(0)
            output = output[1:]
            print(output)
            output_max = output.argmax(1)
            print("OutputMax",output_max)
            mapping = vocab_transform[TRG_LANGUAGE].get_itos()

            # Save the input sentence to the list of previous queries
            previous_queries.append(input_sentence)
            translation_result = []

            # Process the output tokens
            for token in output_max:
                token_str = mapping[token.item()]
                if token_str not in ['[CLS]', '[SEP]', '[EOS]','<eos>']:
                    translation_result.append(token_str)
                    print(translation_result)

            # Join the list of tokens into a single string
            translation_result = ' '.join(translation_result)

    # Render the translation result in the HTML template
    return render_template('pages/a3.html', translation_result=translation_result, previous_queries=previous_queries)


# @app.route('/a4', methods=['GET', 'POST'])
# def a4():
#     results=[]
#     if request.method == 'POST':
#         # Handle file upload
#         uploaded_files = request.files.getlist('fileInput')
#         for file in uploaded_files:
#             # Handle file upload
#             # Handle file upload
#             pdf_content = file.read()
#             pdf_document = fitz.open(stream=io.BytesIO(pdf_content), filetype="pdf")
#             if pdf_document:
#             # Process the uploaded PDF file
#                 text = ""
#                 for page_num in range(pdf_document.page_count):
#                     page = pdf_document[page_num]
#                     text += page.get_text()
#                 text = preprocessing(text)
#                 result = get_entities(text)
#                 results.append(result)
#                 print("Results",results)
#                 matcher = Matcher(nlp.vocab)
#                 pattern = [
#                     {"POS": "PROPN",  # person's name should be a proper noun
#                     "OP": "{2}",  # person's name usually consists of 2 parts; first name and last name (in some scenario, 3 if a person has middle name)
#                     "ENT_TYPE": "PERSON"  # person's name is of 'PERSON' entity type|
#                     },
#                 ]
#                 matcher.add("PERSON NAME", [pattern], greedy="LONGEST")
#                 doc = nlp(text)
#                 matches = matcher(doc)
#                 matches.sort(key = lambda x: x[1])

#                 person_names = []

#                 for match in matches:
#                     person_names.append((str(doc[match[1]:match[2]]),
#                                         nlp.vocab.strings[match[0]]))

#                 person_names = list(set(person_names))
#                 matcher.add("PERSON", [[{"POS": "PROPN", "OP": "{2}", "ENT_TYPE": "PERSON"}]], greedy="LONGEST")
#                 matcher.add("EMAIL", [[{"LIKE_EMAIL": True}]], greedy="LONGEST")
#                 matcher.add("URL", [[{"LIKE_URL": True}]], greedy="LONGEST")
#                 matcher.add("PHONE NUMBER", [
#                     [{"ORTH": {"in": ["(", "["]}, "is_digit": True}, {"SHAPE": "dddd"}, {"ORTH": {"in": [")", "]"]}}, {"SHAPE": "dddd"}, {"SHAPE": "dddd"}],
#                     [{"ORTH": {"in": ["(", "["]}, "is_digit": True}, {"SHAPE": "ddd"}, {"ORTH": {"in": [")", "]"]}}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}],
#                     [{"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "ddd"}, {"ORTH": "-"}, {"SHAPE": "dddd"}],
#                     [{"SHAPE": "ddd"}, {"SHAPE": "ddd"}, {"SHAPE": "dddd"}],
#                     ])

#                 print(results)
#                 extracted_info = result  # Assign the results list to extracted_info


#                 return render_template('pages/a4.html', extracted_info=extracted_info)
            

#     return render_template('pages/a4.html', extracted_info=None)

@app.route('/a5', methods=['POST', 'GET'])
def a5():
    # Initialize translation result
    similarity = None

    # Check if the request method is POST
    if request.method == "POST":
        # Get the input sentence from the form
        input_sentence_1 = request.form['prompt_a5_1']
        input_sentence_2 = request.form['prompt_a5_2']

        # Check if the input sentence is not empty
        if input_sentence_1 and input_sentence_2:
            # Load the pre-trained Seq2Seq model
            load_path = '../Jupyter Files/model/bert.pt'
            params, state = torch.load(load_path)
            model_bert = BERT(**params, device=device).to(device)
            model_bert.load_state_dict(state)

            # Print the input sentence
            print(input_sentence_1)
            print(input_sentence_2)

            score = calculate_similarity(model_bert, tokenizer, params['max_len'], input_sentence_1, input_sentence_2, device)

            # Save the input sentence to the list of previous queries
            previous_queries.append((input_sentence_1, input_sentence_2))

    # Render the translation result in the HTML template
    return render_template('pages/a5.html', generated_scores=score, previous_queries=previous_queries)


    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
