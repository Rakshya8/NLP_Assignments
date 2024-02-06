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
from torchtext.data.utils import get_tokenizer
from nepalitokenizers import WordPiece
from torch.nn.utils.rnn import pad_sequence


# Initialize Flask app
app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

# Load the Gensim model
model_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/model_gensim.pkl'
with open(model_path, 'rb') as model_file:
    model_gensim = pickle.load(model_file)

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

# Specify the file path from where you want to load the vocab_transform
vocab_transform_path = '../Jupyter Files/model/vocab'

# Load the vocab_transform using pickle
vocab_transform = torch.load(vocab_transform_path)

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

# Function to preprocess a sentence (tokenization, normalization, etc.)
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
    def func(txt_input):
        for transform in transforms:
            try:
                txt_input = transform(txt_input)
            except:
                txt_input = transform.encode(txt_input).tokens
        return txt_input
    return func

def tensor_transform(token_ids):
    return torch.cat((torch.tensor([2]),
                      torch.tensor(token_ids),
                      torch.tensor([2])))

# src and trg language text transforms to convert raw strings into tensors indices
text_transform = {}
token_transform["en"] = get_tokenizer('spacy', language='en_core_web_sm')
token_transform["ne"] = WordPiece()
for ln in [SRC_LANGUAGE, TRG_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], #Tokenization
                                               vocab_transform[ln], #Numericalization
                                               tensor_transform)


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
    translation_result = None

    if request.method == "POST":
        input_sentence = request.form['input_sentence']

        if input_sentence:
            # Load the model
            load_path = '../Jupyter Files/model/additive_Seq2SeqTransformer.pt'
            params, state = torch.load(load_path)
            model3 = Seq2SeqTransformer(**params, device=device).to(device)
            model3.load_state_dict(state)
            model3.eval()
            print(input_sentence)
            # Access 'en' key in token_transform
            input = text_transform[SRC_LANGUAGE](input_sentence).to(device)
            print("==",input)
            output = text_transform[TRG_LANGUAGE]("").to(device)
            input = input.reshape(1,-1)
            output = output.reshape(1,-1)
            with torch.no_grad():
                output, _ = model3(input, output)
            output = output.squeeze(0)
            output = output[1:]
            print(output)
            output_max = output.argmax(1)
            print("OutputMax",output_max)
            mapping = vocab_transform[TRG_LANGUAGE].get_itos()

            # Save the query to the list of previous queries
            previous_queries.append(input_sentence)
            translation_result = []

            for token in output_max:
                token_str = mapping[token.item()]
                if token_str not in ['[CLS]', '[SEP]', '[EOS]','<eos>']:
                    translation_result.append(token_str)
                    print(translation_result)

            # Join the list of tokens into a single string
            translation_result = ' '.join(translation_result)


    return render_template('pages/a3.html', translation_result=translation_result, previous_queries=previous_queries)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
