# Import necessary modules
from flask import Flask, render_template, request
import pickle
import numpy as np
from heapq import nlargest
import torch, torchtext
from lstm import LSTMLanguageModel

# Initialize Flask app
app = Flask(__name__)

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

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
