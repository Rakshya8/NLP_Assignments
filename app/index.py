from flask import Flask, render_template, request
import pickle
import numpy as np
from heapq import nlargest

app = Flask(__name__)

# Load all embedding dictionaries from pickled files
embedding_dicts = {}

for embedding_type in ['glove', 'skipgram_positive', 'skipgram_negative']:
    file_path = f'../Jupyter Files//model//embed_{embedding_type}.pkl'
    
    with open(file_path, 'rb') as pickle_file:
        embedding_dicts[embedding_type] = pickle.load(pickle_file)

# Load the Gensim model
model_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/model_gensim.pkl'
with open(model_path, 'rb') as model_file:
    model_gensim = pickle.load(model_file)

# Previous queries list
previous_queries = []

# more formally is to divide by its norm
def cosine_similarity(A, B):
    dot_product = np.dot(A, B)
    norm_a = np.linalg.norm(A)
    norm_b = np.linalg.norm(B)
    similarity = dot_product / (norm_a * norm_b)
    return similarity

def find_next_10_cosine_words_for_word(target_word, embeddings, top_n=10):
    if target_word not in embeddings:
        return ["Word not in Corpus"]

    target_vector = embeddings[target_word]
    cosine_similarities = [(word, cosine_similarity(target_vector, embeddings[word])) for word in embeddings.keys()]
    top_n_words = nlargest(top_n + 1, cosine_similarities, key=lambda x: x[1])

    # Exclude the target word itself
    top_n_words = [word for word, _ in top_n_words if word != target_word]

    return top_n_words[:10]

@app.route('/')
def home():
    return render_template('pages/index.html', previous_queries=previous_queries)

@app.route('/a1', methods=['POST','GET'])
@app.route('/a1', methods=['GET', 'POST'])
def a1():
    similar_words = []

    if request.method == "POST":
        search_query = request.form['search_query']
        selected_embedding = request.form['embedding_type']
        
        # Use the selected embedding
        embedding_dict = embedding_dicts.get(selected_embedding, {})
        
        if search_query:
            # Find the most similar words using the Gensim model
            similar_words = find_next_10_cosine_words_for_word(search_query, embedding_dict, top_n=10)
            
            # Save the query to the list of previous queries
            previous_queries.append(search_query)

    return render_template('pages/a1.html', results=similar_words, previous_queries=previous_queries)

if __name__ == '__main__':
    app.run(debug=True)
