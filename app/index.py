from flask import Flask, render_template, request
import pickle
import numpy as np
from heapq import nlargest

app = Flask(__name__)

# Load the embedding dictionary from the pickled file
embedding_dict_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/embed_skipgram_negative.pkl'
with open(embedding_dict_path, 'rb') as pickle_file:
    embedding_dict = pickle.load(pickle_file)


# Load the Gensim model
model_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/model_gensim.pkl'
with open(model_path, 'rb') as model_file:
    model_gensim = pickle.load(model_file)

#more formally is to divide by its norm
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
    # Convert all words to lowercase
    return render_template('pages/index.html')

@app.route('/a1', methods=['GET', 'POST'])
def a1():
    results = []
    if request.method=="POST":
        search_query = request.form['search_query']
        if search_query:
            # Find the most similar words using the Gensim model
            similar_words = find_next_10_cosine_words_for_word(search_query, embedding_dict, top_n=10)        

    return render_template('pages/a1.html', results=similar_words)

if __name__ == '__main__':
    app.run(debug=True)
