from flask import Flask, render_template, request
import pickle
app = Flask(__name__)

# Load the Gensim model
model_path = 'D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/model_gensim.pkl'
with open(model_path, 'rb') as model_file:
    model_gensim = pickle.load(model_file)


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
            similar_words = model_gensim.most_similar(search_query, topn=10)
            results = [word for word, _ in similar_words]

    return render_template('pages/a1.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
