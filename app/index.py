from flask import Flask, render_template

app = Flask(__name__)

# Sample data (you can replace this with your own data reading logic)
sample_data = [
    "Algiers Algeria Antananarivo Madagascar",
    "Algiers Algeria Apia Samoa",
    # Add more data as needed
]

@app.route('/')
def home():
    # Convert all words to lowercase
    processed_data = [line.lower().split() for line in sample_data]
    return render_template('pages/index.html', data=processed_data)

@app.route('/a1')
def a1():
    return render_template('pages/a1.html')

if __name__ == '__main__':
    app.run(debug=True)
