# Import necessary modules
from flask import Flask, render_template, request
from gpt import answer_question

# Initialize Flask app
app = Flask(__name__)


# Route for the home page
@app.route('/')
def home():
    return render_template('pages/index.html')


@app.route('/a7', methods=['POST', 'GET'])
def a7():
    if request.method == "POST":
        user_message = request.form.get('user_message', '')  # Use 'user_message' instead of 'query'
        # Process the user's message and generate a response
        generated_response = answer_question(user_message)  # Use 'answer_question' instead of 'generate_response'
        return render_template('pages/a7.html', generated_response=generated_response)

    return render_template('pages/a7.html')
    
# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
