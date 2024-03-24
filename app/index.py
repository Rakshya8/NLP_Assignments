# Import necessary modules
from flask import Flask, render_template, request
from gpt import answer_question
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

# Initialize Flask app
app = Flask(__name__)


# Route for the home page
@app.route('/')
def home():
    return render_template('pages/index.html')


@app.route('/a7', methods=['POST', 'GET'])
def a7():
    user_message = ''
    if request.method == "POST":
        user_message = request.form.get('user_message', '')  # Use 'user_message' instead of 'query'
        # Process the user's message and generate a response
        generated_response = answer_question(user_message)  # Use 'answer_question' instead of 'generate_response'
        return render_template('pages/a7.html', generated_response=generated_response, user_message=user_message)

    return render_template('pages/a7.html', user_message=user_message)

@app.route('/a8', methods=['POST', 'GET'])
def a8():
    user_message = ''
    if request.method == "POST":
        user_message = request.form.get('user_message', '')  # Use 'user_message' instead of 'query'
        model_name_or_path = "D:/AIT/Sem2/NLP/NLP_Assignments/Jupyter Files/model/instruction_tuning"
        tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path)
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, device_map = 'auto')
        text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=500
    )
        generated_response=text_generator(user_message)
        return render_template('pages/a8.html', generated_response=generated_response, user_message=user_message)

    return render_template('pages/a8.html', user_message=user_message)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
