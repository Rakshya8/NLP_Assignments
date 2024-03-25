# Import necessary modules
from flask import Flask, render_template, request
# from gpt import answer_question
from transformers import AutoModelForCausalLM,AutoTokenizer,pipeline

# Initialize Flask app
app = Flask(__name__)


# Route for the home page
@app.route('/')
def home():
    return render_template('pages/index.html')

def instruction_prompt(instruction, prompt_input=None):
	
	if prompt_input:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{prompt_input}

### Response:
""".strip()
			
	else:
		return f"""
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
""".strip()


# @app.route('/a7', methods=['POST', 'GET'])
# def a7():
#     user_message = ''
#     if request.method == "POST":
#         user_message = request.form.get('user_message', '')  # Use 'user_message' instead of 'query'
#         # Process the user's message and generate a response
#         generated_response = answer_question(user_message)  # Use 'answer_question' instead of 'generate_response'
#         return render_template('pages/a7.html', generated_response=generated_response, user_message=user_message)

#     return render_template('pages/a7.html', user_message=user_message)
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
      
@app.route('/a8', methods=['POST', 'GET'])
def a8():
    user_message = ''
    if request.method == "POST":
        user_message = request.form.get('user_message', '')
        instruction = request.form.get('instruction','')
        output = text_generator(instruction_prompt(instruction, user_message))
        generated_response = output[0]['generated_text'].split("### Response:\n")[-1]
        return render_template('pages/a8.html', generated_response=generated_response, user_message=user_message, instruction=instruction)

    return render_template('pages/a8.html', user_message=user_message)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
