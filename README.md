# Assignment 4:  Resume Parser

Creating resume parser that uses spacy to get key information from resume/resumes

### 2.1 Document Loaders 
Use document loaders to load data from a source as Document's. A Document is a piece of text and associated metadata. For example, there are document loaders for loading a simple .txt file, for loading the text contents of any web page, or even for loading a transcript of a YouTube video.

[PDF Loader](https://python.langchain.com/docs/modules/data_connection/document_loaders/pdf)

## Relevant Sources Related to AIT

The dataset containing relevant sources related to the Asian Institute of Technology (AIT) has been sourced from the official AIT website ([www.ait.ac.th](www.ait.ac.th)). The website serves as a reputable repository of information about AIT, including documents, publications, and other resources. The dataset includes both documents and website links related to AIT.

### Dataset Source
The dataset has been sourced directly from the official AIT website, ensuring the authenticity and reliability of the information provided.

### Relevant Documents
1. **AIT Annual Reports**
   - Description: Annual reports provide comprehensive information about AIT's activities, achievements, and financial status.
   - Source: [AIT Annual Reports](www.ait.ac.th/reports/annual)

2. **Research Publications**
   - Description: Research publications showcase the academic and research contributions of AIT faculty and students.
   - Source: [AIT Research Publications](www.ait.ac.th/research/publications)

### Relevant Websites
1. **AIT Official Website**
   - Description: The official website of the Asian Institute of Technology contains a wealth of information about AIT, including academic programs, research centers, and news updates.
   - Source: [www.ait.ac.th](www.ait.ac.th)

2. **AIT About**
   - Description: The about us about AIT
   - Source: [AIT About](https://ait.ac.th/about/)

3. **AIT Housing**
   - Description: The housing information about AIT
   - Source: [AIT Housing](https://ait.ac.th/student-housing/)

4. **DSAI**
   - Description: The information about data science and artificial ingtelligence course in AIT
   - Source: [AIT DASI](https://ait.ac.th/program/data-science-and-ai/)

5. **Athletics**
   - Description: The information about atletics in AIT
   - Source: [AIT Athletics](https://ait.ac.th/athletcis/)

### Dataset Usage
Researchers, students, and stakeholders interested in AIT-related information can utilize this dataset for various purposes, including academic research, institutional analysis, and decision-making processes.

### Credits
The dataset has been compiled from the official AIT website ([www.ait.ac.th](www.ait.ac.th)). Proper credit is given to AIT for providing access to relevant information and resources related to the institution.

## Analysis of Model Performance in Information Retrieval

The performance of the AIT GPT-2 model in retrieving information can be assessed based on several factors, including the quality of responses, relevance of generated content, and efficiency in providing accurate information. However, the model's performance may be hampered by several factors:

### Limited Training Data:
The AIT GPT-2 model may not have been trained on a sufficiently diverse and comprehensive dataset related to the specific domain of interest. Limited training data can result in the model's inability to generate relevant and accurate responses to queries related to specialized topics, such as those specific to AIT.

### Domain Specificity:
GPT-2, including the AIT version, is a general-purpose language model trained on a diverse range of internet text. While it can generate coherent text on a wide array of topics, its performance may suffer when tasked with domain-specific queries, such as those related to academic institutions like AIT. The model may lack specialized knowledge or context to provide accurate responses in such cases.

### Fine-tuning and Specialization:
To enhance the model's performance in retrieving information related to AIT, fine-tuning on a dataset specifically curated from AIT-related documents, publications, and websites could be beneficial. Fine-tuning allows the model to adapt to the nuances and vocabulary of the target domain, improving its ability to generate relevant and accurate responses.

### Evaluation Metrics:
Performance evaluation metrics, such as precision, recall, and F1 score, can be employed to quantitatively assess the model's performance in information retrieval tasks. These metrics provide insights into the model's accuracy, completeness, and relevance of generated responses compared to ground truth data.

### User Feedback and Iterative Improvement:
Continuous evaluation of the model's performance through user feedback and iterative refinement is essential for enhancing its effectiveness in information retrieval. Soliciting feedback from users interacting with the system can help identify areas for improvement and guide future model development efforts.

In conclusion, while the AIT GPT-2 model serves as a valuable tool for natural language processing tasks, including information retrieval, its performance may be hindered by factors such as limited training data, domain specificity, and the need for fine-tuning. Addressing these challenges through targeted training data collection, domain adaptation techniques, and continuous evaluation can lead to improvements in the model's ability to retrieve accurate and relevant information related to AIT.

## Addressing Issues of Unrelated Information and Relevant Document Provision in AIT GPT-2 Model

The AIT GPT-2 model faces challenges in providing only relevant documents and avoiding unrelated information. Here are some strategies to address these issues:

### 1. Data Filtering:
Utilize a pre-processed dataset that includes only relevant documents from reputable sources such as VectorDB. By filtering the dataset based on relevance to AIT and removing irrelevant or outdated documents, the model's training data can better reflect the domain-specific context.

### 2. Domain-Specific Fine-Tuning:
Perform fine-tuning of the GPT-2 model on a specialized dataset comprising AIT-related documents and sources. Fine-tuning allows the model to adapt to the specific language patterns and topics relevant to AIT, resulting in more accurate and contextually appropriate responses.

### 3. Contextual Understanding:
Enhance the model's contextual understanding by incorporating techniques like context windowing and attention mechanisms. These techniques enable the model to consider a broader context of preceding text, improving its ability to generate coherent responses that align with the user's query.

### 4. Relevance Scoring:
Implement a relevance scoring mechanism to evaluate the relevance of generated documents. By assigning scores based on factors such as semantic similarity and topical alignment with the query, the model can prioritize and present the most relevant documents to the user while filtering out irrelevant ones.

### 5. User Feedback Loop:
Establish a feedback loop where users can provide input on the relevance and usefulness of generated documents. By collecting user feedback and iteratively refining the model based on this input, the system can continuously improve its document recommendation capabilities and minimize the provision of unrelated information.

### 6. Quality Assurance:
Incorporate quality assurance measures to validate the accuracy and reliability of document recommendations. This may involve human review processes or automated checks to verify the correctness and relevance of the documents before presenting them to users.

### 7. Continuous Monitoring and Iterative Improvement:
Regularly monitor the model's performance and effectiveness in providing relevant documents. Analyze user interactions and feedback to identify areas for improvement and iteratively update the model and dataset accordingly, ensuring that it remains up-to-date and capable of delivering high-quality recommendations.

By implementing these strategies, the AIT GPT-2 model can enhance its ability to provide only relevant documents from VectorDB while minimizing the occurrence of unrelated information, thereby improving the overall user experience and utility of the system.


## Web Application Documentation

This documentation outlines how the web application interfaces with the language model to generate coherent responses and retrieve source documents.

### 1. Overview
The web application leverages a Flask framework to create a user interface for interacting with a language model. Users can input queries or messages, and the application generates responses using the language model. Additionally, it provides related source documents based on the user's input.

### 2. Components
- **Flask**: Flask is a micro web framework for Python, used to build the web application.
- **Language Model (GPT)**: The application interfaces with a language model (such as OpenAI's GPT) to generate responses to user queries.
- **HTML/CSS Templates**: HTML and CSS templates are used to structure and style the web pages.
- **Python Backend**: Python code handles the server-side logic, including routing, request handling, and interaction with the language model.
- **Source Documents**: Source documents contain additional information relevant to user queries and are provided as links in the application.

### 3. Interaction Flow
1. **User Input**: Users input their queries or messages via a text input field on the web application.
2. **Request Handling**: The Flask backend handles the user's input through defined routes and methods.
3. **Language Model Processing**: The user input is passed to the language model (e.g., GPT) for processing and generating a response.
4. **Response Rendering**: The generated response is sent back to the web application.
5. **Display**: The response, along with related source documents, is displayed on the web page for the user to view.

### 4. Implementation Details
- **Routes**: Flask routes are defined for different pages and functionalities, such as the home page and processing user input.
- **Request Handling**: Flask's `request` object is used to access user input data, such as form submissions.
- **Language Model Interaction**: Python functions interface with the language model to send user queries and receive generated responses.
- **HTML/CSS Styling**: HTML templates structure the content, while CSS stylesheets provide visual formatting.
- **Dynamic Content**: The application dynamically updates content based on user interactions without page reloads using AJAX or form submissions.

### 5. Retrieving Source Documents
- **Related Articles**: Source documents related to user queries are stored locally or accessed from external sources. These documents are linked within the application for users to explore further.
- **Link Formatting**: Source document links are formatted in the HTML response, making them clickable for easy access.

### 6. Future Enhancements
- **Improved Response Generation**: Enhance the language model's capabilities to generate more accurate and coherent responses.
- **Enhanced User Experience**: Implement features such as autocomplete, suggestions, and multi-step interactions to improve user experience.
- **Integration with External APIs**: Integrate with external data sources or APIs to retrieve 
additional relevant information.
- **User Authentication and Personalization**: Implement user authentication and personalized recommendations for a tailored user experience.

### 7. Conclusion
The web application provides users with an interface to interact with a language model, generating coherent responses and providing access to related source documents. By leveraging Flask, HTML/CSS, and Python, the application offers a seamless user experience for querying and accessing information.


## Instructions to run website:
Navigate to the project folder and on terminal run: 
1. cd app
2. python index.py
3. http://127.0.0.1:5000/ ( to navigate to home page)
4. To view app click on Project list --> a7 from nav or navigate to http://127.0.0.1:5000/a7
![a7](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/22cee159-c53a-40f4-8999-404ec3d3d9f3)

