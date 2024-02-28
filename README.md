# Assignment 5:  Sentence Embedding with BERT

Creating senetence embedding model that finds cosine similarity between sentences.

## Detailed evaluation of your sentence transformer model, considering different types of sentences and relevant metrics


| Model Type      | Training Loss with CNN | Training Loss with SNLI + MLI | Cosine Similarity with SNLI and MLI Dataset | Inference Cosine Similarity (Positive Sentences) | Inference Cosine Similarity (Negative Sentences) |
|-----------------|---------------------------|-------------------------------|----------------------------------------------|--------------------------------------------------|--------------------------------------------------|
| Custom Model    | 13                    | 2.25                          |          0.9983                             |0.9989                                            |0.9983                                              |

## Comparision of Custom Model with Pretrained Model

| Model Type      | Inference Cosine Similarity (with similar sentence)| Inference Cosine Similarity (with dissisimilar sentence)|  
|-----------------|---------------------------|-----------------------------|
| Custom Model    | 0.9989                    | 0.9983                      |
| Pretrained Model (AutoModel)| 0.999         | 0.8057                      |



## Instructions to run website:
Navigate to the project folder and on terminal run: 
1. cd app
2. python index.py
3. http://127.0.0.1:5000/ ( to navigate to home page)
4. To view app click on Project list --> a5 from nav or navigate to http://127.0.0.1:5000/a5

![Recording 2024-02-28 150250](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/5937512d-bdcb-4681-88ff-7c35bcac3d9b)




   
