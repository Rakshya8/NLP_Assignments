# Assignment 3:  Translate your own language

Creating language translator which can convert from English to Nepali

## Performance Table

| Attention Variant | Training Loss | Training PPL | Validation Loss | Validation PPL      |
|-------------------|---------------|--------------|------------------|--------------------|
| General           | 5.415         | 224.67       | 4.900            |  132.231           |
| Multiplicative    | 5.307         | 201.66       | 4.862            | 129.31             |
| Additive          | 5.171         | 176.082      | 4.731            | 113.436            |

| Attention Variant | Average Time per epoch | Overall time | Inference Time | Model Size (MB)|    Test Loss      |  Test Perplexity     |
|-------------------|------------------------|--------------|----------------|----------------|-------------------|----------------------|
| General           |         198.01         |     0.0316   |      0.0101    |  52.9          |    4.842          |        126.783       |
| Multiplicative    |         195.57         |     0.0312   |      0.0138    |  27.71         |    4.753          |        115.899       | 
| Additive          |         229.69         |     0.036    |      0.0229    |  27.71         |    4.788          |        120.098       |


## Performace Graph

### General Attention
![image](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/a97eeaad-a75f-47a8-a466-f575560f5ab1)

### Multiplicative Attention
![image](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/24af92b7-e7d1-4f1a-b0ce-039215a46973)

### Additive Attention
![image](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/fff3ac78-2130-4b15-b97a-bd5427ee3d43)

## Instructions to run website:
Navigate to the project folder and on terminal run: 
1. cd app
2. python index.py
3. http://127.0.0.1:5000/ ( to navigate to home page)
4. To view app click on Project list --> a3 from nav or navigate to http://127.0.0.1:5000/a3
![image](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/45ea8071-1046-44de-96b5-26b63a17f468)


   
