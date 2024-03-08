# Assignment 6:  Student Layers Initialization

Initialising a  6 layer student from 12 layer teacher

# Documentation

## Perform a detailed evaluation of your distilled student model, analyzing the impact of the initial
 layer selection (top K layers, bottom K layers, odd layers) on its performance.


|Student Layer         | Training Loss | Validation Loss | Validation Accuracy |
|---------------|---------------|------------------|----------------------|
| Top-K  Layer | 0.274         | 0.808            | 0.665                 |
| Bottom-K Layer| 0.265        | 0.808              | 0.665                  |
| Odd Layer     | 0.266           | 0.808          | 0.665                  |
| Even Layer    | 0.266           | 0.808              | 0.665                  |

Training Loss: The training loss is consistent across the different layer selection strategies, indicating that the models converge similarly during the training process.

Validation Loss: The validation loss values are identical for all layer selection methods, suggesting that the selected layers have similar generalization capabilities on the validation set.

Validation Accuracy: The validation accuracy is also consistent, reinforcing the idea that the initial layer selection strategy does not significantly affect the model's ability to make accurate predictions on unseen data.

Based on these results, it seems that the choice of initial layer selection does not lead to noticeable differences in the performance metrics of your distilled student model. This finding could be beneficial as it implies that the model is robust to variations in the initial layer selection strategy.

## Discuss any limitations or challenges encountered during the implementation of student distillation, specifically focusing on the analysis of how the initial layer selection affects the overall performance. Propose potential improvements or modifications to address these challenges.

### Limitations and Challenges:

Layer Importance Dynamics: The approach assumes that the importance of layers is fixed, and the selected layers from the teacher model are directly transferred to the student. However, the importance of layers may vary during the training process, leading to suboptimal knowledge transfer.

Model Sensitivity: The model might be sensitive to the choice of layers initially selected, especially if the selected layers do not capture crucial features for the given task. This sensitivity can result in similar performance across different layer selections.

Overfitting to Specific Layers: Depending on the dataset and task, the model might overfit to specific layers selected during distillation. This overfitting could limit the model's ability to generalize to diverse inputs.

Lack of Dynamic Adaptation: The initial layer selection is static throughout the training process, which may not adapt well to changes in the model's learning dynamics. The model might benefit from dynamically adjusting the selected layers during training.

### Potential Improvements:

Dynamic Layer Selection: Implement a mechanism for dynamically adjusting the selected layers during training. This could involve monitoring the performance of different layers and adapting the selection strategy based on the model's learning progress.

Fine-Tuning Selected Layers: Allow for fine-tuning of the selected layers during the training process. This enables the model to refine its representation based on the task-specific information present in the dataset.

Ensemble Learning: Train multiple student models with different layer selections and create an ensemble. This approach helps mitigate the risk of overfitting to specific layers and improves overall model robustness.

Attention Mechanisms: Integrate attention mechanisms into the distillation process. This allows the model to dynamically focus on informative parts of the teacher's representation, enhancing the knowledge transfer process.

Regularization Techniques: Apply regularization techniques to prevent overfitting to specific layers. Techniques such as dropout or layer-wise regularization can promote more diverse learning across layers.

By addressing these potential improvements, you can enhance the flexibility and adaptability of the student distillation process, potentially leading to improved performance across various layer selection strategies. Experimenting with these modifications and monitoring their impact on the model's generalization would provide valuable insights into optimizing the distillation process.

## Instructions to run website:
Navigate to the project folder and on terminal run: 
1. cd app
2. python index.py
3. http://127.0.0.1:5000/ ( to navigate to home page)
4. To view app click on Project list --> a4 from nav or navigate to http://127.0.0.1:5000/a4
![Recording 2024-02-13 105615](https://github.com/Rakshya8/NLP_Assignments/assets/45217500/15d5abcc-2dc5-4769-b137-e2d2ae0aa091)



   
