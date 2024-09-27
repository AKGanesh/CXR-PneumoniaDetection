![Logo](https://github.com/AKGanesh/CXR-PneumoniaDetection/blob/main/cxr.png)

# Pneumonia Detection using CXR - Deep Learning

This project utilizes deep learning techniques to detect pneumonia from chest X-ray images. The model is built using transfer learning with pre-trained architectures like CNN,VGG16, ResNet50 - fine-tuned for binary classification (normal vs. pneumonia). Once trained, the models can make predictions on new chest X-ray images to classify them as either normal or indicating pneumonia. 

## Implementation Details

- Dataset: 
  - The dataset used in this project is sourced from Kaggle’s Chest X-Ray Images (Pneumonia) dataset (https://github.com/ieee8023/covid-chestxray-dataset). 
  - It consists of two main classes:
    - Normal: Chest X-ray images of patients without pneumonia.
    -  Pneumonia: Chest X-ray images of patients diagnosed with pneumonia.
  - The dataset is split into training, validation, and test sets, with appropriate preprocessing applied using Keras’ ImageDataGenerator.
- Model: Custom CNN, ResNet50, VGG16
- Input: Chest X-Ray Image
- Output: Class (Healthy / Pneumonia)

## Process
- Data Composition and study
- Data Pre-processing (Resize, Color space, Augmentation and Normalization)
- Making use of ImageDataGenerator for the above
- Model Development, Compilation and function
- Check with ResNet50 Transfer Learning + Finetuning
- Check with VGG16 Transfer Learning + Finetuning
- Test

## Evaluation and Results
  | Type | Test Accuracy | Test Loss |Score |
  |------|-----------|---------------|------|
  |CNN From Scratch|86%|0.34|0.714|
  |ResNet50 (TF+FT)|82%|0.39|0.820|
  |VGG16 (TF+FT)|92%|0.22|0.696|

Model performance is evaluated on a separate test set to assess its ability to generalize to unseen data. Evaluation metrics include accuracy, F1 score, precision, and recall on both training and validation sets across epochs.

Philosophy:
- CNN from Scratch gave a result of score 0.70, We can experiment with no of neurons and layers to improve this up to some extent
- With Pre-existing model ResNet50, No of unfreeze layers 10 and 5(better result) gave different results. Experimenting with Optimizer Adam and Ndam(better result-0820) gave different results
- With Pre-existing model VGG16, No of unfreeze layers 5, Nadam produced a score of 0.696, despite it's great accuracy of 93%
- Dropout and BatchNormalization has an impact on the result. Experiment with various Drop out range 0.2-0.5
- No of Layers and No of perceptrons has an impact on the result

Once trained, the models can make predictions on new chest X-ray images to classify them as either normal or indicating pneumonia. The predictions are based on the highest probability output from the softmax layer of the model.

## Conclusion
This project demonstrates the application of deep learning techniques for medical image classification, specifically for detecting pneumonia from chest X-ray images. The use of transfer learning with pre-trained models significantly improves the model’s performance and generalization capabilities

## Libraries

**Language:** Python

**Packages:** Pandas, Numpy, Matplotlib, Tensorflow, Keras

## Roadmap

- To experiment with ensemble methods
- Processing images helps like histrogram equavilization, canny edge detection and supply along with context, needs to be experimented

## FAQ

#### What is Transfer Learning?
Transfer learning is a machine learning technique where a model developed for a particular task is reused as the starting point for a model on a second task. This approach is particularly useful when the second task has limited data available.

#### What is Fine Tuning?
Fine-tuning is a process in machine learning where a pre-trained model is further trained on a new, often smaller and more specific dataset to adapt it to a particular task. This technique leverages the knowledge the model has already acquired from a large, diverse dataset and refines it to perform well on the new task.

#### What is ImageDataGenerator?
The ImageDataGenerator class in Keras is a powerful tool for real-time data augmentation and preprocessing of image data. It allows you to generate batches of tensor image data with real-time data augmentation, which can help improve the robustness and generalization of your deep learning models. 
The flow_from_directory method in Keras’ ImageDataGenerator class is used to load images directly from a directory on disk and generate batches of augmented/normalized data. This is particularly useful when working with large datasets that cannot fit into memory.

#### What are ResNet50 and VGG16?
ResNet50 and VGG16 are two popular convolutional neural network (CNN) architectures used for image classification and other computer vision tasks. While more efficient than VGG16 in terms of depth and performance, ResNet50 can still be computationally intensive. Both are commonly used for Image classification, Object detection and Medical image analysis.

#### How to handle Imbalance Datasets?
From Model side, adjustments like class_weight helps.
Using compute_class_weight/sklearn can help improve the performance of your model on imbalanced datasets by ensuring that the model pays more attention to the minority class.
From data side, Augmentaion and resampling techniques like SMOTE helps.

## Acknowledgements
- https://www.kaggle.com/abbhinavvenkat
- https://www.tensorflow.org/
- https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
- https://www.tensorflow.org/api_docs/python/tf/keras/applications/VGG16

## Contact

For any queries, please send an email (id on github profile)

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
