# Deep-Learning-for-TinyImageNet30-Image-Classification

## Overview

This project aims to explore and implement deep learning techniques for both image classificationusing a subset of the Tiny ImageNet dataset, referred to as TinyImageNet30;  and image captioning tasks using the COCO dataset. The TinyImageNet dataset consists of 30 different categories, each containing 450 resized images (64x64 pixels) for training, totaling 13,500 images. The project is divided into two parts:

### Image Classification

To develop and evaluate convolutional neural network (CNN) models for image classification. The task involves training models to correctly classify images into one of the 30 predefined categories. Strategies are employed to mitigate issues such as overfitting, including data augmentation and dropout regularization. Model performance will be assessed based on accuracy metrics.

### Image Captioning

The second part focuses on utilizing recurrent neural networks (RNNs) for generating captions for images. Given an image, the model will predict a textual description that describes the content of the image. We will leverage established word vocabularies and sequence-to-sequence architectures to accomplish this task. The generated captions will be evaluated qualitatively, and their coherence and relevance to the corresponding images will be assessed.

## Dataset

The dataset used for this project is a subset of Tiny ImageNet, containing 30 different categories. It is provided as resized images (64x64 pixels), with 450 images per category for training, totaling 13,500 images. The dataset can be accessed [here](https://www.kaggle.com/t/9105198471a3490d9057026d27d8a711).

The COCO dateset can be accessed from [here](https://cocodataset.org/#download)

## Methodology

- **Image Classification:** Develop and evaluate DNN and CNN models, employ data augmentation and dropout regularization to mitigate overfitting, compare model performance with benchmark results.
- **Image Captioning:** Utilize RNNs for generating captions, assess the coherence and relevance of generated captions.
  
## Tools and Libraries

- Python
- Pytorch
- Scikit
- NumPy
- Matplotlib
  
## Conclusion

This project provides valuable insights into the application of deep learning techniques for image classification and captioning tasks. By experimenting with various model architectures and optimization strategies, accurate and reliable models could be developed that can effectively analyze and describe images.
