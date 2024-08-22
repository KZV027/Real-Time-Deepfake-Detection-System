
# Real-Time Deepfake Detection

This repository contains a deepfake detection system that uses a MobileNet model to identify whether an image or video frame is a deepfake or a genuine image. The system provides real-time results for practical applications. The model achieves an accuracy of 79% based on the ROC curve evaluation.



## Table of Contents
1. Features
2. Model Description
3. ROC Curve
4. Usage
5. License

## Features
* Real-time detection of deepfake images.
* MobileNet model trained with the Adam optimizer.
* Performance evaluation using ROC curve.

## Model Description
The primary model used for deepfake detection is based on the MobileNet architecture, which is lightweight and efficient for real-time applications. The model is trained to classify images as either "Real" or "Deepfake."

* Model Architecture: MobileNet
* Optimizer: Adam
* File: best_deepfake_model.h5


## ROC Curve
The ROC (Receiver Operating Characteristic) curve provides a graphical representation of the model performance. The area under the ROC curve (AUC) for this model is approximately 79%, indicating a good balance between sensitivity and specificity.
![ROC](https://github.com/user-attachments/assets/68dc6e56-e60d-4b3d-bbae-7f77dd89a6e2)


## Usage
To use the deepfake detection model, you can load the model and run predictions as follows:
```bash
 import tensorflow as tf

 # Load the model
 model = tf.keras.models.load_model('best_deepfake_model.h5')
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.
