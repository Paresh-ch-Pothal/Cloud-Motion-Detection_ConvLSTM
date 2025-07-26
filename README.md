# Cloud Motion Prediction using ConvLSTM

This project aims to predict the next cloud frame based on a sequence of previous cloud frames using a **ConvLSTM** model. The model is trained on sequences of 10 cloud frames and predicts the next frame (cloud image) based on the previous sequence.

## Model Overview

### **What the Model Does**
The model is designed to predict the next frame in a sequence of cloud images. Given 10 previous frames, the model forecasts the next cloud frame, allowing us to simulate cloud movement over time.

### **Model Architecture**

The model architecture is based on **ConvLSTM2D** (Convolutional LSTM), a type of Recurrent Neural Network (RNN) that combines convolutional operations with LSTM (Long Short-Term Memory) units to handle spatiotemporal dependencies in sequential data, particularly image sequences.

### **Layers Used**

1. **ConvLSTM2D Layer (1st Layer)**:
   - **Purpose**: This layer is designed to capture both spatial and temporal features from the input image sequence.
   - **Details**: The layer takes a sequence of images (10 frames) as input, performs 2D convolutions across the spatial dimensions, and retains temporal dependencies across frames.
   - **Parameters**:
     - Filters: `32`
     - Kernel Size: `(3, 3)`
     - Activation: `relu`
     - Return Sequences: `True` (to retain the full output for the next ConvLSTM layer)

2. **BatchNormalization Layer**:
   - **Purpose**: This layer normalizes the activations from the previous layer, which helps speed up training and stabilize the learning process.
   - **Details**: By normalizing the activations, this layer helps in preventing overfitting and ensures the model trains efficiently.

3. **ConvLSTM2D Layer (2nd Layer)**:
   - **Purpose**: This layer further captures temporal dependencies by using ConvLSTM, but it returns only the last frame of the sequence.
   - **Details**: This layer receives the output from the previous layer and processes it.
   - **Parameters**:
     - Filters: `32`
     - Kernel Size: `(3, 3)`
     - Activation: `relu`
     - Return Sequences: `False` (only the final output of the sequence is passed forward)

4. **Conv2D Layer**:
   - **Purpose**: This convolutional layer extracts spatial features from the output of the previous layer.
   - **Details**: It works by applying convolutional operations to the feature map obtained from the previous ConvLSTM layer.
   - **Parameters**:
     - Filters: `64`
     - Kernel Size: `(3, 3)`
     - Activation: `relu`

5. **Conv2D Layer (Final Output)**:
   - **Purpose**: This is the final convolutional layer that outputs the predicted frame.
   - **Details**: The output has 11 channels, one for each cloud type, which is used to classify each pixel in the predicted frame.
   - **Parameters**:
     - Filters: `11` (one for each cloud type)
     - Kernel Size: `(1, 1)`
     - Activation: `softmax` (to output probabilities for each class at each pixel)

### **How the Model Works**

1. **Input**: The model takes 10 consecutive frames (each of size 128x128 pixels) as input. Each frame is represented by a 2D array of pixel values. The input is normalized to have pixel values between `0` and `1`.

2. **ConvLSTM Layers**: 
   - The **first ConvLSTM2D layer** processes the sequence of frames, capturing both spatial features (image features) and temporal features (how the images change over time).
   - The **second ConvLSTM2D layer** further processes the output of the first ConvLSTM, focusing on capturing deeper temporal patterns.

3. **Convolutional Layers**: 
   - The model then uses **Conv2D layers** to further process the spatial features and refine the output.

4. **Final Prediction**: The **final Conv2D layer** outputs a prediction for each pixel in the frame, with 11 different class probabilities (cloud types) per pixel.

5. **Output**: The output is a **128x128 predicted cloud frame**, where each pixel is assigned a cloud type (out of 11 possible types).

### **Model Training**

- **Loss Function**: The model uses **categorical cross-entropy** loss since this is a multi-class pixel-wise classification task.
- **Optimizer**: The Adam optimizer is used to train the model, as it has been shown to perform well in many types of image-related tasks.

---

## Results

The model is able to predict the next cloud frame from a sequence of previous frames. Given the nature of cloud motion, the predictions show the model's capability to understand temporal changes in cloud patterns.

---

## Conclusion

This model is an effective approach for predicting the movement of clouds based on satellite image sequences, leveraging the power of **ConvLSTM** to handle both spatial and temporal dependencies in the cloud images.

---

### **Next Steps**
- **Fine-Tuning**: You can improve the model’s accuracy by fine-tuning it with more data or experimenting with different network architectures.
- **Evaluation**: Further evaluation metrics such as pixel-wise accuracy or Intersection over Union (IoU) can be used to assess performance.

