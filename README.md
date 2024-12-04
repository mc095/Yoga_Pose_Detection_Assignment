# Yoga Pose Detection using Deep Learning

This project aims to build a yoga pose detection model using transfer learning. The model identifies yoga poses from images and provides feedback on alignment and accuracy. The implementation is carried out using TensorFlow and Keras with a MobileNetV2 backbone.

---

## **Dataset**

Dataset used : https://www.kaggle.com/datasets/niharika41298/yoga-poses-dataset

The dataset used for this project contains yoga pose images divided into five classes:
1. **downdog**
2. **goddess**
3. **plank**
4. **tree**
5. **warrior2**

The dataset is organised into two directories:
- `TRAIN`: Contains images used for training the model.
- `TEST`: Contains images used for testing the model.

Each class has its own subdirectory under the `TRAIN` and `TEST` folders.

Before training, the images were validated to ensure no corrupted or invalid files were present. Corrupted files were removed automatically.

---

## **Steps Followed**

### 1. **Dataset Validation**
To ensure the dataset is free from issues:
- Each image was loaded using the Python PIL library.
- Corrupted images were identified and removed to avoid runtime errors.

### 2. **Data Augmentation and Preprocessing**
- Augmentation was applied to the training data to improve generalisation. Techniques used include:
  - Rotation
  - Width and height shifts
  - Zoom
  - Horizontal flips
- Both training and testing images were rescaled to normalise pixel values to a range of [0, 1].

### 3. **Model Architecture**
The model uses **MobileNetV2** as a pre-trained backbone:
- **Pre-Trained Base:** The MobileNetV2 model was loaded with ImageNet weights.
- **Custom Layers:** New dense layers were added:
  - A `Flatten` layer to convert features into a 1D array.
  - A fully connected layer with 128 units and ReLU activation.
  - A dropout layer with 50% rate for regularisation.
  - An output layer with a `softmax` activation function for multi-class classification.

The base model's layers were frozen to retain pre-trained weights during initial training.

### 4. **Training**
- The model was compiled with the **Adam optimizer** using a learning rate of 0.0001.
- Loss function: `categorical_crossentropy`.
- Metrics: `accuracy`.
- The model was trained for 10 epochs using a batch size of 32.

### 5. **Evaluation**
- After training, the model's performance was evaluated using the test dataset.
- Accuracy was reported as a percentage.

---

## **Results**
The trained model achieved an accuracy of **96.38%** on the test dataset.

---

## **How to Run the Code**
1. Ensure you have Python and the required libraries installed:
   - TensorFlow
   - PIL
   - NumPy
2. Place the dataset in a folder named `DATASET` with the structure as described above.
3. Run the script to train the model:
   ```bash
   python yoga_pose_detection.py


### Video Demo

You can view the demo video of the Yoga Pose Detection model here:

<video width="600" controls>
  <source src="demo.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

