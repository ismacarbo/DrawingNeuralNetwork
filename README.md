
# Object Recognition Drawing Program

## Overview

This project is a simple drawing application that allows users to draw objects and get them recognized using a pre-trained Convolutional Neural Network (CNN) model. The project is divided into two main parts: the drawing application using Pygame and the model training using TensorFlow/Keras.

## Features

1. **Drawing Application:**
   - Allows users to draw objects on a 500x500 pixel canvas.
   - Recognizes the drawn object and displays the name of the object along with the confidence level.
   - Uses Pygame for creating the drawing interface.

2. **Model Training:**
   - Loads Quick, Draw! dataset for training.
   - Preprocesses the data and trains a CNN model to recognize different categories of objects.
   - Uses data augmentation to improve model generalization.
   - Saves the best model based on validation loss.

## Dependencies

- Python 3.6 or later
- Pygame
- NumPy
- TensorFlow/Keras

## How to Run

1. **Install dependencies:**
   Ensure you have all the necessary libraries installed. You can install them using pip:

   ```bash
   pip install pygame numpy tensorflow
   ```

2. **Train the Model:**
   - Place the Quick, Draw! dataset files in the specified directory.
   - Run the model training script to train and save the model.

3. **Run the Drawing Application:**
   - Execute the Pygame script to start the drawing application.
   - Draw an object on the canvas and release the mouse button to see the prediction.

## Model Training Details

- The model is trained on the Quick, Draw! dataset which consists of various categories of doodles.
- The dataset is preprocessed by normalizing the pixel values and reshaping the images.
- A CNN model is used for training with layers including Conv2D, MaxPooling2D, Flatten, Dense, and Dropout.
- Data augmentation techniques such as rotation, zoom, and shift are applied during training.
- The model is compiled using the Adam optimizer and categorical cross-entropy loss function.
- Model training includes callbacks for saving the best model and reducing the learning rate when the model plateaus.

## Drawing Application Details

- The drawing application is built using Pygame.
- Users can draw on the canvas using the mouse. The drawing is converted to a grayscale image and resized to 28x28 pixels.
- The pre-trained model is used to predict the category of the drawn object.
- The recognized object name and confidence level are displayed on the screen.

## Conclusion

This project provides a basic framework for creating a drawing application with object recognition capabilities. The model can be trained on different categories of objects, and the application can be extended with additional features and improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.md) file for details.