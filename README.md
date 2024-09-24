# Neural Network Doodle Recognition

This project is an interactive application that allows users to draw digits on a grid and uses a TensorFlow neural network to predict the drawn number in real-time.

## Features

- Interactive drawing grid for users to input digits
- Real-time prediction of drawn digits using a TensorFlow neural network
- User-friendly interface for easy interaction
- Responsive design for various screen sizes

## Technologies Used

- TensorFlow.js for the neural network model
- HTML5 Canvas for the drawing interface
- JavaScript for frontend logic and interactivity
- CSS for styling and responsiveness

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/lucasarano/Neural-Network-Doodle-Recognition.git
   cd Neural-Network-Doodle-Recognition
   ```

2. Open `index.html` in a modern web browser.

## Usage

1. Open the application in your web browser.
2. Use your mouse or touchscreen to draw a digit (0-9) on the provided grid.
3. The neural network will attempt to predict the drawn digit in real-time.
4. Clear the grid using the "Clear" button to draw a new digit.

## How It Works

1. **User Input**: The user draws a digit on the HTML5 Canvas grid.
2. **Data Preprocessing**: The drawn image is converted into a format suitable for the neural network.
3. **Prediction**: The TensorFlow.js model processes the input and predicts the digit.
4. **Display**: The predicted digit is displayed to the user.

## Model Training

The neural network model was trained on the MNIST dataset of handwritten digits. The training process involved:

1. Data preparation and normalization
2. Model architecture design
3. Training the model using TensorFlow
4. Converting the model to TensorFlow.js format for web use

## Contributing

We welcome contributions to improve the digit recognition model or enhance the user interface. Please feel free to submit pull requests or open issues on our GitHub repository: https://github.com/lucasarano/Neural-Network-Doodle-Recognition

## Future Improvements

- Implement a feature to allow users to correct misclassifications to improve the model
- Add support for recognizing multiple digits or simple mathematical operations
- Optimize the model for better performance on mobile devices

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The TensorFlow team for their excellent machine learning framework
- The MNIST dataset creators for providing a robust dataset for training
- All contributors and users of this doodle recognition app
