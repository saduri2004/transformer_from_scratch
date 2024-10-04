# Transformer Model Interface
![Screenshot 2024-10-04 at 3 03 18 PM](https://github.com/user-attachments/assets/43f9f93a-b34c-46e6-a4f3-c2f241e33d68)

This repository contains the code for a web-based Transformer model interface, designed to visualize attention maps and test predictions using Flask and PyTorch.

## Features

- **Interactive Web Interface**: A Flask-based web interface for interacting with the Transformer model.
- **Attention Map Visualization**: Visualize how the model focuses on different parts of the input.
- **Model Prediction**: Test the model's predictions based on user input.

## Installation

To set up the project environment, follow these steps:

- Clone the repository:
  ```bash
  git clone https://github.com/yourusername/your-repository.git
  cd your-repository


- Install the required packages:
  ```bash
  pip install torch torchvision torchaudio matplotlib flask datasets

- Run the application with python3 main.py


The application will be available at http://127.0.0.1:5000/ on your local machine.

Usage

To use the application, navigate to http://127.0.0.1:5000/ in your web browser. You can start training the model, visualize the attention map, and enter text to see the model’s predictions.
