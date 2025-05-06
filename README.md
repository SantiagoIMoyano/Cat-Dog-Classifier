# Cat-Dog-Classifier

This repository contains all the code necessary to carry out the task of classifying images of dogs and cats using a convolutional neural network combined with dense layers. The workflow covers everything from data preprocessing to making classification predictions on a set of images.
Additionally, the “notebooks” folder includes the notebook that was submitted as the solution for earning the FreeCodeCamp “Machine Learning with Python” certification, to which this project is linked.

## Technologies Used

- Language: Python
- Framework: TensorFlow
- Libraries: NumPy, Matplotlib, pytest, Keras

## Project Execution Notes

Note that to fully run this project using the `run_all.sh` script, you must open a Linux Bash terminal.

Feel free to execute commands to train, evaluate, and predict as many times as you like, since this program is modularized to let you run the three stages independently—so you don’t have to repeat the entire training process each time you want to evaluate or predict on some data.

To run this project successfully, follow these steps:

1. Clone the repository using the command `git clone`.

2. Navigate to the project’s root directory and run `python -m venv venv` to create a virtual environment that isolates the project and ensures it runs correctly.

3. Finally, execute the command `sh run_all.sh`, which will start the program in its entirety.