<p align="center"><a href="https://laravel.com" target="_blank"><img src="https://www.python.org/static/img/python-logo.png" width="400" alt="Laravel Logo"></a></p>

# About This Project
This is [Python](https://en.wikipedia.org/wiki/Python_(programming_language)) based [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning#:~:text=Machine%20learning%20(ML)%20is%20a,perform%20tasks%20without%20explicit%20instructions.) Model using [K-Nearest Neighbors Algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) for predicting college student's graduation based on their attendance, assignments and exams scores, and their GPA.
I made this for a subject for my 6th Semester scientific papers.

## Python

Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation. Python is dynamically type-checked and garbage-collected. It supports multiple programming paradigms, including:
- structured (particularly procedural)
- object-oriented and functional programming.
It is often described as a "batteries included" language due to its comprehensive standard library.

## K-Nearest Neighbors
K-Nearest Neighbors (KNN) is a non-parametric, supervised machine learning algorithm used for both classification and regression tasks. In Python, the scikit-learn library provides a robust and easy-to-use implementation of KNN. 

## How KNN Works:

- Define k: Choose the number of nearest neighbors (k) to consider.
- Calculate Distances: For a new, unclassified data point, calculate its distance (e.g., Euclidean distance) to all data points in the training set.
- Find Nearest Neighbors: Identify the k data points from the training set that are closest to the new data point.
- Prediction: Classification, assign the new data point to the class that is most frequent among its k nearest neighbors (majority voting). Regression, Predict the value of the new data point as the average (or weighted average) of the values of its k nearest neighbors.

## How To Use:

- Clone the Repository using  `git clone`
- Go to the project's directory
- train the model by running the `./train_model.py`
- After that, run the Web Browser view by running the `app.py`
