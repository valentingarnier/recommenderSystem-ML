# Recommender System Project

This is our project for the Machine Learning Course at EPFL - 2019.
In this folder you can find:
* Our report
* Notebooks: Folder containing Data analysis notebook and implementation of exercise 10 for ALS algorithm
* BlendModels: Notebook representing our best model - Its python version is run.py
* als.py: Functions necessary to implement ALS algorithm for the best model
* helpers.py: Functions needed for ALS algorithm. This folder comes from exercise 10 solutions
* implementations.py: Functions required for best model. Very useful for refactoring the code inside notebooks
* run.py: Python file of our best model
* validation_gridsearch: This notebook computes the optimal weights of each model (expanded with feature expansion)
    It does a grid search for each algorithm individually, train them on a train set, and take predictions on a validation set.
    It then run a ridge regression using Scikit to obtain optimal weights that we copy inside Blendmodels
