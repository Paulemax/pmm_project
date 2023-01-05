# Practical multimodal-multisensor data analysis pipelines Project

The data for this project is weather data from the irish government.  
The data is available at : [link](https://data.gov.ie/dataset/weather-buoy-network?package_type=dataset)  

The data consists of data from 6 different weather buoys that were deployed from 2001 to 2006 around ireland.  

## Installation

1. Install requirements specified in ```requirements.txt```  
2. run ``` python3 main.py ``` to download the data and train the models
3. run ``` python3 models.py ``` to train the models by hand

## Data Questions

The data questions for my dataset for the second exercise are the following:
    - How does the data change with time, e.g. is there are trend in temperature?
    - Are Variables with the same units correlated focus on all | mutual information / dependency
    - Can we predict the temperature forecast based on the other parameters in the dataset
        - How certain is the model in the predictions?

The Answers to these questions can be found in the ```visualize.ipynb``` jupyter Notebook

## Model Visualizations

The visualization goals for this exercise is to plot the model prediction over time.
(Since the assignment does not clarify which visualizations to use / not all of them apply to my model, i just do what i think is right)

- Mean Squared Error and R^2 Score as Performance Metrics
  - R^2 Score describes how good the classifier is scores near to 1 are better. 0 is the baseline
- temperature / time plot
- temperature / time plot with uncertainties

The visualizations for the third exercise can be found in the ```model_visualization.ipynb``` jupyter Notebook

## User Interactivity Plans

My tentative goals for user interactivity are:

- Let the user specify the quantiles.
- Let the user specify the prediction parameters of the visualization task.
- Let the user specify the prediction target, eg. temperature / wave height, etc...
