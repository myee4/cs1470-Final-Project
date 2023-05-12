# README

## https://devpost.com/software/youtube-thumbnail-evaluation

## preprocess.py

`preprocess.py` generates data for training and testing. This file must be run prior to running the model
in order to extract data to be used for training and testing from the `data.csv` file within the `/data` 
directory. Once the script finishes running it will dump the relevant data as a pickle file named `data.p` 
within the same directory. It can be run as follows:

usage: `python preprocess.py [number of data points] [percentage of data for training]`

example: `python preprocess.py 1000 0.7`

If no arguments are provided then the example is run, these numbers were used to generate the data files
we used i.e., the data we used consisted of 1000 YouTube videos with a 70-30 train-test data split. We 
selected these parameters because running the models on datasets larger than 1000 samples was infeasible
on local machines and required the use of dedicated GPUs which we did not have access to.

## main.py

`main.py` contains the train and test methods of our model. This is the file that runs the model and can
be run as follows:

usage: `python main.py [model] [learning rate] [batch size] [epochs]`

example: `python main.py EnhancedModel 0.1 10 25`

If no arguments are provided then the example is run, these numbers were used to run all the models we used 
i.e., all models were run with a learning rate of 0.1, batch size of 10, and 25 epochs. The model has a 
stabilization function that terminates training once the model trains for 5 consecutive epochs without 
improving accuracy by more than 5%.

## /models

This directory contains 3 python files `enhanced_model.py`, `combined_models.py`, and `simple_models.py` 
which contain the various models we used in our quantitative ablation experiments to better interpret the 
results of our model. The architecture used in every model is a subset of the architecture used in the
`EnhancedModel` that can be found in `enhanced_model.py`. This architecture is explained in detail in 
the DevPost, as are further metrics and details on the quantitative ablation experiments performed.

## /utils

This directory contains 2 python files `plot.py` and `scrape.py` that were scripts built with the help
of ChatGPT. These were used for utility purposes only i.e., to plot the distribution of our dataset,
and to help explore how YouTube's API works to extract better quality data.