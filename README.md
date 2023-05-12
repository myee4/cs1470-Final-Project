# README

## https://devpost.com/software/youtube-thumbnail-evaluation

## preprocess.py

`preprocess.py` generates data for training and testing. This file must be run prior to running the model
in order to extract the relevant data from the `data.csv` file. Once the script finishes running it will 
dump the data as a pickle file `data.p` within the `/data` directory. It can be run as follows:

usage: `python preprocess.py [number of data points] [percentage of data for training]`

example: `python preprocess.py 1000 0.7`

If no arguments are provided then the example is run, these numbers were used to generate the data files
used in all of our testing i.e., the data we used consisted of 1000 YouTube videos with a 70-30 train-test
split of data. We selected these parameters because running the models on datasets larger than a 1000 samples
was infeasible on local machines and required the use of dedicated GPUs.

## main.py

`main.py` contains the train and test methods of our model. This is the file that runs the model and can
be run as follows:

usage: `python main.py [model] [learning rate] [batch size] [epochs]`

example: `python main.py EnhancedModel 0.1 10 25`

If no arguments are provided then the example is run, these numbers were used to run all the models used
in our testing i.e., all models were run with a learning rate of 0.1, a batch size of 10, and 25 epochs.
The model has a stabilization function that terminates training once the model trains for 5 consecutive 
epochs with an accuracy within 5% in each of those epochs.

## /models

This directory contains 3 python files `enhanced_model.py`, `combined_models.py`, and `simple_models.py` 
which contain the various models we used in our quantitative ablation experiments to better interpret the 
results of our model. The architecture used in every model is a subset of the architecture used in the
`EnhancedModel` that can be found in `enhanced_model.py`. This architecture is explained in detail in 
the DevPost.

## /utils

This directory contains 2 python files `plot.py` and `scrape.py` that were scripts built with the help
of ChatGPT. These were used for utility purposes only i.e., to plot the distribution of our dataset,
and to help explore how YouTube's API works to extract better quality data.