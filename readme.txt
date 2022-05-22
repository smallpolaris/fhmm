The folder contains the code for "functional concurrent hidden Markov model"

1. data.py is to generate simulated data.
In the demo, we set rep to 1 to generate one dataset for simulation


2. update.py is to sample parameters using MCMC and MH algorithm.


3. tool.py contains some function to be used in other .py


4. main.py is the main loop to get the estimators.


The user first needs to run data.py to generate data and then run main.py to obtain the estimators.
