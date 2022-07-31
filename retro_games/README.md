This directory presents our adapted code for the Atari experiments. 

Original source of the code presented here:

https://github.com/openai/large-scale-curiosity

An example script for the Mario experiment is provided. If 
Bank Heist experiments are performed stats are automatically 
recorded for the figures in the table.

Requirements can be installed via: 

pip install -r requirements.txt

We found the most painless approach to get the original 
code from Burda et al. running was to use the following docker container 
as a base for your installation: 

mcr.microsoft.com/azureml/base-gpu:openmpi3.1.2-cuda10.0-cudnn7-ubuntu18.04
