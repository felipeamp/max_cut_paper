# max_cut_paper

Code used to run tests for the Max Cut paper.

## How to run an experiment

In order to run an experiment, first prepare an `experiment_config.json` file (use `experiment_config_example.json` as an example). Then run

```python3 run_experiments.py /path/to/experiment_config.json```.

The experiment results will appear on the output_folder selected in the experiment_config file.

## How to add a new dataset

In order to add a new dataset, create a folder inside `./datasets/` with the name of the dataset. Inside, create a `config.json` file containing the dataset configurations and a `data.csv` containing the dataset samples (including a header line).

## Pre-requisites:
- Python 3
- Numpy + Scipy + Scikit-learn + Cvxpy
- LAPACK + OpenBLAS (both optional, but highly recommended)



| Names in paper | Names in source code |
|:--------------:|:--------------------:|
| GL Squared Gini | LS Squared Gini|
| GL Chi Square | LS Chi Square |
| \<DatasetName\>-ext | \<DatasetName\> with aggreg |
