# vad

author: steeve LAQUITAINE

date: 28/08/2021

[TODO]: add table of content  
[TODO]: add file structure  

## Train, inference & eval in 3 steps

Prerequisites installations :

You have:
  - `conda 4.8.3` (`which conda` in a terminal).
  - you have `Git 2.32.0`    

You can get and run the codebase in 3 steps:

1. Setup:

```bash
git clone https://github.com/slq0/vad_deepnet.git
cd vad_deepnet
conda create -n vad python==3.6.13 
pip install kedro==0.17.4  
bash setup.sh
```

2. Move the dataset to `vad_deepnet/data/01_raw/` 
  
3. Run:

```bash
kedro run --pipeline train --env train
kedro run --pipeline predict_and_eval --env predict_and_eval
```

## Overview of tool stack used

* Development:
  * `VSCODE`: coding in an integrated development environment
  * `Conda`: isolate environment and manage dependencies
  * `Git`: code versioning
  * `Github`: centralize repo for collaboration
  * `Kedro`: standardize codebase

* Experiment tracking & reproducibility:  
  * `mlflow`: pipeline parameters & model experiment tracking
  * `Tensorboard`: model experiment inspection & optimization
  * `git-graph`: keep track of flow of commit and branches
  
* Readability:  
  * `black`: codebase formatting
  * `pylint`: codebase linting

* Test coverage:  
  * `pytest`: minimal unit-tests

## Setup

### Create a virtual environment

Create conda environment, install python and kedro for codebase standardization.

```bash
conda create -n vad python==3.6.13 kedro==0.17.4
```

### Clone the code repository from Github

* I used a light version of the `Gitflow Workflow` methodology for code versioning 
and collaboration.
* Master branch will be our `production` branch
* I created and move to develop branch and branch out a feature branch to start developing
  * develop branch would be our `integration` branch (for continuous integration and testing)
* I kept track of my commit and branch workflow with `git-graph` ([TODO] add snapshot of commit DAG)

```bash
git clone https://github.com/slq0/vad_deepnet.git
```

### Build & install the dependencies

Run this bash script to build and install the project's dependencies:  

```bash
bash setup.sh
```

## Run pipelines

Train the basic model:

Run the training pipeline:

```bash
kedro run --pipeline train --env train
```

Run inference with the model:

```bash
kedro run --pipeline predict --env predict
```

Evaluate its performance metrics:

```bash
kedro run --pipeline predict_and_eval --env predict_and_eval
```


# Model inspection, tracking & optimization

## Tensorboad

The model runs are logged in `tbruns/`.

```bash
tensorboard --logdir tbruns
# http://localhost:6006/
```

## Mlflow

I used mlflow to track experiments and tested hyperparameter runs. 
The logs are stored in `mlruns/`.

```bash
kedro mlflow ui --env train --host 127.0.0.1 --port 6007
# http://localhost:6007/
```

## Kedro-viz

To keep track of the pipeline:

```bash
kedro viz
# http://127.0.0.1:4141
```

# Testing

Run unit-tests on the code base:

```bash
pytest src/tests/test_run.py
```

# Perspectives 

## Data augmentation

We can use the pure speech and noise corpora below for speech vs. silence classes. We can also augment pure 
speech dataset by adding noisy speech data created by summing speech and noise data.

* `TIMIT` corpus for clean speech (1)
  * license: ([TODO]: check)
* `QUT- NOISE `: corpus of noise (1)
    * license: CC-BY-SA ([TODO]: check)

# Report 

Format the report:  

- The report has collapsible table of content (see in preview for mac and adobe reader on windows)

```bash
jupyter nbconvert notebooks/report.ipynb --to pdf --no-input
```

# References

*note: references are formatted according to the Amercian Psychological Association (APA) style*

(1) Dean, D., Sridharan, S., Vogt, R., & Mason, M. (2010). The QUT-NOISE-TIMIT corpus for evaluation of voice activity detection algorithms. 
In Proceedings of the 11th Annual Conference of the International Speech Communication Association (pp. 3110-3113). International Speech 
Communication Association.



