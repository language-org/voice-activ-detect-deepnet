# vad

author: steeve LAQUITAINE

date: 28/08/2021

**Short description**: Voice activity detection is critical to reduce the 
computational cost of continuously monitoring large volume of speech data 
necessary to swiftly detect command utterances such as wakewords. My objective 
was to code a Voice Activity Detector (VAD) with reasonable performances 
(Low false rejection rate) based on a neural network within a week and with 
low computing resources. I trained and tested the model on labelled audio data 
containing speech from diverse speakers including male, female, synthetic, low 
and low volume, slow and fast space speech properties. The dataset came from 
LibriSpeech and was prepared and provided by SONOS. I used a variety of tools 
to extract, preprocess and develop and test the model but I mostly relied on 
Tensorflow advanced Subclassing api, tensorflow ops and Keras, 
Tensorboard, Seaborn and the more classical matplotlib visualization tools to make 
sense of the data, clean the data and inspect the inner workings of the model. 

## The full report 

notebooks/report.pdf  

notebooks/report.ipynb

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
  
3. Run basic model training (takes 30min) and predict-eval (20 secs):

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

## Setup break down

### Create a virtual environment

Create conda environment, install python and kedro for codebase standardization.

```bash
conda create -n vad python==3.6.13 kedro==0.17.4
```

### Clone the code repository from Github

* I used a light version of the `Gitflow Workflow` methodology for code versioning 
and collaboration.
* A `Master` branch will be our `production` branch (final deployment):
* I created and moved to a `Develop` branch and branched out a `feature` branch to start developing
  * The `Develop` branch would hypothetically be an `integration` branch (for continuous integration and testing)
* I kept track of my commits and the workflow of branches with `git-graph`

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

## Tensorboard

Visualize layers' weights, biases across epochs, training and validation loss,
performance metrics on validation, the model's conceptual and structural graph
to dive into decreasing levels of abstraction.

The model runs are logged in `tbruns/`.

```bash
tensorboard --logdir tbruns
# http://localhost:6006/
```

## Mlflow

I used mlflow to track experiments and tested hyperparameter runs 
(e.g., run duration). 

The logs are stored in `mlruns/`.

```bash
kedro mlflow ui --env train --host 127.0.0.1 --port 6007
# http://localhost:6007/
```

## Kedro-viz

To keep track of the pipeline and optimize it, I used Kedro-viz which
described the pipelines with Directed Acyclic graphs:

```bash
kedro viz
# http://127.0.0.1:4141
```

# Testing

Run unit-tests on the code base. I initialized unit-tests but did not
have to implement more than one test. You can run unit-tests with:

```bash
pytest src/tests/test_run.py
```


# "vad" package's documentation (Sphynx)

You can open the package's Sphynx documentation by opening `docs/build/html/index.html`
in your web browser (double click on the file):  

```bash
kedro build-docs --open
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

The final report is `notebook/report.pdf` with a collapsible table of content 
(see in preview for mac and adobe reader on windows)

To format the .ipynb report into a .pdf run in the terminal :  

```bash
jupyter nbconvert notebooks/report.ipynb --to pdf --no-input
```

# References

*note: references are formatted according to the Amercian Psychological Association (APA) style*

(1) Dean, D., Sridharan, S., Vogt, R., & Mason, M. (2010). The QUT-NOISE-TIMIT corpus for evaluation of voice activity detection algorithms. 
In Proceedings of the 11th Annual Conference of the International Speech Communication Association (pp. 3110-3113). International Speech 
Communication Association.



