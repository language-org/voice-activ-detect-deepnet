# vad

author: steeve LAQUITAINE

date: 28/08/2021

# Setup development environment

* `VSCODE`: coding in an integrated development environment
  * `git-graph`: keep track flow of commit and branches
* `Conda`: isolate environment and manage dependency
* `Git`: code versioning
* `Github`: centralize repo for collaboration
* `Kedro`: standardize codebase

## Create conda environment

Create conda environment, install python and kedro for codebase standardization.
```bash
conda create -n vad python==3.6.13 kedro==0.17.4
```

## Create codebase

```bash
kedro new
```

## Initialize Github central repo for code versioning

* We will use a light version of the `Gitflow Workflow` methodology for code versioning 
and collaboration.

* Create `vad` repo in Github 
* Initialize codebase as local repo with default branch master
  * Master branch will be our `production` branch

```bash
echo "# vad" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M master
git remote add origin https://github.com/slq0/vad.git
git push -u origin master
```

* create and move to develop branch and branch out a feature branch to start developing
  * develop branch would be our `integration` branch (for continuous integration and testing)

* We will keep track of our commit and branch workflow with `git-graph` ([TODO] add snapshot of commit DAG)