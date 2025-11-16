# Spotify-pyspark-analysis

Multidiscipline project (Big Data processing + Unsupervised machine learning) using PySpark to process a Spotify Charts dataset with over 26M rows - downloaded from Kaggle - and produce analytics applying xxx ML technique.

Project goal: demonstrate an end-to-end pipeline — ingest large chart data, clean & transform with PySpark, extract features, train and evaluate ML model (define one).

Dataset: Spotify Charts by Dhruvildave (Kaggle)
Source: https://www.kaggle.com/datasets/dhruvildave/spotify-charts/code

## Table of Contents

- **Project summary**: brief overview and use cases
- **Repo structure**: important files and folders
- **Requirements**: what to install and where
- **Quickstart (WSL / Linux)**: copy/paste commands to get started
- **How to run**: notebooks and CLI examples
- **Support files**: `requirements.txt`

## Project summary

This repository shows how to process very large CSVs with PySpark (in a virtual environment) and apply ML techniques using Spark ML. The pipeline is designed to be reproducible and shareable.

## Repo structure

```
spotify-pyspark-analysis/
├── data/                          # place `charts.csv` here (do NOT commit the full CSV)
├── labs_venv/                     # local venv (do NOT commit)
├── notebooks/
│   └── analysis.ipynb
├── docs/                          # project documentation
├── requirements.txt
└── README.md
```

## Requirements

- WSL2 (Ubuntu recommended) or Linux
- Java 17 installed (OpenJDK 17 is good)
- Python 3.10+ (3.12 is fine)
- Virtual environment recommended

Example `requirements.txt` is provided in the repo. It includes PySpark and common data/ML libraries.

## Quickstart (WSL / Linux)

Open WSL/terminal and run the following (copy/paste):

```bash
# 1) update system and install java 17 (if not already)
sudo apt update
sudo apt install -y openjdk-17-jdk

# verify
java -version

# 2) create project folder (if not already) and a python venv
cd /home/<you>/your_path
python3 -m venv your_virtual_env
source your_virtual_env/bin/activate

# 3) upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) place your charts.csv into ./data/ (download from Kaggle)

# 5) open the notebook in Jupyter or VSCode and run cells

```


## How to run

- Jupyter / Notebook (recommended for exploration):
  - Activate venv: `source labs_venv/bin/activate`
  - Start notebook: `jupyter notebook` (or use VSCode Remote - WSL)
  - Open `notebooks/analysis.ipynb` and run cells

- Run ETL script

- Run ML script

## Examples & tips

- Create SparkSession (local):

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder
spark = spark.appName("SpotifyAnalysis")
spark = spark.master("local[4]") # 4 CPUs
spark = spark.getOrCreate()
```

- Read CSV sample (convert `Path` to str):

```python
from pathlib import Path
file_path = str(Path("data") / "charts.csv")

df = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load(file_path)
df.printSchema()
```

- Use Spark UI (`http://localhost:4040`) to inspect jobs


## Data dictionary (suggested)

-

Derived columns:

## Support files

- `requirements.txt` — Python dependency list


## License & credits


---
