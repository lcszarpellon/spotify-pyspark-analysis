# Documentation — spotify-pyspark-analysis

This `docs/docs.md` offers detailed guidance for the project: dataset notes, ETL recommendations and ML pipeline sketches.

## Dataset

- Source: Kaggle — *Spotify Charts* by Dhruvildave.
- Columns: 

## ETL guidance (PySpark)

1. Read CSV using `spark.read.csv` with `header=True` and `inferSchema=True`.
2. 

Example: 

```python
code example
```


## ML pipeline sketch (Spark ML)

Goal example: clusterization analysis (example)

Steps:
- step 1
- step 2

Saving model example:

```python
model.write().overwrite().save('models/spotify_rf_model')
```

## Performance & engineering tips

- Don't do that `example.code()`
- Do that...


## Other tips
