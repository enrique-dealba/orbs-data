# orbs-data

## Running the Pipeline

This uses DVC to manage the data processing and analysis pipeline. To run the entire pipeline:

```bash
dvc repro
```

This command will run all stages of the pipeline that have changed or have dependencies that have changed.

To run a specific stage:

```bash
dvc repro <stage_name>
```

Currently, this project uses the following stages:
* preprocess
* train
* analyze

## Viewing Results

After running the analysis, you can find the results in the `analysis_results/` directory:

* `data_info.json`: Contains basic statistics about the preprocessed data
* `random_frames.png`: Visualizes random frames from each video
* `intensity_distribution.png`: Shows the distribution of pixel intensities for each video

## Updating the Project

When pulling updates from the remote repository:

1. Pull the latest code changes:

```bash
git pull
```

2. Update the data and pipeline:
```bash
dvc pull
dvc repro
```

## Adding New Data

To add new AVI files to the project:

1. Place the new AVI file(s) to the `data/raw_avis/` directory.

2. Add the file(s) to DVC:

```bash
dvc add data/raw_avis/new_file.avi
```

3. Update the `preprocess.py` script to include the new file(s).

4. Update the `dvc.yaml` file to include the new file(s) as a dependency for the preprocess stage

```yaml
deps:
  - preprocess.py
  - data/raw_avis/control1_stack.avi
  - data/raw_avis/control2_stack.avi
  - <new avi files here...>
```

5. Commit the changes:

```bash
git add .
git commit -m "Add new AVI file"
```

6. Finally, push the changes (git first, then dvc):

```bash
git push
dvc push
```
