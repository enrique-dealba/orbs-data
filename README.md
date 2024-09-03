# orbs-data

## Setting up with DagsHub

This project uses DagsHub for data version control. To set up:

1. Ensure you have a DagsHub account and access to the repository.

2. Clone the repository:
```bash
git clone https://dagshub.com/enrique-dealba/orbs-data.git
cd orbs-data
```

3. Set up DVC with DagsHub:

```bash
dvc remote add -d origin https://dagshub.com/enrique-dealba/orbs-data.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <your-dagshub-username>
dvc remote modify origin --local password <your-dagshub-token>
```

4. Pull the data:
```bash
dvc pull
```


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

DagsHub Integration

This project is integrated with DagsHub for version control of both code and data. The DagsHub repository can be found at: `https://dagshub.com/enrique-dealba/orbs-data`

To collaborate on this project:

Ensure you have access to the DagsHub repository.
Follow the setup instructions at the beginning of this README.
Make your changes locally.
Push your code changes to git and your data changes to DVC as described in the "Adding New Data" section.

For detailed instructions on integrating DVC with DagsHub, see the [DagsHub DVC Integration Guide](https://dagshub.com/docs/integration_guide/dvc/).
