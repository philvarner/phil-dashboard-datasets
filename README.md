# dashboard-datasets-starter

This repository is an example of a github workflow and dataset configuration file which generates and stores a `<stage>-dataset-metadata.json` file for use in the dashboard-api-starter.

## Usage

1. Create a fork or copy of this repository.
2. Update config.yml with the datasets, stage and bucket of interest.
3. Update github with secrets for accessing AWS.
3. Commit and verify. Note only branches configured in `.github/workflows.yml` will run the workflow (e.g. generate the dataset metadata files).
