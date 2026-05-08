# Dataset Reference

## Official Source

Dataset page:
- https://www.scidb.cn/en/detail?dataSetId=5654e40ae6d14ade84bac79cb0753852&version=V1

## Archive Packages

The source page provides archive files for the coal classes:
- `01.Non-destructive coal.zip`
- `02.Destructive coal.zip`
- `03.Strongly destructive coal.zip`
- `04.Pulverized coal.zip`
- `05.Fully pulverized coal.zip`

## Local Setup Guidance

1. Download the archive packages from SciDB.
2. Extract them to a local data directory outside this repository.
3. Update each project's dataset YAML/config to point to your local dataset path.
4. Keep raw images and derived splits out of git.

## Notes

- This repository intentionally excludes all dataset images and heavy generated artifacts.
- Keep only metadata files (for example classes list and YAML templates) inside version control.