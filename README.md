# LS-MinIO-CV-Setup
A free deployment method for Label Studio to handle annotation task for Image Classification.

## Test Environments
- Label Studio: 1.8.0

## Commands
- setup
  - generate .env & docker-compose.*.yml
  - print command
- upload
  - root path
- create
  - project name
  - ...

## Data Folder Structure
### Image Classification

<details>

<summary>TODOs</summary>

### Setups
- [x] Upload data to MinIO
- [x] Create json files for data
- [x] Create Project via Python SDK
- [x] Adjust Tab View

### Evaluation
- [x] Validate task count with s3 storage (to avoid annotator deleting task)
- [x] Validate tasks all annotated
- [x] Evaluate annotator performance
- [x] Evaluate annotation quality
- [ ] Make workflow more generalized

### Export
- [x] Export annotations
- [ ] Convert annotations to common CV task format

</details>
