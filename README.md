# LS-MinIO-CV-Setup
A project for processing data and building project for label studio for Computer Vision tasks.

- [LS-MinIO-CV-Setup](#ls-minio-cv-setup)
  - [Test Environments](#test-environments)
  - [Commands](#commands)
  - [Data Folder Structure](#data-folder-structure)
    - [Image Classification](#image-classification)
  - [TODO](#todo)
    - [Setups](#setups)
    - [Evaluation](#evaluation)
    - [Export](#export)


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

## TODO
### Setups
- [x] Upload data to MinIO
- [x] Create json files for data
- [x] Create Project via Python SDK
- [x] Adjust Tab View

### Evaluation
- [x] Validate task count with s3 storage (to avoid annotator deleting task)
- [x] Validate tasks all annotated
- [ ] Evaluate annotator performance
- [ ] Evaluate annotation quality
- [ ] Make workflow more generalized

### Export
- [ ] Export annotations
- [ ] Convert annotations to common CV task format