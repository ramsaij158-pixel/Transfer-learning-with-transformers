# GitHub setup

This folder is the repository root. Do not upload the parent folder or the ZIP from the original download.

## Push from a terminal

```bash
cd Made-With-ML-GitHub
git init
git add .
git commit -m "Build production-ready ML classification pipeline"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Before enabling cloud workflows

The code and quality workflow can be pushed immediately. Cloud training and serving require your own infrastructure values.

1. Replace `CHANGE_ME` in `deploy/jobs/workloads.yaml` and `deploy/services/serve_model.yaml`.
2. In GitHub repository settings, add Actions secrets named:
   - `AWS_ROLE_ARN`
   - `ANYSCALE_HOST`
   - `ANYSCALE_CLI_TOKEN`
3. Configure the AWS role for GitHub OIDC and restrict it to this repository.
4. Never commit `.env`, cloud credentials, tokens, model artifacts, `efs/`, or local logs.

## Recommended repository description

Production-style NLP classification system using SciBERT, PyTorch, Ray, MLflow, FastAPI, automated testing, CI/CD, and containerized serving.
