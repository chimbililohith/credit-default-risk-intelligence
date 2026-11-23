# GitHub Upload Guide

## Upload through the GitHub website

1. Download and extract the ZIP file.
2. Sign in to GitHub and create a new **public** repository named `credit-default-risk-intelligence`.
3. Do not initialize it with another README or license because these are already included.
4. Open the extracted folder on your computer.
5. Upload the repository contents: `app.py`, `requirements.txt`, Markdown files, `.gitignore`, `assets/`, and `notebooks/`.
6. Commit the upload.
7. Open the repository and confirm that `README.md` renders correctly and that all eight chart images appear under `assets/`.

## Upload with Git (recommended for the notebook and image assets)

```bash
cd path/to/credit-default-risk-intelligence
git init
git add .
git commit -m "Initial credit default risk project"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/credit-default-risk-intelligence.git
git push -u origin main
```

Replace `YOUR-USERNAME` with your GitHub username.

## Run or deploy the Streamlit app

- Local: `pip install -r requirements.txt`, then `streamlit run app.py`.
- Streamlit Community Cloud: create a new app from the GitHub repository and select `app.py` as the main file.

## Model artifact reminder

The model and SHAP pickle artifacts are not included in this archive because they were not present in the supplied files. Follow the artifact-export section in `README.md` before expecting the live prediction page and individual SHAP explanation to work.

Review `.gitignore` before adding `.pkl` files. Do not publish raw/private customer data, credentials, or any artifact you are not permitted to share.
