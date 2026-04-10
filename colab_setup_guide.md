# Google Colab Setup (From Existing GitHub Repo)

Since your code is already on GitHub, the process is much easier! We can pull your code straight from GitHub into your Google Drive, and then download the datasets directly into that folder.

## Step 1: Create a Colab Notebook & Mount Google Drive
First, we need to connect a Google Colab notebook to your Google Drive so that your files are saved permanently.

1. Go to [Google Colab](https://colab.research.google.com/) and click **New Notebook**.
2. In the first cell, paste the following code to mount your Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```
3. Run the cell (Shift + Enter). Allow permission when prompted.

## Step 2: Clone Your GitHub Repository to Google Drive
Now we will clone your GitHub repository into your Google Drive securely.

Run this in a new cell (replace `YOUR_GITHUB_USERNAME` and `YOUR_REPO_NAME` with your actual details):

```bash
# Navigate to your Google Drive
%cd /content/drive/MyDrive/

# (Optional) Create a specific folder for your research if you want
# !mkdir -p Final_Year_Research
# %cd Final_Year_Research

# Clone your GitHub repository (This downloads your code from GitHub into your Drive)
!git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git

# Move inside your newly downloaded project folder
%cd YOUR_REPO_NAME
```

## Step 3: Download the Dataset Directly to Google Drive
Instead of using your computer's internet, we will use Google's ultra-fast servers to download the massive dataset directly into your newly cloned folder.

Run this in a new cell:

```bash
# Create a datasets folder inside your project
!mkdir -p datasets/VSD

# Navigate into the dataset folder
%cd datasets/VSD

# Download the dataset directly (Replace URL with the actual dataset link)
!wget -c "INSERT_YOUR_DATASET_URL_HERE" -O vsd_dataset.zip

# Unzip the downloaded dataset
!unzip vsd_dataset.zip
```

## Step 4: Pushing Updates Back to GitHub (Optional)
When you make changes to your code inside Colab and want to save them back to GitHub, you will need a GitHub Personal Access Token (Settings -> Developer Settings -> Personal access tokens).

```bash
# Navigate to your project folder
%cd /content/drive/MyDrive/YOUR_REPO_NAME

# Add your changes
!git add .
!git commit -m "Updated code from Colab"

# Push back to GitHub (You will need to use your token for authentication)
!git push https://<YOUR_GITHUB_TOKEN>@github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git main
```
