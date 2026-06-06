# Learners Clustering System

## Overview
This is a Machine Learning web application built using Streamlit that clusters learners based on their academic performance, study habits, and readiness indicators.

It uses K-Means Clustering and PCA to group learners and visualize patterns interactively.



##  Live Demo
https://your-app-name.streamlit.app



##  Features

- Upload CSV dataset
- Automatic data cleaning and preprocessing
- Missing value handling (mean/mode)
- Label encoding for categorical data
- Feature scaling using StandardScaler
- Interactive K-Means clustering (choose K value)
- PCA 2D visualization of clusters
- Cluster summary statistics
- Download clustered results as CSV



##  Machine Learning Workflow

### Data Preprocessing
- Missing values handled using mean and mode
- Irrelevant columns removed
- Categorical variables encoded

### Feature Scaling
- StandardScaler used to normalize data before clustering

### Clustering
- K-Means algorithm applied
- Number of clusters (K) controlled via Streamlit slider

### Dimensionality Reduction
- PCA used to reduce features to 2D for visualization



##  Cluster Labels

- High Potential Achievers  
- At-Risk Learners  
- High Achievers / Job-Ready Professionals  
- Consistent Mid Performers  
- Emerging Learners / Developing Talent  



##  Project Structure
Learners_Clustering_App/
│
├── app.py
├── requirements.txt
└── README.md


##  Installation & Setup

### 1. Clone repository
git clone https://github.com/Carolinenkatha/Learners_Clustering_App.git
cd Learners_Clustering_App

### 2. Create virtual environment
python -m venv venv

Activate it:

Windows:
venv\Scripts\activate

Mac/Linux:
source venv/bin/activate

### 3. Install dependencies
pip install -r requirements.txt

### 4. Run the app
streamlit run app.py



##  Requirements
streamlit
pandas
numpy
scikit-learn
matplotlib
seaborn




##  Output

After uploading a dataset, the app generates:

- Clustered dataset with labels
- PCA visualization plot
- Cluster summary table
- Downloadable CSV file



##  Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn



##  Use Case

- Student performance analysis
- Learner segmentation
- Academic insights
- Data-driven education decisions




