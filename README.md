# Movie Search Engine

This project allows user to search for movie based on a description using a sematic search engine.
The search is performed using a pre-trained AI model for embedding text and Fasis for vector similarity search.

## Requirements
- Python 3.8+
- pip
- Streamlit
- Hugging Face Transformers
- Faiss
- pandas

## Installation
1. Clone the repository:
    git clone https://github.com/your_username/dataslush.git
    cd dataslush

2. Set up a virtual environment(recommended)
    - python -m venv venv
    - .\venv\Scripts\activate

3. Install required packages:
    pip install -r requirements.txt

4. Create a GitHub Repository
    a. Go to Github and login
    b. Create a new repository
        - click on the "NEW" button.
        - Give your repository a name, description and set it as your need(priavte or public)
        - Do not initialize the repository with a README.
        - Click "create repository"
    c. Connect your local project to the github repository
        - git init
        - git add.
        - git commit -m "Initial commit"
        - git remote add origin https://github.com/<username>/<repository>.git
        - git push -u origin main

5. Running the App
    - streamlit run app.py
    - http://localhost:8501