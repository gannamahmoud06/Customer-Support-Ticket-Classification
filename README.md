# 🎫 Customer-Support-Ticket-Classification-Project 🎫

An end-to-end NLP project that classifies customer support tickets into categories (Technical, Billing, Account, Other) using Machine Learning + Data Preprocessing + Interactive Web Deployment using Gradio. 🤩

# Table of Contents
1. **Introduction**
2. **Dataset Content**
3. **Machine Learning Pipeline**
4. **Web Application Deployment**
5. **Project Structure**
6. **The Usage**

# Introduction 🤌
In this project, we address a core business problem for customer support teams: the manual sorting of tickets. By leveraging **Natural Language Processing (NLP)**, we built a system that "understands" the ticket's subject and description to automatically route it to the correct department, saving time and reducing human error. 📈

# Data Set Content 🤔
The dataset contains customer inquiries with various features. After cleaning and preprocessing, we focused on the text-based columns to gain insights and train our model:

* **Ticket Subject**: The brief title of the customer's issue.
* **Ticket Description**: The detailed explanation of the problem.
* **Ticket Type**: The original category (Target for our model).
* **Ticket Priority**: The urgency level (Critical, High, Medium, Low).
* **Ticket Channel**: Where the ticket came from (Email, Chat, Phone).

# Machine Learning Pipeline 🧠

**1. EDA and Data Cleaning**
* **Drop Unnecessary Columns**: Removed IDs and personal info (GDPR compliance).
* **Label Mapping**: Unified inconsistent categories (e.g., merging "Billing Inquiry" and "Refund Request").
* **Handling Missing Values**: Ensuring text columns are complete.

**2. Text Preprocessing (NLP)**
* **Cleaning**: Removing URLs, special characters, and numbers.
* **Tokenization & Stopwords**: Removing common words that don't add meaning.
* **Lemmatization**: Converting words back to their dictionary base.

**3. Feature Engineering**
* **TF-IDF Vectorization**: Converting text to numerical weights (Unigrams & Bigrams).
* **Label Encoding**: Converting target categories into numerical format for the model.

**4. Modeling & Evaluation**
* Trained **30+ Model Variants** including Logistic Regression, SVM, Random Forest, and XGBoost.
* Used **Champion Logic**: Automatically saving the best-performing model from each family.
* Evaluated using **Accuracy, F1-Score, and Confusion Matrix**.

# Web Application Deployment 🚀
### Uses the **Gradio** library to provide a user-friendly interface for the ML model:
1.  **Ticket Input**: Textboxes for Subject and Description.
2.  **Live Prediction**: Real-time classification into Technical, Billing, etc.
3.  **Examples**: Pre-set examples to test the model's intelligence immediately.

# Project Structure 📂
The project is organized into modular Python scripts for better maintainability:
* `preprocessing.py`: Text cleaning and NLTK setup.
* `features.py`: TF-IDF and Label Encoding logic.
* `train_ml.py`: The training loop and model saving logic.
* `evaluate.py`: Performance reports and visualization plots.
* `app_gradio.py`: The main entry point for the web application.

# Usage of Web Application
1.  **Install Requirements**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Train the Models** (Optional):
    Run the training script to generate the `best_models` folder.
3.  **Run the App**:
    Execute the Gradio script:
    ```bash
    python app_gradio.py
    ```
4.  **Access the Interface**:
    Open the local or public URL provided in the terminal.