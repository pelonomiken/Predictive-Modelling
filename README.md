# Mental Health Issues Prediction

## Introduction
This project aims to predict if a student has mental health issues based on various features. The process includes data preprocessing, cleaning, exploratory data analysis, and model building using several classification algorithms. A web application is also provided to make predictions based on user inputs.

## Libraries Used
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Flask
- Scikit-learn

## Usage
To run this project:
1. **Clone the Repository**: Clone this repository to your local machine.
    ```bash
    git clone https://github.com/your-repo-url.git
    ```
2. **Install Dependencies**: Install the required libraries using `pip`.
    ```bash
    pip install -r requirements.txt
    ```
3. **Run the Jupyter Notebook or Python Script**: Execute the notebook or script to preprocess data and train models.
4. **Run the Web Application**:
    - Load the trained KNN model from the pickle file (`knnc.pkl`).
	```bash
        python model.py
        ```
    - Run the Flask application.
        ```bash
        python app.py
        ```
    - Open your web browser and go to `http://127.0.0.1:5000/`.

## Acknowledgments
- Developed by Aobakwe.R.P.Kenosi.
