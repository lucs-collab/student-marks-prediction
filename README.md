Study Hours vs Scores - Linear Regression Model
This project demonstrates the application of Linear Regression to predict student scores based on the number of study hours. The dataset used in this project is stored locally on your system. The file path to the dataset is "C:\Users\wwwlu\Documents\student_scores.csv".

Key Steps in the Project:
Data Loading: The dataset is loaded using Pandas from a local CSV file.

Data Visualization: A scatter plot is created to visualize the relationship between study hours and scores.

Data Splitting: The data is split into training and testing sets using train_test_split from Scikit-learn (80% training, 20% testing).

Model Training: A Linear Regression model is trained on the training data.

Prediction & Evaluation: The model's performance is evaluated using metrics such as Mean Squared Error (MSE) and R2 Score. An R2 score of 0.9 indicates that the model is highly effective at predicting the scores based on the study hours.

Regression Line: The regression line is visualized along with the original data points to show the model's fit.

Prediction: A user can input the number of study hours, and the model will predict the corresponding score.

Predicted vs Actual Visualization: A final plot compares the predicted scores with the actual scores, displaying how well the model performs.

Requirements:
Python 3.x

Libraries: Pandas, NumPy, Matplotlib, Scikit-learn

How to Use:
Clone the repository.

Install the required libraries using pip install -r requirements.txt.

Make sure the dataset file is present at the specified location: C:\Users\wwwlu\Documents\student_scores.csv.

Run the script to load the dataset, train the model, and predict scores for given study hours.

Example Output:
R2 Score: 0.9 (indicating that 90% of the variance in the data is explained by the model).

Predicted Score: For example, a student who studies for 9.25 hours would likely score around a predicted value.
