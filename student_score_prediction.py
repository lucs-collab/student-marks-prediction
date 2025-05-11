import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


url = "http://bit.ly/w-data"
df = pd.read_csv(url)


plt.scatter(df['Hours'], df['Scores'], color='blue')
plt.title('Study Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()


X = df[['Hours']] 
y = df['Scores']  


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("Mean Squared Error:", mean_squared_error(y_test, y_pred)) 
print("R2 Score:", r2_score(y_test, y_pred)) 

line = model.coef_ * X + model.intercept_ 

plt.scatter(X, y, color='blue') 
plt.plot(X, line, color='red')   
plt.title('Regression Line: Study Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Score')
plt.grid(True)
plt.show()


hours = int(input("whats the hour"))
hours = np.array([[hours]]) 
predicted_score = model.predict(hours)
print(f"Predicted score for {hours[0][0]} study hours = {predicted_score[0]:.2f}")


plt.scatter(y_test, y_pred)
plt.plot([0, 100], [0, 100], color='red') 
plt.title("Predicted vs Actual Scores")
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.show()





