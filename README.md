# Simple Linear Regression Model

This project demonstrates how to build and evaluate a simple linear regression model using Python and scikit-learn. The example uses a salary dataset to predict salary based on years of experience.

## Steps Covered

- **Data Loading:**  
  The dataset is loaded using pandas.

- **Data Exploration:**  
  Basic statistics and missing values are checked.  
  A scatter plot is used to visualize the relationship between years of experience and salary.

- **Feature Selection:**  
  - `YearsExperience` is used as the input feature.
  - `Salary` is used as the target variable.

- **Train-Test Split:**  
  The data is split into training and testing sets to evaluate model performance.

- **Model Training:**  
  A `LinearRegression` model is trained on the training data.

- **Model Evaluation:**  
  The model's accuracy is checked using the test set (R² score).

- **Prediction:**  
  The model predicts salary for a given value of years of experience.

- **Visualization:**  
  The regression line is plotted along with the data points to show the model fit.

## Example Usage

```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Prepare data
input_data = dataset[['YearsExperience']]
output_data = dataset['Salary']

# Split data
X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=30)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
print(model.score(X_test, y_test))

# Predict
model.predict(pd.DataFrame([[2.3]], columns=['YearsExperience']))
```

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

This notebook is a practical guide for beginners to understand and implement simple linear regression for predictive modeling.