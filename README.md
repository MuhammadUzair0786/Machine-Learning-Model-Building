# Linear Regression Models: Simple & Multiple

This project demonstrates how to build, evaluate, and interpret both **Simple Linear Regression** and **Multiple Linear Regression** models using Python and scikit-learn. The examples use salary datasets to predict salary based on one or more input features.

---

## 1. Simple Linear Regression

**Simple Linear Regression** is used to predict a target variable (e.g., Salary) using a single input feature (e.g., Years of Experience).

### Steps Covered:
- Load and explore the dataset.
- Visualize the relationship between the feature and target.
- Split the data into training and testing sets.
- Train a linear regression model.
- Evaluate model performance using R² score.
- Predict salary for new data.
- Visualize the regression line.

---

## 2. Multiple Linear Regression

**Multiple Linear Regression** predicts the target variable using two or more input features (e.g., Age, Experience, Education).

### Steps Covered:
- Load and inspect the dataset.
- Visualize relationships and correlations between features.
- Split the data into training and testing sets.
- Train a multiple linear regression model.
- Evaluate model performance using R² score.
- Make predictions on new data.
- Interpret model coefficients.

---

## Example Code Snippets

**Simple Linear Regression:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

input_data = dataset[['YearsExperience']]
output_data = dataset['Salary']

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
model.predict([[2.5]])  # Predict for 2.5 years experience
```

**Multiple Linear Regression:**
```python
input_data = dataset.iloc[:, :-1]  # All columns except last (target)
output_data = dataset['salary']

X_train, X_test, y_train, y_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
model.predict([X_test.iloc[0].values])  # Predict for first test row
```

---

# Polynomial Regression

This notebook demonstrates how to use polynomial regression to model non-linear relationships between temperature and ice cream sales.

## Steps
- Load and explore the dataset.
- Visualize the relationship between temperature and sales.
- Transform features using `PolynomialFeatures`.
- Split data into train and test sets.
- Train a `LinearRegression` model on polynomial features.
- Evaluate model performance (R² score).
- Visualize the polynomial regression curve.
- Predict sales for a given temperature.

## Note
- Polynomial regression is useful when data is not linear.
- Always predict within the range of your training data for reliable results.

This notebook demonstrates how to use **Ridge** and **Lasso** regression for regularization in linear models. These techniques help prevent overfitting and improve model generalization, especially when working with datasets that have many features.

## What are Ridge and Lasso Regression?

- **Ridge Regression (L2 Regularization):**
  - Adds a penalty equal to the square of the magnitude of coefficients to the loss function.
  - Shrinks coefficients but does not set them exactly to zero.
  - Useful when all features are important but need to reduce their impact.

- **Lasso Regression (L1 Regularization):**
  - Adds a penalty equal to the absolute value of the magnitude of coefficients.
  - Can shrink some coefficients to exactly zero, effectively performing feature selection.
  - Useful when you suspect that only a few features are important.

## Steps Covered

- Load and preprocess the dataset (handle missing values, encode categorical variables, scale features).
- Split the data into training and testing sets.
- Train a standard linear regression model for comparison.
- Train and evaluate Ridge and Lasso regression models.
- Compare model performance using R² score on both train and test sets.
- Discuss the effect of the regularization parameter (`alpha`).

## Example Code

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# Split data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Ridge Regression
ridge = Ridge(alpha=1.0, max_iter=10000, random_state=42)
ridge.fit(x_train, y_train)
print("Ridge Train Score:", ridge.score(x_train, y_train))
print("Ridge Test Score:", ridge.score(x_test, y_test))

# Lasso Regression
lasso = Lasso(alpha=1.0, max_iter=10000, random_state=42)
lasso.fit(x_train, y_train)
print("Lasso Train Score:", lasso.score(x_train, y_train))
print("Lasso Test Score:", lasso.score(x_test, y_test))
```

## Notes

- Adjust the `alpha` parameter to control the strength of regularization.
- Use cross-validation to find the best `alpha` value for your data.
- Ridge is preferred when all features are useful; Lasso is preferred for feature selection.

## Requirements

- Python 3.x
- pandas
- matplotlib
- seaborn
- scikit-learn
- Jupyter Notebook

---

## Usage

1. Open the notebook in Jupyter.
2. Follow the steps for simple or multiple linear regression as needed.
3. Visualize and interpret the results.
4. Modify the code for your own datasets and features.

---
