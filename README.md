# Single-and-Multiple-Linear-Regression-From-Scratch
Pure Python and NumPy implementation of Single &amp; Multiple Linear Regression models from scratch using vectorized Matrix operations.
# Multiple Linear Regression from Scratch using NumPy 🚀

This repository contains a pure Python and NumPy implementation of a **Multiple Linear Regression** model, built completely from scratch without relying on machine learning libraries like Scikit-Learn. The project demonstrates the core mathematical and optimization concepts behind Gradient Descent in both single-variable and multi-dimensional feature spaces.

---

## 📌 Project Overview
The goal of this project is to predict a vehicle's fuel efficiency (**MPG - Miles Per Gallon**) based on multiple engineering features. 

By building this model from scratch, I explored and resolved several key machine learning challenges:
- Handling the **Exploding Gradient** problem using Feature Scaling.
- Transforming a loop-based algorithm into a highly optimized **Matrix Form (Vectorized Operations)** using Linear Algebra.
- Analyzing how adding multiple features improves model convergence and accuracy compared to Simple Linear Regression.

---

## 🔁 Evolution: From Single to Multiple Gradient Descent

This project showcases the transition from a **Simple (Single-Variable)** model to a **Multiple-Variable** optimization system, highlighting how Linear Algebra simplifies complex loops:

### 1. Single-Variable Approach (The Beginning)
Initially, the model predicted MPG using only one feature (`horsepower`) by updating the weights individually inside the loop:
- **Hypothesis:** $\hat{y} = \theta_1 \cdot x + \theta_0$
- **Code implementation:**
  ```python
  # Separate scalar calculations for each gradient
  gradient_theta0 = error.sum() / len(df)
  gradient_theta1 = (error * features['horsepower_FS']).sum() / len(df)
