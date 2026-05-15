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

## 🛠️ Tech Stack & Tools (الأدوات المستخدمة)
To build this project from scratch, the following professional data science and machine learning stack was utilized:

* **Python 3:** The core programming language used for scripting the algorithms.
* **NumPy:** Used for high-performance scientific computing and handling **Vectorized Matrix Operations** (Dot Products, Transpositions, and Matrix stacking).
* **Pandas:** Applied for data manipulation, loading the dataset, handling missing values, and data parsing.
* **Matplotlib & Seaborn:** Used for data visualization, plotting the Cost Function reduction, and visualizing predictions vs actual outcomes.

---

## 📊 Dataset & Features (شرح المتغيرات)
The model was trained on the classic vehicle performance dataset, using the following features after dropping missing values:
1.  **Bias ($X_0$):** A column of ones ($1s$) injected into the feature matrix to represent the intercept ($\theta_0$). It allows the regression plane to cross the Y-axis at a non-zero point.
2.  **Horsepower ($X_1$):** Engine horsepower, normalized using Min-Max Scaling to prevent large values from dominating the gradient updates.
3.  **Weight ($X_2$):** Vehicle weight (in pounds), normalized using Min-Max Scaling.
4.  **Displacement ($X_3$):** Engine displacement (engine size), normalized using Min-Max Scaling.

---

## 📐 Mathematical Formulation & Explanations (المعادلات الرياضية وشرحها)

### 1. Hypothesis Function (Matrix Form) - معادلة التوقع
Instead of using slow `for` loops to calculate predictions row by row, the model uses the dot product of the Feature Matrix ($X$) and the Weight Vector ($\Theta$):

$$\hat{y} = X \cdot \Theta$$

**Explanation of Symbols:**
* $\hat{y}$ *(Predicted Output)*: A $(n \times 1)$ vector holding the model's predicted MPG values for all rows.
* $X$ *(Feature Matrix)*: An $(n \times 4)$ matrix containing the data. $n$ is the number of samples, and the 4 columns represent [Bias, Horsepower, Weight, Displacement].
* $\Theta$ *(Weight Vector)*: A $(4 \times 1)$ vector representing the learnable parameters $[\theta_0, \theta_1, \theta_2, \theta_3]^T$.

---

### 2. Cost Function (Mean Squared Error - MSE) - دالة التكلفة لحساب نسبة الخطأ
To measure how far our model's predictions are from the real answers, we calculate the average squared difference between the predicted values ($\hat{y}$) and the actual values ($y$):

$$J(\Theta) = \frac{1}{2n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2$$

**Explanation of Symbols:**
* $J(\Theta)$ *(Cost / Loss)*: A single scalar value representing the overall error of the model. Lower is better.
* $n$: Total number of training examples (rows).
* $\hat{y}^{(i)}$: The predicted MPG for the $i$-th car.
* $y^{(i)}$: The actual (true) MPG for the $i$-th car.
* $\frac{1}{2n}$: The average over all samples, multiplied by $1/2$ to simplify the math during derivative/gradient calculations.

---

### 3. Gradient Descent Update Rule - قاعدة تحديث الأوزان والنزول التدريجي
To minimize the error, the weights ($\Theta$) are updated simultaneously by taking steps in the opposite direction of the gradient. Using Linear Algebra, the gradients for all weights are computed efficiently in a single line using the matrix transpose ($X^T$):

$$\text{Gradient} = \frac{1}{n} X^T \cdot (\hat{y} - y)$$

$$\Theta = \Theta - (\alpha \cdot \text{Gradient})$$

**Explanation of Symbols:**
* $\text{Gradient}$: A $(4 \times 1)$ vector containing the partial derivatives for each weight. It points in the direction of the steepest increase in error.
* $X^T$: The transpose of the feature matrix $X$, flipping its dimensions from $(n \times 4)$ to $(4 \times n)$ so it can be multiplied by the error vector.
* $(\hat{y} - y)$: The error vector of size $(n \times 1)$, which is the difference between predictions and reality.
* $\alpha$ *(Learning Rate)*: A tuning parameter (set to `0.5` in this project) that controls how big of a step the model takes down the error hill.
* $\Theta$: The updated weight matrix ready for the next iteration.

---

## 🔁 Evolution: From Single to Multiple Gradient Descent

This project showcases the transition from a **Simple (Single-Variable)** model to a **Multiple-Variable** optimization system, highlighting how Linear Algebra simplifies complex loops:

### 1. Single-Variable Approach
Initially, the model predicted MPG using only one feature (`horsepower`) by updating the weights individually inside the loop:
- **Hypothesis:** $\hat{y} = \theta_1 \cdot x + \theta_0$
- **Code implementation:**
```python
# Separate scalar calculations for each gradient
gradient_theta0 = error.sum() / len(df)
gradient_theta1 = (error * features['horsepower_FS']).sum() / len(df)
