# Week 1: Introduction to Machine Learning

## DeepLearning AI - Machine Learning Specialization

## Topic Covered

- Overview of Machine Learning 
- Supervised vs. Unsupervised Learning
- Linear Regression
- Cost Function 
- Cost Function Intuition
- Visualizing the Cost Function 

--- 

# Learning Objectives

After completing this week, I can:

- Explain what Machine Learning is. 
- Distinguish supervised learning from unsupervised learning.
- Understand how Linear Regression predicts continuous values.
- Explain the purpose of the Cost Function.
- Interpret how the Cost Function evaluates model performance.

# 1. Overview of Machine Learning 

## What is Machine Learning?

Machine Learning is a field of Artificial Intelligence that enables computers to learn patterns from data instead of relying on explicitly programmed rules.

Rather than writing fixed instructions, a model learns relationships from examples and improves its prediction based on data.

### Typical Applications

| Industry | Example | 
|----------|----------|
| Finance | Fraud Detection | 
| Retail | Product Recommendation | 
| Marketing | Disease Prediction | 

----

# 2. Supervised vs. Unsupervised Learning 

## Supervised Learning

The model learn from labeled data.

- Input (X)
- Correct Answer (Y)

Goal: 

Predict the correct output for new data.

Examples

- House Price Prediction
- Email Spam Detection
- Credit Risk Prediction 

---

## Unsupervised Learning 

The model learns patterns from unlabeled data. 

Goal: 

Discover hidden structures within data. 

Examples

- Customer Segmentation
- Topic Modeling
- Market Basket Analysis 

## Comparison 

| Supervised Learning | Unsupervised Learning |
|---------------------|-----------------------|
| Uses labeled data | Uses unlabeled data | 
| Predicts outputs | Finds hidden patterns | 
| Regression | Clustering | 
| Classification | Dimensionality Reduction | 

---

# 3. Linear Regression 

## Purpose 

Linear Regression predicts continuous numerical values.

Examples

- House Price
- Sales Revenue
- Temperature
- Stock Price (basic prediction)

---

## Model Representation 

Prediction Function

ŷ = wx + b

Where

| Symbol | Meaning | 
|--------|----------|
| x | Input Feature | 
| w | Weigh (Slope) | 
| b | Bias (Intercept) | 
| ŷ | Predicted Value | 

--- 

## Business Example 

Input 

- House Size

↓

Linear Regression Model

↓

Output

- Estimated House Price 

--- 

# 4. Cost Function

## Purpose 

The Cost Function measures how well the model fits the training data.

Lower Cost means better predictions.

Higher Cost means poorer predictions. 

---

## Formula 

\[
J(w,b)= \frac{1}{2m}\sum_{i=1}^{m}(f(x^{(i)})-y^{(i)})^2
\]

Where 

| Symbol | Meaning | 
|---------|---------|
| m | Number of training examples | 
| f(x) | Predicted value | 
| y | Actual value | 
| J | Cost | 

---

## Cost Function Intuition

Prediction

↓

Prediction Error 

↓ 

Squared Error 

↓

Average Error 

↓

Cost

Goal 

Minimize the Cost Function to find the best model parameters.

---

## Visualizing the Cost Function

The Cost Function forms a bowl-shaped curve. 

The lowest point represents the Optimal values of **w** and **b**.

This minimum point corresponds to the model with the smallest prediction error. 

---

# Optional Labs 

## Objectives 

- Understand model representation
- Visualize Linear Regression
- Observe the effect of changing parameters
- Understand how the Cost Function changes

---

## Key Python Libraries

- NumPy
- Matplotlib

--- 

## Important Functions

```python
import numpy as np
import matplotlib.pyplot as plt

plt.scatter()

plt.plot()
```
## 4. Train the Model with Gradient Descent

### 4.1 Gradient Descent 

Gradient Descent is an optimization algorithm used to minimize the cost function.

The goal is to find the values of parameters `w` and `b` that minimize: 

J(w, b)

The parameters are repeatedly updated using: 

w = w - α * dJ(w,b)/dw

b = b- α * dJ(w,b)/db

where: 

- `w`, `b`: model parameters
- `α` (alpha): learning rate
- `dJ/dw`, `dJ/db`: partial derivatives of the cost function

The partial derivative tells us the direction and rate at which the cost function changes.

Gradient descent updates the parameters in the opposite direction of the gradient in order to reduce the cost.

### 4.2 Implementing Gradient Descent 

Gradient descent repeats the parameter update until the cost function converges.

Basic process:

1. Initialize `w` and `b`
2. Calculate predictions
3. Calculate the cost 
4. Calculate gradients 
5. Update `w` and `b`
6. Repeat until convergence

Pseudo-code:
repeat until convergence:
    w = w - α * dj_dw
    b = b - α * dj_db

### 4.3 Gradient Descent Intuition

Gradient descent can be understood as moving downhill toward the minimum of the cost function.

If the derivative is positive: 

w= w - α * (postlive value) 

→ `w` decreases.

If the derivatives is negative:

w = w - α * (negative value) 

→ `w` increases.

Therefore, the update rule naturally moves the parameters toward a lower cost.

As the algorithm approaches a local minimum, the derivative becomes smaller and the parameter updates also become smaller.

### 4.4 Learning Rate 

The learning rate 'α' controls how large each gradient descent step is.

#### Learning rate too small
- Gradient descent works
- Convergence is very slow
- More iterations are required 

#### Learning rate too large
- The algorithm may overshoot the minimum 
- Cost may increase 
- Gradient descent may fail to converge 

Therefore, choosing an appropriate learning rate is important. 

Example: 

α = 0.01 

In Python:

alpha = 1.0e-2

### 4.5 Gradient Descent for Linear Regression

For linear regression: 

f_wb(x) = wx + b

The cost function is: 

J(w,b) = (1 / 2m) ∑(f_wb(xᵢ) - yᵢ)²

Gradient descent calculates the gradients: 

dj_dw = (1 / m) Σ(f_wb(xᵢ) - yᵢ)xᵢ

dj_db = (1 / m) Σ(f_wb(xᵢ) - yᵢ)

Then updates: 

w = w - α * dj_dw

b = b- α * dj_db

By repeatedly updating `w` and `b`, the regression line gradually moves towards the line that minimizes prediction error. 

### 4.6 Running Gradient Descent 

When gradient descent is running  correctly: 

- The cost `J(w,b)` should decrease over iterations.
- The parameters `w` and `b` gradually approach optimal values.
- Eventually, the cost changes very little and the model converges.

Example: 

Iteration 0     → High cost 
Iteration 100   → Lower cost
Iteration 1,000 → Much lower cost 
...
Convergence     → Cost changes very little

Monitoring the cost during training is useful for checking whether gradient descent is working properly.

## Key Takeaways 

- Linear regression predicts values using `f(x) = wx + b`.
- The cost function measures how well the model fits the training data.
- Gradient descent minimizes the cost function.
- Partial derivative determine the direction in which parameters should move.
- The learning rate controls the size of each update step.
- A learning rate that is too small causes slow convergence.
- A learning rate that is too large may prevent convergence.
- During successful training, the cost should decrease over iterations.

### Machine Learning Training Flow

Training Data 
    ↓
Linear Regression Model 
    ↓
Prediction f(x)
    ↓
Cost Function J(w,b)
    ↓
Calculate Gradients
    ↓
Update w and b
    ↓
Repeat
    ↓
Convergence 

## Optional Lab Notes

### Gradient Descent

The optional lab demonstrated how gradient descent updates model parameters over multiple iterations.

What I observed: 

- The cost decreased as the number of iterations increased.
- `w` and `b` gradually moved toward values that better fit the training data.
- The learning rate affected how quickly the model converged.
- Gradient descent connects the mathematical cost function with actual model training. 

### Important Code Concepts

```python
alpha = 0.01
iterations = 10000
