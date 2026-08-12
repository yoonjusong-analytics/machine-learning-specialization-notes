# Course 1 - Week2: Regression with Multiple Input Variables

## 1. Multiple Linear Regression

### 1.1 Multiple Features 

In real-world problem, predictions are usually based on multiple features rather than a single feature.

For example, when predicting house prices, we may use: 

- `x1`: Size of the house
- `x2`: Number of bedrooms
- `x3`: Number of floors
- `x4`: Age of the house

The model can be written as: 

f_w,b(x) = w1x1 + W2x2 + ... +wnxn + b

Using vector notation: 

f_w,b(x) = w · x + b

where:

- `x` is the feature vector.
- `w` is the parameter vector.
- `b` is the bias parameter.
- `w · x` represents the dot product.

### Key Idea

Multiple linear regression extends simple linear regression by allowing multiple input features.

Each parameter `wj` represents how strongly the corresponding feature `xj` contributes to the prediction.

### Review Point 

- Understand the difference between single and multiple linear regression.
- Understand why `x` and `w` are represented as vectors.
- Be able to explain the role of each `wj`. 
- Understand the meaning of the dot product `w · x`.

## 2. Vectorization 

Vectorization allows mathematical operations to be performed on entire vectors instead of using explicit Python loops. 

Without vectorization: 

```python
f = 0

for j in range(n): 
    f = f + w[j] * x[j]

    f= f + b
```

With NumPy vectorization: 

```python
f= np·dot(w,x) + b

``` 

### 2.2 Why Vectorization Matters 

Vectorized operations are:

- Shorter and easier to read
- Faster than Python loops
- Efficient for large datasets
- Optimized by numerical libraries such as NumPy

Vectorization becomes increasingly important as the number of features and training examples grows.

### Key Idea

Instead of calculating each feature separately: 

w1x1 + w2x2 + ... + wnxn

We can calculate them together: 

w · x

### Review Point

- Understand what np·dot() calculates.
- Compare loop-based and vectorized implementations.
- Understand why vectorization improves performance.
- Recognize vectorization as a fundamental ML programming technique.

## 3. Gradient Descent for Multiple Linear Regression

With multiple features, gradient descent updates every parameter in the weight vector w.

For each parameter: 

wj = wj - α * ∂(w,b)/∂wj

The bias is also updated: 

b = b- α * ∂J(w,b)/ ∂b

Where:
- α is the learning rate
- J(w,b) is the cost function.
- The partial derivatives represent the direction of cost change. 

The gradient for each feature is based on: 

prediction error x corresponding feature value

Conceptually:

dj_dw[j] + = error * X[i,j]

### Key Idea

The basic idea of gradient descent does not change from Week 1.

The major difference is that the model now has multiple w parameters, so the gradient must be calculated for every feature.

### Review Point

- Understand why there are multiple w gradients.
- Understand the relationship between X[i, j] and w[j].
- Understand how prediction error affects each gradient.
- Connect gradient descent with vectorized implementation. 

## 4. Gradient Descent in Practice 

### 4.1 Feature Scaling 

Feature can have very different numerical ranges.

Example:

- House size: 300 - 2,000 
- Number of bedrooms: 1 - 5 

Large differences in feature scale can make gradient descent inefficient.

Feature scaling transforms features into comparable numerical ranges.

Mean Normalization 

x = (x - μ) / range

Z-score Normalization 

x = (x - μ)/ σ

Where: 

- `μ` is the mean.
- `σ` is the standard deviation.

After Z-score normalization, features generally have: 

- Mean ≈ 0
- Standard deviation ≈ 1

### Why Feature Scaling Matters 

Without scaling, the cost function can become elongated and gradient descent may take an inefficient path toward the minimum.

With scaling, gradient descent can converge more quickly.

### Review Point

- Understand why different feature scales affect gradient descent. 
- Understand mean normalization and Z-score normalization.
- Know that scaling does not remove information from the feature.
- Understand the relationship between scaling and convergence speed.

### 4.2 Checking Gradient Descent for Convergence

During training, we can monitor the cost function:

J(w,b)

The cost should generally decrease as the number of iterations increases.

A learning curve can be used to visualize:

iteration → cost

If the cost stops decreasing significantly, gradient descent may have conveyed.

### Key Idea

A properly working gradient descent algorithm should reduce the cost over time.

### Review Point 

- Check whether J(w,b) decrease during training. 
- Understand what convergence means.
- Recognize abnormal patterns such as increasing or unstable cost. 

### 4.3 Choosing the Learning Rate 

The learning rate α determines the size of each gradient descent step.

If α is too small: 
- Gradient descent is slow.
- Many iterations are required.

If α is too large: 
- Cost may increase.
- Gradient descent may oscillate.
- The algorithm may fail to converge. 

Possible learning rates can be tested using values such as: 

```
0.001
0.003
0.01
0.03
0.1
0.3
1
```
### Key Idea

The learning rate should be large enough to learn efficiently but small enough to maintain stable convergence.

### Review Point

- Compare cost curves using different learning rates.
- Recognize divergence caused by a large learning rate.
- Understand why there is no signal best learning rate for every problem. 

### 4.4 Feature Engineering 

Feature engineering creates new features from existing features using domain knowledge or relationships in the data.

For example, if a dataset contains:

- Length
- Width

We can create: 

Area = Length x Width 

The new feature may represent the underlying relationship more directly than the original features alone. 

### Key Idea 

A model can only learn patterns from the features it receives.

Better feature representation can make important relationships easier for the model to learn. 

Review Point

- Understand the purpose of creating new features.
- Distinguish feature engineering from feature scaling.
- Think about the business meaning of engineered features.
- Avoid creating features without a logical or domain-based reason.

### 4.5 Polynomial Regression 
Linear regression can model nonlinear relationships by creating polynomial features.

For example: 

f(x) = w1x + w2x² + w3x³ + b

Although the relationship between x and the prediction is nonlinear, the model is still linear with respect to the parameters: 

w1, w2, w3 

Polynomial features can therefore allow linear regression to model curved relationships.

### 4.6 Feature Scaling with Polynomial Features

Polynomial features can have very different ranges.

For example:

x = 10
x² = 100
x³ = 1,000

Therefore, feature scaling becomes especially important when using polynomial regression.

### Review Point

- Understand how polynomial features create nonlinear relationships.
- Understand why polynomial regression is still a form of linear regression.
- Understand why scaling is important for polynomial features.
- Recognize that higher polynomial degrees do not automatically create better models.

## 5. Optional Labs
### 5.1 Python, NumPy and Vectorization

### Learning Goal
Understand how mathematical vector operations are implemented using NumPy.

Focus on:

```Python
np.array()
nd.dot()
np.zeros()
x.shape
```
Compare: 

```Python
for j in range(n):
    result += w[j] * x[j]

result = np·dot(w, x)

``` 
### Practice Strategy

1. Check the shape of each vector.
2. Calculate a small dot product manually.
3. Compare the manual result with np·dot().
4. Compare loop-based and vectorized implementations.
5. Explain why both methods produce the same result.

### Key Takeaway

Do not memorize NumPy syntax alone.

Focus on connecting: 

Mathematical vector operation
→ NumPy implementation
→ Machine learning calculation

### 5.2 Multiple Linear Regression

### Learning Goal

Understand how multiple features are used to calculate predictions and gradients.

Focus on the relationship between: 

X → training features 
w → model parameters
b → bias
prediction → np.dot(X[i], w) + b
error → prediction - y[i]

### Practice Strategy

Trace one training example manually: 

```
X[i]
↓
np.dot(X[i], w) + b
↓
prediction
↓
prediction - y[i]
↓
error
↓
gradient
```

Then compare the manual calculation with the Python implementation.

### Key Takeaway 

Focus on understanding the data flow rather than memorizing the entire function. 

### 5.3 Feature Scaling and Learning Rate

### Learning Goal

Observer how feature scaling and learning rate affect gradient descent.

Compare: 

1. Original features
2. Scaled features
3. Different learning rates

Observe how quickly the cost decreases in each case.

### Questions to Answer 

- Did scaling make convergence faster?
- Which learning rate converged fastest? 
- What happened when the learning rate was too large?
- How did the shape of the cost curve change? 

### Key Takeaway 

Feature scaling and learning rate are closely related to optimization efficiency.

### 5.4 Feature Engineering and Polynomial Regression 

### Learning Goal

Observe how transformed features allow linear regression to model nonlinear relationships.

Experiment with: 
```
x
x²
x³
```
Compare predictions before and after adding polynomial features.

### Question to Answer

- Why does adding x² change the shape of the prediction?
- Does adding more polynomial features always improve the model?
- Why should polynomial features usually be scaled?

### Key Takeaway

Feature engineering changes the representation of the problem so that the model cna capture relationships that were difficult to represent using the original features.

### 5.5 Linear Regression with Scikit-learn

### Learning Goal 

Compare the manual implementation of linear regression with a production-style machine learning library. 

```Python

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)

predictions = model.predict(X)
``` 
Important attributes:
```Python
model.coef_
model.intercept_
```

Compare these with: 
```
w ↔ model.coef_
b ↔ model.intercept_
```

### Practice Strategy

Do not treat Scikit-learn as a black box.

Connect each API step with the concepts learned previously:

```Python
model.fit(X, y)
        ↓
learn w and b

model.predict(X)
        ↓
calculate predictions 

model.coef_
        ↓

w

model.intercept_
        ↓

b
``` 

### Key Takeaway

The manual implementation explains how linear regression works.

Scikit-learn provides an efficient and standardized implementation for practical machine learning workflows. 

## 6. Week 2 Review 

After completing Week 2, I should be able to explain:
1. How multiple features extend linear regression.
2. Why vectors are useful for representing features and parameters.
3. How np.dot() implements the dot product.
4. Why vectorization is faster than Python loops.
5. How gradient descent works with multiple parameters.
6. Why feature scaling improves gradient descent.
7. How to determine whether gradient descent is converging.
8. How the learning rate affects convergence.
9. Why feature engineering can improve model representation.
10. How polynomial features model nonlinear relationships.
11. How Scikit-learn relates to the manual implementation of linear regression. 

## 7. My Observation

Add personal observations after reviewing the Week 2 concepts and completing the Optional Labs. 

| Optional lab | 중요도 | 학습 초점 |
|---|---:|---|
| Python, NumPy & Vectorization | ⭐⭐⭐⭐⭐ | `shape`, indexing, `np.dot()` | 
| Multiple Linear Regression | ⭐⭐⭐⭐⭐| `X`, `w`, `b`, prediction, gradient | 
| Feature Scaling & Learning Rate | ⭐⭐⭐⭐⭐| scaling과 convergence 관계 | 
| Feature Engineering & Polynomial | ⭐⭐⭐⭐| 새로운 feature가 모델에 미치는 영향 |
| Linear Regression with Scikit-learn | ⭐⭐⭐⭐⭐ | 직접 구현 ↔ sklearn 연결 | 




