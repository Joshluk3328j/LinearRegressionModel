# Custom Linear Regression (Python & C++)

This repository contains simple, hand-crafted **linear regression models** implemented from scratch in both **Python** and **C++**, without using any machine learning libraries like `scikit-learn`. The goal is to understand how linear regression truly works under the hood by building it from first principles.

---

## 🔧 Features

| Feature                      | Python | C++  |
|-----------------------------|--------|------|
| Manual fit/predict logic    | ✅     | ✅   |
| R² score implementation     | ✅     | ✅   |
| Residual visualization      | ✅     | ❌   |
| Library-independent logic   | ✅     | ✅   |

---

## 🧠 What I Learned

- How to compute the slope (`m`) and intercept (`c`) from first principles
- How linear regression fits the "best line" using the least squares method
- The meaning of the **R² score** as a measure of model accuracy
- How to visualize prediction errors (residuals) using vertical error lines
- The difference between:
  - Perfectly linear vs. noisy data
  - Local slope approximation vs. true regression fit
  - Python vs. C++ implementations of the same algorithm

---

## 🔬 Data Samples & Visualization

### 🟢 Perfectly Linear Data: `y = 2x + 1`

This clean dataset is used to show that the model **perfectly fits** linear data:

```python
x = [1, 2, 3, ..., 10]
y = [3, 5, 7, ..., 21]  # y = 2x + 1
```

### 🟠 Noisy Linear Data: `y = 2x + 1`

This noisy dataset is used to show that the model **tries to generalise** noisy inear data:

``` python
x = [1, 2, 3, ..., 10]
y = [3, 6, 7, 10, 9, 13, 14, 15, 19, 18]  # some deviation from perfect line
```

### 📁 Folder Structure
```
custom-linear-regression/
│
├── python/
│   ├── linear_regression.py
│   ├── plot_perfect.png
│   ├── plot_noisy.png
│
├── cpp/
│   └── linear_regression.cpp
│
├── README.md
```

#### 🙌 Acknowledgment
- [YouTube: Linear Regression Explained](https://www.youtube.com/watch?v=P8hT5nDai6A).
- [YouTube: Linear Regression Tutorial](https://www.youtube.com/watch?v=S0ptaAXNxBU).

This was built as a personal learning project to truly understand how linear regression models work internally.
If you're learning too — fork it, play around, or suggest improvements!

