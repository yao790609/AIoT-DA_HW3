# Interactive Classification with Streamlit

This project demonstrates an interactive machine learning classification example using Streamlit. It features Gaussian-distributed data generation, Linear Support Vector Classification (LinearSVC), and real-time 3D visualization with adjustable parameters.

---

## Problem Description

This project solves the following classification problem:

1. **Data Generation**:
   - Generate **600 random points** from a Gaussian distribution centered at (0, 0) with a variance of 10.
2. **Distance Calculation**:
   - Compute the **distance** of each point from the origin.
3. **Classification**:
   - Classify points into **Class 0** if the distance is less than a threshold (user-defined).
   - Classify points into **Class 1** if the distance is greater than or equal to the threshold.
4. **Gaussian Function**:
   - Define a function as:
     ```python
     def gaussian_function(x1, x2):
         # Implementation here
     ```
     - This function takes two inputs (`x1`, `x2`) and outputs a Gaussian-distributed value.
5. **Feature Matrix**:
   - Combine `x1`, `x2`, and the Gaussian function's output (`x3`) into a **3D feature matrix**.
6. **Training a Classifier**:
   - Use **LinearSVC** to find the hyperplane separating the two classes.
   - Retrieve and display the model's coefficients (`coef`) and intercept (`intercept`).
7. **Visualization**:
   - Display a **3D scatter plot** with data points classified into two categories (Class 0 and Class 1) using different colors.
   - Overlay a gray hyperplane representing the decision boundary.
8. **Interactive Adjustment**:
   - Implement a **Streamlit app** with a slider to adjust the classification threshold.
   - Provide real-time updates to the visualization based on the user's input.

---

## Features

- Generate random Gaussian-distributed data points.
- Classify data into two categories based on user-defined thresholds.
- Train and visualize a Linear Support Vector Classifier (LinearSVC).
- Interactive adjustment of the classification threshold using Streamlit sliders.
- 3D visualization of classified points and the decision boundary.

---

## Requirements

To run this project, ensure the following dependencies are installed:

- **Python** (3.8+)
- **Streamlit**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**

---

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/streamlit-classification.git
2. Install dependencies:
pip install -r requirements.txt

## Usage
Run the Streamlit app:
streamlit run AIoT_HW3_2.py
Open the app in your web browser using the provided local URL (e.g., http://localhost:8501).

## Visualization
3D Scatter Plot: Displays data points classified into two classes with different colors.
Decision Boundary: Shows the hyperplane separating the two classes.
Contributing
Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License. See the LICENSE file for details.
