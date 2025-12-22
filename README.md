<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Salary Prediction using Simple Linear Regression</title>
    <style>
        body {
            font-family: Arial, Helvetica, sans-serif;
            background-color: #f9f9f9;
            color: #222;
            line-height: 1.7;
            margin: 40px;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        code {
            background-color: #eee;
            padding: 3px 6px;
            border-radius: 4px;
        }
        .box {
            background: #ffffff;
            padding: 20px;
            border-left: 6px solid #4CAF50;
            margin: 20px 0;
            border-radius: 6px;
        }
        pre {
            background: #272822;
            color: #f8f8f2;
            padding: 15px;
            border-radius: 6px;
            overflow-x: auto;
        }
        ul li {
            margin-bottom: 8px;
        }
    </style>
</head>

<body>

<h1>💼 Salary Prediction using Simple Linear Regression</h1>

<p>
This project demonstrates a complete <b>Machine Learning workflow</b> using 
<b>Simple Linear Regression</b>, starting from raw salary data to deploying a 
fully working <b>web application</b>.
</p>

<hr>

<h2>📌 Problem Statement</h2>
<p>
To predict an employee's salary based on years of experience using a 
<b>Simple Linear Regression</b> model.
</p>

<hr>

<h2>📊 Dataset Description</h2>
<ul>
    <li><b>X (Independent Variable):</b> Years of Experience</li>
    <li><b>y (Dependent Variable):</b> Salary</li>
</ul>

<hr>

<h2>⚙️ Technologies & Libraries Used</h2>
<ul>
    <li>NumPy</li>
    <li>Pandas</li>
    <li>Matplotlib</li>
    <li>Scikit-learn</li>
    <li>Pickle</li>
    <li>HTML, CSS (Frontend)</li>
    <li>Python (Backend)</li>
</ul>

<hr>

<h2>🧠 Project Workflow</h2>

<div class="box">
<pre>
Raw Salary Data
      │
      ▼
Train-Test Split (sklearn)
      │
      ▼
Exploratory Data Analysis
(Scatter Plot)
      │
      ▼
Simple Linear Regression Model
(y = mx + c)
      │
      ▼
Model Training
      │
      ▼
Predictions
      │
      ▼
Evaluation (MSE & R² Score)
      │
      ▼
Model Serialization (Pickle)
      │
      ▼
Web Application Deployment
</pre>
</div>

<hr>

<h2>🔍 Data Preparation</h2>
<p>
The dataset is split into training and testing sets using 
<code>train_test_split</code> from <b>sklearn</b>.
</p>

<pre>
training_data = pd.DataFrame({
    'X_train_values': X_train.to_numpy().ravel(),
    'y_train_values': y_train
})

testing_data = pd.DataFrame({
    'X_test_values': X_test.to_numpy().ravel(),
    'y_test_values': y_test
})
</pre>

<hr>

<h2>📈 Data Visualization</h2>
<p>
A scatter plot is used to visualize the actual training data points, and 
the regression line represents model predictions.
</p>

<pre>
plt.scatter(X_train, y_train, color='red', marker='*')
plt.plot(X_train, predictions, color='green')
plt.title('Training Data with Regression Line')
plt.show()
</pre>

<hr>

<h2>📐 Model Building</h2>
<p>
The model is built using <b>LinearRegression</b> from sklearn.
</p>

<ul>
    <li><b>m (Slope):</b> Learned from training data</li>
    <li><b>c (Intercept):</b> Bias term</li>
</ul>

<pre>
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(X_train, y_train)

predictions = reg.predict(X_train)
</pre>

<hr>

<h2>📊 Model Evaluation</h2>
<p>
The model is evaluated using:
</p>

<ul>
    <li><b>Mean Squared Error (MSE)</b></li>
    <li><b>R² Score (Accuracy)</b></li>
</ul>

<p>
Both training and testing datasets are evaluated to check model performance 
and overfitting.
</p>

<hr>

<h2>💾 Model Saving</h2>
<p>
The trained model is saved using <b>pickle</b> for later use in deployment.
</p>

<pre>
import pickle

with open('salary_model.pkl', 'wb') as file:
    pickle.dump(reg, file)
</pre>

<hr>

<h2>🌐 Deployment</h2>
<p>
The saved model is loaded in a virtual environment and integrated with:
</p>

<ul>
    <li>Frontend for user input</li>
    <li>Backend for prediction logic</li>
</ul>

<p>
The web application takes years of experience as input and predicts salary 
in real time.
</p>

<hr>

<h2>✅ Conclusion</h2>
<p>
This project showcases a complete end-to-end implementation of a 
<b>Simple Linear Regression model</b> — from data preprocessing and visualization 
to model deployment as a web application.
</p>

<p>
Perfect for beginners learning Machine Learning fundamentals and real-world deployment 🚀
</p>

</body>
</html>
