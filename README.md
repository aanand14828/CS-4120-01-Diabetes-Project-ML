# CS-4120-01-Diabetes-Project-ML
Set up and running instructions
This project involves predicting diabetes disease progression using the Diabetes Dataset available in scikit-learn. datasets. Our goal was to build simple baseline models for both regression and classification before moving toward more complex neural network models.

To run the project, first clone the repository and open the folder in your IDE or terminal. Install all required packages by running pip install -r requirements.txt. You don’t need to download any dataset manually — it’s automatically loaded through src/data.py.

Once everything is set up, you can train the baseline models by running python src/train_baselines.py. This will train both Linear Regression and Decision Tree Regressor for the regression task, as well as Logistic Regression and Decision Tree Classifier for the classification task. The results will be saved as CSV tables under src/notebooks/tables/.

If you’d like to generate visualizations such as the correlation heatmap, confusion matrix, residual plots, and target distribution, run python src/viz.py. The plots will be stored in src/notebooks/figures/.

The best-performing baseline models were Linear Regression for regression and Logistic Regression for classification. Both achieved stable validation and test performance, showing that the dataset is mostly linear and well-behaved. These models serve as a strong starting point for the upcoming neural network phase, where we plan to experiment with deeper architectures and optimizers like Adam to improve performance further.

Quick Results Summary
- Linear Regression performed best for regression (lowest MSE).
- Logistic Regression achieved the highest accuracy for classification.
- Residuals showed a roughly normal distribution, indicating a stable model fit.
- The dataset was balanced, and both models generalized reasonably well.

Some Important Points
- Random seeds are fixed (random_state=42) for reproducibility.
- No raw data is committed — the dataset loads automatically from scikit-learn.
- The project structure is organized to be compatible with MLflow for later logging, though tracking is not yet active in this phase.

We used ChatGPT & Gemini for code review and formatting guidance for the plots.
