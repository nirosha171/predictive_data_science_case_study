from sklearn import svm
import numpy as np

# Sample Data
X = np.array([[1], [2], [3], [4], [5], [6]])
y = np.array([0, 0, 0, 1, 1, 1])

# Create SVM Model
model = svm.SVC(kernel='linear')

# Train Model
model.fit(X, y)

# Prediction
prediction = model.predict([[3.5]])

print("Prediction for 3.5:", prediction[0])