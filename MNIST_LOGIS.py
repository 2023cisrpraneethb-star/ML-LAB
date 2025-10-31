from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

mnist = fetch_openml("mnist_784", version=1, as_frame=False)
X = mnist.data
y = mnist.target.astype(int)

plt.figure(figsize=(5,5))
for i in range(9):
    idx = np.random.randint(0, len(X))
    img = X[idx].reshape(28,28)
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(y[idx])
    plt.axis("off")
plt.show()

X = X / 255.0

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='gray')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

misclassified = np.where(y_pred != y_test)[0]
print(len(misclassified))

plt.figure(figsize=(6,6))
for i in range(9):
    idx = misclassified[i]
    img = X_test[idx].reshape(28,28)
    plt.subplot(3,3,i+1)
    plt.imshow(img, cmap="gray")
    plt.title(f"{y_test[idx]} vs {y_pred[idx]}")
    plt.axis("off")
plt.show()
