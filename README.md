# Higgs
Data of higgs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -----------------------------
# 1. ساخت دیتاست شبیه سازی شده
# -----------------------------

np.random.seed(42)

n_samples = 5000

# background events
background = pd.DataFrame({
    "energy": np.random.normal(100, 20, n_samples),
    "pt": np.random.normal(50, 10, n_samples),
    "mass": np.random.normal(90, 15, n_samples),
    "angle": np.random.uniform(0, 3.14, n_samples),
    "label": 0
})

# signal events (مثلاً Higgs around 125 GeV)
signal = pd.DataFrame({
    "energy": np.random.normal(140, 15, n_samples),
    "pt": np.random.normal(70, 8, n_samples),
    "mass": np.random.normal(125, 5, n_samples),
    "angle": np.random.uniform(0, 3.14, n_samples),
    "label": 1
})

data = pd.concat([background, signal])

# -----------------------------
# 2. تقسیم داده
# -----------------------------

X = data[["energy", "pt", "mass", "angle"]]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# -----------------------------
# 3. مدل ماشین لرنینگ
# -----------------------------

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# -----------------------------
# 4. نمایش توزیع جرم
# -----------------------------

plt.hist(signal["mass"], bins=50, alpha=0.5, label="Signal")
plt.hist(background["mass"], bins=50, alpha=0.5, label="Background")
plt.legend()
plt.xlabel("Invariant Mass (GeV)")
plt.ylabel("Counts")
plt.title("Mass Distribution")
plt.show()
