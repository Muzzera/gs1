import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import joblib

data = {
    "umidade": [15, 30, 50, 65, 70, 80, 85, 90, 95, 100],
    "chuva_mm": [0, 10, 30, 50, 55, 70, 85, 100, 120, 140],
    "risco": ["baixo", "baixo", "baixo", "médio", "médio", "médio", "alto", "alto", "alto", "alto"]
}

df = pd.DataFrame(data)
X = df[["umidade", "chuva_mm"]]
y = df["risco"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

joblib.dump(model, "modelo_risco_enchente.pkl")
print("Modelo treinado e salvo.")
