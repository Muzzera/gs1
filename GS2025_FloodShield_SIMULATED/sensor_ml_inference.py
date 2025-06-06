import joblib
import pandas as pd
import time

# Carregar modelo treinado
model = joblib.load("modelo_risco_enchente.pkl")

print("Simulando leitura de dados de umidade e chuva...")

# Simulação de leituras
leituras = [
    {"umidade": 20, "chuva_mm": 10},
    {"umidade": 45, "chuva_mm": 35},
    {"umidade": 75, "chuva_mm": 90},
    {"umidade": 90, "chuva_mm": 130}
]

for leitura in leituras:
    entrada = pd.DataFrame([[leitura["umidade"], leitura["chuva_mm"]]],
                           columns=["umidade", "chuva_mm"])
    risco = model.predict(entrada)[0]
    print(f"🔎 Umidade: {leitura['umidade']}% | Chuva: {leitura['chuva_mm']}mm → Risco: {risco.upper()}")

    if risco == "alto":
        print("🚨 ALARME: RISCO DE ENCHENTE ELEVADO!")
    time.sleep(3)
