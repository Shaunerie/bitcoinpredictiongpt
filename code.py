import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Charger les données historiques du Bitcoin (assurez-vous d'avoir un fichier CSV avec des données)
# Les colonnes attendues sont : Date, Open, High, Low, Close, Volume, Market Cap
data = pd.read_csv('historical_bitcoin_data.csv')

# Convertir la colonne 'Date' en datetime et en valeur numérique
data['Date'] = pd.to_datetime(data['Date'])
data['Date'] = data['Date'].map(lambda x: x.timestamp())

# Sélectionner les colonnes d'intérêt
X = data[['Date', 'Open', 'High', 'Low', 'Volume', 'Market Cap']]
y = data['Close']

# Diviser les données en ensembles de formation et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraîner le modèle de régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédire les valeurs de test
y_pred = model.predict(X_test)

# Calculer et afficher l'erreur quadratique moyenne
mse = mean_squared_error(y_test, y_pred)
print('Erreur quadratique moyenne :', mse)
