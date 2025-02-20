#import package
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neighbors import KNeighborsRegressor
from catboost import CatBoostRegressor


##############Prétraitement des Données#####

# Charger data
data=pd.read_csv("MH.csv",sep=";",index_col='Date',parse_dates=True)
print(data)
print(data.head)

data.index = pd.to_datetime(data.index)
print(data.index)
data['Year'] = data.index.year
print(data['Year'])
data.index = data['Year']



#chage type
data['T2M(°C)'] = pd.to_numeric(data['T2M(°C)'], errors='coerce')
data['PRECTOTCORR(mm)'] = pd.to_numeric(data['PRECTOTCORR(mm)'], errors='coerce')
data['Humidité'] = pd.to_numeric(data['Humidité'], errors='coerce')
data['ALLSKY_SFC_SW_DWN'] = pd.to_numeric(data['ALLSKY_SFC_SW_DWN'], errors='coerce')
data['production(T)*10000'] = pd.to_numeric(data['production(T)*10000'], errors='coerce')


# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()


# Analyse exploratoire des données
sns.pairplot(data, x_vars=['T2M(°C)', 'Humidité', 'PRECTOTCORR(mm)', 'ALLSKY_SFC_SW_DWN'], y_vars=['production(T)*10000'], kind='scatter')
plt.savefig('analyse donnée.png') 
plt.show()



# Calculer la corrélation entre les variables
correlation_matrix = data.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Matrice de corrélation')
plt.savefig('Matrice_corréla.png') 
plt.show()

"""
# Créer des boxplots pour les variables climatiques
plt.figure(figsize=(14, 8))
sns.boxplot(data=data[['T2M(°C)', 'Humidité', 'PRECTOTCORR(mm)', 'ALLSKY_SFC_SW_DWN']])
plt.title('Boxplots des variables climatiques')
plt.xlabel('Variables climatiques')
plt.ylabel('Valeurs')
plt.show()
"""

##########MODELE###########

# Préparer les données
X = data.drop(columns="production(T)*10000")
y = data['production(T)*10000']
y = y.values.ravel()


'''
# Vérifier les corrélations
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
'''



# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Lasso": Lasso(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "XGBoost": XGBRegressor(),
    "LightGBM": LGBMRegressor()
}

# Entraîner et évaluer les modèles
results = {}
coefficients = {}  # Dictionnaire pour stocker les coefficients des modèles

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        "Mean Squared Error": mse,
        "R^2 Score": r2
    }
    # Sauvegarder les coefficients des modèles interprétables
    if isinstance(model, LinearRegression):
        coefficients[name] = {
            "Coefficients": model.coef_,
            "Intercept": model.intercept_
        }
    
    print(f"{name}:")
    print(f"  Mean Squared Error: {mse}")
    print(f"  R^2 Score: {r2}")
    print()

# Sélectionner le meilleur modèle
best_model_name = min(results, key=lambda name: results[name]["Mean Squared Error"])
best_model = models[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"  Mean Squared Error: {results[best_model_name]['Mean Squared Error']}")
print(f"  R^2 Score: {results[best_model_name]['R^2 Score']}")



"""
# Afficher les coefficients du modèle
# Afficher les coefficients du meilleur modèle
if best_model_name == "Linear Regression":
    coefficients = pd.DataFrame(best_model.coef_, X.columns, columns=['Coefficient'])
    print(coefficients)
elif best_model_name == "Support Vector Machine":
    # Pour le modèle de Machine à Vecteurs de Support, les coefficients sont accessibles via le champ "dual_coef_" 
    # pour les noyaux linéaires.
    # Vérifiez d'abord si le noyau est linéaire
    if best_model.kernel == "linear":
        coefficients = pd.DataFrame(best_model.dual_coef_.T, X.columns, columns=['Coefficient'])
        print(coefficients)
    else:
        print("Ce modèle utilise un noyau non linéaire, les coefficients ne sont pas directement interprétables.")
else:
    # Pour les autres modèles comme RandomForestRegressor et GradientBoostingRegressor,
    # les coefficients ne sont pas directement interprétables comme dans le cas de la régression linéaire.
    print("Ce modèle ne fournit pas de coefficients directement interprétables.")
"""
'''
# Afficher les coefficients des modèles interprétables
for name, coef_data in coefficients.items():
    print(f"Coefficients du modèle {name}:")
    for i, coef in enumerate(coef_data["Coefficients"]):
        print(f"Variable {i+1}: {coef}")
    print(f"Intercept du modèle {name}: {coef_data['Intercept']}")
    print()

# Sélectionner le meilleur modèle
best_model_name = min(results, key=lambda name: results[name]["Mean Squared Error"])
best_model = models[best_model_name]

print(f"Best Model: {best_model_name}")
print(f"  Mean Squared Error: {results[best_model_name]['Mean Squared Error']}")
print(f"  R^2 Score: {results[best_model_name]['R^2 Score']}")


# Faire des prédictions avec le meilleur modèle
best_model.fit(X_train, y_train)  # Réentraîner le meilleur modèle sur l'ensemble d'entraînement complet
y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test

# Afficher les prédictions
print("Prédictions du meilleur modèle:")
print(y_pred)



# Tracer les valeurs réelles par rapport aux valeurs prédites
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.show()

'''

# Prévoir les futures années
last_year = data['Year'].iloc[-1]
future_years = range(last_year + 1, last_year + 6)


# Prévoir les années futures
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                           columns=X.columns)
future_data['Year'] = future_data.index.year
future_data = future_data.fillna(data.mean())
future_data_scaled = scaler.transform(future_data)
future_predictions = best_model.predict(future_data_scaled)
future_data['production(T)*10000'] = future_predictions

future_pre=future_data['production(T)*10000']
print(future_pre)

# Visualiser les prédictions
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.savefig('Prédiction_Produ(scatter).png') 
plt.show()


# Prédire le rendement agricole futur
data['Predicted_prod'] = best_model.predict(X_scaled)
print(data['Predicted_prod'])


# Visualiser les prédictions 
plt.plot(data.index, data['production(T)*10000'], label='Rendement actuel')
plt.plot(data.index, data['Predicted_prod'], label='Prédiction du rendement')
plt.xlabel('Date')
plt.ylabel('Rendement agricole')
plt.title('Prédiction du rendement agricole')
plt.legend()
plt.savefig('Prédiction_Produ.png') 
plt.show()







# Grouper les données futures par année
future_data.index = pd.to_datetime(future_data.index)
future_data_by_year = future_data.groupby(future_data.index.year).mean()
print(future_data_by_year)
future_data_by_year.to_csv('future_data_year.csv')














