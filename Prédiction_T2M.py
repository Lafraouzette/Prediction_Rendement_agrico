#prédiction du temp
"""
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

# Charger data
data = pd.read_csv("Data.csv", index_col='Date', parse_dates=True)
print(data)
print(data.head())

data['Year'] = data.index.year
data['Month'] = data.index.month
print(data['Month'])

# Modifier le DatetimeIndex pour ne contenir que les années et les mois
date_index = pd.to_datetime(data.index)
date_index_year_month = date_index.strftime('%Y-%m')
data.index = date_index_year_month
print(data)

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Changer le type pour toutes les données
data['T2M'] = pd.to_numeric(data['T2M'], errors='coerce')
data['RH2M'] = pd.to_numeric(data['RH2M'], errors='coerce')
data['T2M_MAX'] = pd.to_numeric(data['T2M_MAX'], errors='coerce')
data['T2M_MIN'] = pd.to_numeric(data['T2M_MIN'], errors='coerce')
data['PRECTOTCORR'] = pd.to_numeric(data['PRECTOTCORR'], errors='coerce')
data['ALLSKY_SFC_SW_DWN'] = pd.to_numeric(data['ALLSKY_SFC_SW_DWN'], errors='coerce')

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()




# Vérifier les corrélations
corr_matrix = data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()






# Préparer les données
X = data.drop(columns="T2M")
y = data['T2M']
y = y.values.ravel()




# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Lasso": Lasso(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
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

# Faire des prédictions avec le meilleur modèle
best_model.fit(X_train, y_train)  # Réentraîner le meilleur modèle sur l'ensemble d'entraînement complet
y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test

# Afficher les prédictions
print("Prédictions du meilleur modèle:")
print(y_pred)
print(y_test)

# Tracer les valeurs réelles par rapport aux valeurs prédites
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.show()

# Prévoir la température des années suivantes
last_year = data['Year'].iloc[-1]  # Dernière année dans vos données
future_years = range(last_year + 1, last_year + 4)  # Années suivantes
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                           columns=X.columns)
print(future_data)
# Remplir les colonnes Year et Month
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month

# Remplacer les valeurs manquantes par les moyennes des colonnes
future_data = future_data.fillna(data.mean())

# Normaliser les features
future_data_scaled = scaler.transform(future_data)

# Faire des prédictions
future_predictions = best_model.predict(future_data_scaled)

# Ajouter les prédictions au DataFrame
future_data['Predicted_T2M'] = future_predictions

print(future_data[['Predicted_T2M']])



#####prédiction du Préc

# Préparer les données
X = data.drop(columns="PRECTOTCORR")
y = data['PRECTOTCORR']
y = y.values.ravel()


# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Lasso": Lasso(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
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

# Faire des prédictions avec le meilleur modèle
best_model.fit(X_train, y_train)  # Réentraîner le meilleur modèle sur l'ensemble d'entraînement complet
y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test

# Afficher les prédictions
print("Prédictions du meilleur modèle:")
print(y_pred)
print(y_test)

# Tracer les valeurs réelles par rapport aux valeurs prédites
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.show()

# Prévoir la température des années suivantes
last_year = data['Year'].iloc[-1]  # Dernière année dans vos données
future_years = range(last_year + 1, last_year + 4)  # Années suivantes
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                           columns=X.columns)
print(future_data)
# Remplir les colonnes Year et Month
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month

# Remplacer les valeurs manquantes par les moyennes des colonnes
future_data = future_data.fillna(data.mean())

# Normaliser les features
future_data_scaled = scaler.transform(future_data)

# Faire des prédictions
future_predictions = best_model.predict(future_data_scaled)

# Ajouter les prédictions au DataFrame
future_data['Predicted_PRECTOTCORR'] = future_predictions

print(future_data[['Predicted_PRECTOTCORR']])


#####pre de humidite 

# Préparer les données
X = data.drop(columns="RH2M")
y = data['RH2M']
y = y.values.ravel()



# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Support Vector Machine": SVR(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
    "Lasso": Lasso(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
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

# Faire des prédictions avec le meilleur modèle
best_model.fit(X_train, y_train)  # Réentraîner le meilleur modèle sur l'ensemble d'entraînement complet
y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test

# Afficher les prédictions
print("Prédictions du meilleur modèle:")
print(y_pred)
print(y_test)

# Tracer les valeurs réelles par rapport aux valeurs prédites
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.show()

# Prévoir la température des années suivantes
last_year = data['Year'].iloc[-1]  # Dernière année dans vos données
future_years = range(last_year + 1, last_year + 4)  # Années suivantes
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                           columns=X.columns)
print(future_data)
# Remplir les colonnes Year et Month
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month

# Remplacer les valeurs manquantes par les moyennes des colonnes
future_data = future_data.fillna(data.mean())

# Normaliser les features
future_data_scaled = scaler.transform(future_data)

# Faire des prédictions
future_predictions = best_model.predict(future_data_scaled)

# Ajouter les prédictions au DataFrame
future_data['Predicted_RH2M'] = future_predictions

print(future_data[['Predicted_RH2M']])




###PRE DE ALLSKY_SFC_SW_DWN



# Préparer les données
X = data.drop(columns="ALLSKY_SFC_SW_DWN")
y = data['ALLSKY_SFC_SW_DWN']
y = y.values.ravel()

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Normaliser les features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialiser les modèles
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(),
    "Gradient Boosting": GradientBoostingRegressor(),
    "Lasso": Lasso(),
    "K-Neighbors Regressor": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Support Vector Machine": SVR(),
    "XGBoost": XGBRegressor(),
    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
    "AdaBoost Regressor": AdaBoostRegressor(),
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

# Faire des prédictions avec le meilleur modèle
best_model.fit(X_train, y_train)  # Réentraîner le meilleur modèle sur l'ensemble d'entraînement complet
y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test

# Afficher les prédictions
print("Prédictions du meilleur modèle:")
print(y_pred)
print(y_test)

# Tracer les valeurs réelles par rapport aux valeurs prédites
plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
plt.xlabel('Valeurs réelles')
plt.ylabel('Valeurs prédites')
plt.title('Comparaison entre valeurs réelles et valeurs prédites')
plt.legend()
plt.show()

# Prévoir la température des années suivantes
last_year = data['Year'].iloc[-1]  # Dernière année dans vos données
future_years = range(last_year + 1, last_year + 4)  # Années suivantes
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                           columns=X.columns)
print(future_data)
# Remplir les colonnes Year et Month
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month

# Remplacer les valeurs manquantes par les moyennes des colonnes
future_data = future_data.fillna(data.mean())

# Normaliser les features
future_data_scaled = scaler.transform(future_data)

# Faire des prédictions
future_predictions = best_model.predict(future_data_scaled)

# Ajouter les prédictions au DataFrame
future_data['Predicted_ALLSKY_SFC_SW_DWN'] = future_predictions

print(future_data[['Predicted_ALLSKY_SFC_SW_DWN']])























#######dataframe of all this future data yeeey

# Ajouter les prédictions au DataFrame futur

print(future_data)



# Remplacer les valeurs manquantes par les moyennes des colonnes dans les données futures pour chaque modèle
future_data_temp = future_data.copy()
print(future_data_temp)
future_data_prec = future_data.copy()
print(future_data_prec)
future_data_hum = future_data.copy()
print(future_data_hum)
future_data_sw = future_data.copy()
print(future_data_sw)


future_data_temp = future_data_temp.reindex(columns=X.columns).fillna(data.mean())
future_data_prec = future_data_prec.reindex(columns=X.columns).fillna(data.mean())
future_data_hum = future_data_hum.reindex(columns=X.columns).fillna(data.mean())
future_data_sw = future_data_sw.reindex(columns=X.columns).fillna(data.mean())

# Normaliser les features
future_data_temp_scaled = scaler.transform(future_data_temp)
future_data_prec_scaled = scaler.transform(future_data_prec)
future_data_hum_scaled = scaler.transform(future_data_hum)
future_data_sw_scaled = scaler.transform(future_data_sw)

# Faire des prédictions pour chaque variable
future_predictions_temp = best_model.predict(future_data_temp_scaled)
future_predictions_prec = best_model.predict(future_data_prec_scaled)
future_predictions_hum = best_model.predict(future_data_hum_scaled)
future_predictions_sw = best_model.predict(future_data_sw_scaled)

# Ajouter les prédictions au DataFrame futur
future_data['Predicted_T2M'] = future_predictions_temp
future_data['Predicted_PRECTOTCORR'] = future_predictions_prec
future_data['Predicted_RH2M'] = future_predictions_hum
future_data['Predicted_ALLSKY_SFC_SW_DWN'] = future_predictions_sw


print(future_data[['Predicted_T2M', 'Predicted_PRECTOTCORR', 'Predicted_RH2M', 'Predicted_ALLSKY_SFC_SW_DWN']])


print(future_data.head)

"""

################AUTRE CODE##################


####CHARGE PACKAGE
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
data = pd.read_csv("Data.csv", index_col='Date', parse_dates=True)

data['Year'] = data.index.year
data['Month'] = data.index.month

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Changer le type pour toutes les données
data['T2M'] = pd.to_numeric(data['T2M'], errors='coerce')
data['RH2M'] = pd.to_numeric(data['RH2M'], errors='coerce')
data['T2M_MAX'] = pd.to_numeric(data['T2M_MAX'], errors='coerce')
data['T2M_MIN'] = pd.to_numeric(data['T2M_MIN'], errors='coerce')
data['PRECTOTCORR'] = pd.to_numeric(data['PRECTOTCORR'], errors='coerce')
data['ALLSKY_SFC_SW_DWN'] = pd.to_numeric(data['ALLSKY_SFC_SW_DWN'], errors='coerce')

# Supprimer les lignes avec des valeurs manquantes
data = data.dropna()

# Créer des boxplots pour les variables climatiques
plt.figure(figsize=(14, 8))
sns.boxplot(data=data[['T2M', 'RH2M', 'PRECTOTCORR', 'ALLSKY_SFC_SW_DWN']])
plt.title('Boxplots des variables climatiques')
plt.xlabel('Variables climatiques')
plt.ylabel('Valeurs')
plt.savefig('boxplot.png') 
plt.show()


##########MODELE###########

# Fonction pour entraîner et prédire
def train_and_predict(data, target_column, future_years, save_name):
    X = data.drop(columns=target_column)
    y = data[target_column].values.ravel()
    
    # Normaliser les features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
    
    # Initialiser les modèles
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(),
        "Gradient Boosting": GradientBoostingRegressor(),
        "Support Vector Machine": SVR(),
        "CatBoosting Regressor": CatBoostRegressor(verbose=False),
        "AdaBoost Regressor": AdaBoostRegressor(),
        "Lasso": Lasso(),
        "K-Neighbors Regressor": KNeighborsRegressor(),
        "Decision Tree": DecisionTreeRegressor(),
        "XGBoost": XGBRegressor(),
        "LightGBM": LGBMRegressor()
    }
    
    # Entraîner et évaluer les modèles
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            "Mean Squared Error": mse,
            "R^2 Score": r2
        }
        
        
    
       
        
    # Sélectionner le meilleur modèle
    best_model_name = min(results, key=lambda name: results[name]["Mean Squared Error"])
    best_model = models[best_model_name]
    
    print(f"Best Model: {best_model_name}")
    print(f"  Mean Squared Error: {results[best_model_name]['Mean Squared Error']}")
    print(f"  R^2 Score: {results[best_model_name]['R^2 Score']}")

    
    # Faire des prédictions avec le meilleur modèle
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)  # Faire des prédictions sur l'ensemble de test
    
    
    
    
    # Afficher les prédictions
    print("Prédictions du meilleur modèle:")
    print(y_pred)
    print(y_test)

    # Tracer les valeurs réelles par rapport aux valeurs prédites
    plt.scatter(y_test, y_pred, color='blue', label='Prédiction')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Valeurs réelles')
    plt.xlabel('Valeurs réelles')
    plt.ylabel('Valeurs prédites')
    plt.title('Comparaison entre valeurs réelles et valeurs prédites')
    plt.savefig(f'{save_name}.png') 
    plt.legend()

    plt.show()

    

    # Prévoir les années futures
    future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'),
                               columns=X.columns)
    future_data['Year'] = future_data.index.year
    future_data['Month'] = future_data.index.month
    future_data = future_data.fillna(data.mean())
    future_data_scaled = scaler.transform(future_data)
    future_predictions = best_model.predict(future_data_scaled)
    future_data[f'Predicted_{target_column}'] = future_predictions
    
    return future_data[[f'Predicted_{target_column}']]

# Prévoir les futures années
last_year = data['Year'].iloc[-1]
future_years = range(last_year + 1, last_year + 6)

# Prédictions pour chaque variable
future_t2m = train_and_predict(data, 'T2M', future_years,'Prédiction_Temp')
future_prec = train_and_predict(data, 'PRECTOTCORR', future_years,'Prédiction_Précipitation')
future_hum = train_and_predict(data, 'RH2M', future_years,'Prédiction_Humidité')
future_sw = train_and_predict(data, 'ALLSKY_SFC_SW_DWN', future_years,'Prédiction_Soliel')


"""
# Combiner toutes les prédictions
future_data_combined = pd.concat([future_t2m, future_prec, future_hum, future_sw], axis=1)

# Afficher les prédictions combinées
print(future_data_combined)

# Sauvegarder les prédictions dans un fichier CSV
future_data_combined.to_csv('future_predictions_combined.csv')




"""


# Ajouter les prédictions au DataFrame futur
future_data = pd.DataFrame(index=pd.date_range(start=f"{future_years[0]}-01", end=f"{future_years[-1]}-12", freq='MS'))
future_data['Predicted_T2M'] = future_t2m.values
future_data['Predicted_PRECTOTCORR'] = future_prec.values
future_data['Predicted_RH2M'] = future_hum.values
future_data['Predicted_ALLSKY_SFC_SW_DWN'] = future_sw.values

print(future_data)

# Sauvegarder les prédictions dans un fichier CSV
future_data.to_csv('future_predictions_combined.csv')


# Grouper les données futures par année
future_data.index = pd.to_datetime(future_data.index)
future_data_by_year = future_data.groupby(future_data.index.year).mean()
print(future_data_by_year)
future_data_by_year.to_csv('future_data_by_year.csv')

dataset=pd.read_csv("MH.csv",sep=";",index_col='Date',parse_dates=True)










































