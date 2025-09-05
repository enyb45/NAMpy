import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from keras.optimizers import Adam
from nampy.models import NAMLSS
from sklearn.model_selection import train_test_split

from nampy.formulas.formula_utils import*
import pandas as pd
from nampy.visuals.plot_predictions import*
from nampy.formulas.formulas import*
import matplotlib.pyplot as plt
import numpy as np


data = pd.read_csv('zambia_height92.raw', sep='\t')
print("SHAPE",data.shape)
X = data.drop(columns=['zscore'])
y = data['zscore']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# we now divide temp-data into test and validation
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print("Train X shape:", X_train.shape)
print("Train y shape:", y_train.shape)
print("Validation X shape:", X_val.shape)
print("Validation y shape:", y_val.shape)
print("Test X shape:", X_test.shape)
print("Test y shape:", y_test.shape)
print(data.head())

my_formula=all_features_additive_model(df=data,target='zscore',intercept=False)
print("Generated formula is **************************************: ", my_formula)

formula_handler= FormulaHandler()
intercept, y =formula_handler._get_intercept(my_formula)





namlss = NAMLSS(
    formula=my_formula,
    data=data, 
    family="Normal", 
    loss="nll",
)
namlss.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss={"output":namlss.Loss}, 
    metrics={"summed_output":"mse"}
    )
print("Training dataset: ", namlss.training_dataset)
namlss.fit(namlss.training_dataset, epochs=200, validation_data=namlss.validation_dataset)

loss = namlss.evaluate(namlss.validation_dataset)
print("Test Loss:", loss)

preds_all=namlss._get_plotting_preds()
print("******************************************** Graphic and static  Overview *********************************")
print('PRED ALL ',preds_all)
namlss._plot_all_effects()
#namlss._plot_single_effects()
#namlss.plot_dist() 
#namlss.plot(interactive=False)
#plot_additive_model(model=namlss)
#plot_multi_output(model=namlss)

# 🔥 Récupérer toutes les contributions des shape functions
mu_total = np.zeros(preds_all['Shapefunction0'].shape[0])
sigma_total = np.zeros(preds_all['Shapefunction0'].shape[0])

for key in preds_all.keys():
    shape_func = preds_all[key]
    mu_total += shape_func[:, 0]   # Somme des contributions à mu
    sigma_total += shape_func[:, 1]  # Somme des contributions à sigma

print("\n📌 μ prédits (premiers 5) :", mu_total[:5])
print("📌 σ prédits (premiers 5) :", sigma_total[:5])



print('***************************************** PREDICTIONS ON TRAINING DATA ************************', namlss._get_plotting_preds(training_data=True)[:5])

# === Calcul du z-score prédictif sur le jeu de test ===

# Prédictions du modèle sur X_test
preds = namlss.predict(X_test)
mu_pred = preds[:, 0]
sigma_pred = preds[:, 1]

# Calcul du z-score pour chaque observation du test
z_scores = (y_test.values - mu_pred) / sigma_pred

print("\n📊 Premiers z-scores prédits :", z_scores[:5])


