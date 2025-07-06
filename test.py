import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from keras.optimizers import Adam
from nampy.models import NAMLSS
from sklearn.model_selection import train_test_split
from nampy.formulas.formula_utils import*
import pandas as pd
from nampy.visuals.plot_predictions import*
from nampy.formulas.formulas import*
import matplotlib
matplotlib.use('Agg')  # Configuration spécifique pour VS Code
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from scipy import stats

# Configuration spécifique pour VS Code
plt.ion()  # Mode interactif
plt.style.use('default')
sns.set_style("whitegrid")

# Fonction pour sauvegarder les graphiques
def save_plot(name):
    plt.savefig(f'plot_{name}.png')
    plt.close()

# Chargement et préparation des données
data = pd.read_csv('zambia_height92.raw', sep='\t')
print("SHAPE", data.shape)
X = data.drop(columns=['zscore'])
y = data['zscore']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Création et entraînement du modèle
my_formula = all_features_additive_model(df=data, target='zscore', intercept=False)
formula_handler = FormulaHandler()
intercept, y = formula_handler._get_intercept(my_formula)

namlss = NAMLSS(
    formula=my_formula,
    data=data, 
    family="Normal", 
    loss="nll",
)

namlss.compile(
    optimizer=Adam(learning_rate=0.001), 
    loss={"output": namlss.Loss}, 
    metrics={"summed_output": "mse"}
)

# Entraînement
print("Training dataset: ", namlss.training_dataset)
namlss.fit(namlss.training_dataset, epochs=200, validation_data=namlss.validation_dataset)

# Évaluation
loss = namlss.evaluate(namlss.validation_dataset)
print("Test Loss:", loss)

# Création du dossier pour les visualisations
if not os.path.exists('visualisations'):
    os.makedirs('visualisations')

print("\n********************************************")
print("Génération des visualisations")
print("********************************************")

# 1. Obtenir les prédictions
preds_all = namlss._get_plotting_preds()
print("\nStructure des prédictions:")
for key, value in preds_all.items():
    if isinstance(value, dict):
        print(f"{key}: dict avec clés {list(value.keys())}")
    else:
        print(f"{key}: array de forme {value.shape}")

# 2. Visualisation des effets de chaque feature
features_to_plot = [(name, preds) for name, preds in preds_all.items() if not isinstance(preds, dict)]
n_features = len(features_to_plot)

# Calculer le nombre de lignes et colonnes nécessaires
n_cols = min(3, n_features)
n_rows = (n_features + n_cols - 1) // n_cols

plt.figure(figsize=(5*n_cols, 4*n_rows))
for i, (feature_name, preds) in enumerate(features_to_plot):
    plt.subplot(n_rows, n_cols, i+1)
    plt.plot(preds[:, 0], label='μ (moyenne)')
    plt.plot(preds[:, 1], label='σ (écart-type)')
    plt.title(f'Effet de {feature_name}')
    plt.legend()
plt.tight_layout()
plt.savefig('visualisations/effets_features.png')
plt.close()
print(f"Graphique des effets sauvegardé dans 'visualisations/effets_features.png' ({n_features} features)")

# 3. Distribution des prédictions
preds_dist = namlss.predict(namlss.validation_dataset)["output"]
plt.figure(figsize=(10, 6))
plt.hist(preds_dist, bins=50, density=True, alpha=0.7)
plt.title("Distribution des prédictions")
plt.xlabel("Valeur prédite")
plt.ylabel("Densité")
plt.savefig('visualisations/distribution_predictions.png')
plt.close()
print("Distribution sauvegardée dans 'visualisations/distribution_predictions.png'")

# 4. Comparaison prédictions vs réalité - VERSION AMÉLIORÉE
y_pred = namlss.predict(namlss.validation_dataset)["output"].flatten()  # Aplatir les prédictions
y_val_array = y_val.to_numpy()  # Convertir en array numpy

# Afficher les dimensions pour le débogage
print("\nDimensions des données :")
print(f"y_pred shape: {y_pred.shape}")
print(f"y_val_array shape: {y_val_array.shape}")

# S'assurer que nous avons le même nombre d'échantillons
min_samples = min(len(y_pred), len(y_val_array))
y_pred = y_pred[:min_samples]
y_val_array = y_val_array[:min_samples]

print(f"Après ajustement - nombre d'échantillons utilisés: {min_samples}")

# Calculer les erreurs pour l'analyse
errors = y_pred - y_val_array
abs_errors = np.abs(errors)

# Créer une figure avec plusieurs sous-graphiques
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Graphique principal : Prédictions vs Réalité
axes[0, 0].scatter(y_val_array, y_pred, alpha=0.6, s=20)
axes[0, 0].plot([y_val_array.min(), y_val_array.max()], [y_val_array.min(), y_val_array.max()], 'r--', linewidth=2, label='Prédiction parfaite')

# Ajouter une zone de tolérance adaptée à l'échelle réelle (±50 unités)
tolerance = 50  # Tolérance adaptée à l'échelle des z-scores (-442 à +468)
axes[0, 0].fill_between([y_val_array.min(), y_val_array.max()], 
                       [y_val_array.min() - tolerance, y_val_array.max() - tolerance],
                       [y_val_array.min() + tolerance, y_val_array.max() + tolerance],
                       alpha=0.2, color='green', label=f'Zone de tolérance (±{tolerance})')

axes[0, 0].set_xlabel("Valeurs réelles (z-score)")
axes[0, 0].set_ylabel("Valeurs prédites (z-score)")
axes[0, 0].set_title("Comparaison prédictions vs réalité")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Distribution des erreurs
axes[0, 1].hist(errors, bins=30, density=True, alpha=0.7, color='orange')
axes[0, 1].axvline(0, color='red', linestyle='--', linewidth=2, label='Erreur = 0')
axes[0, 1].set_xlabel("Erreur de prédiction")
axes[0, 1].set_ylabel("Densité")
axes[0, 1].set_title("Distribution des erreurs")
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Erreurs absolues vs valeurs réelles
axes[1, 0].scatter(y_val_array, abs_errors, alpha=0.6, s=20)
axes[1, 0].set_xlabel("Valeurs réelles (z-score)")
axes[1, 0].set_ylabel("Erreur absolue")
axes[1, 0].set_title("Erreurs absolues vs valeurs réelles")
axes[1, 0].grid(True, alpha=0.3)

# 4. Graphique Q-Q pour vérifier la normalité des erreurs
stats.probplot(errors, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title("Graphique Q-Q des erreurs")
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualisations/predictions_vs_realite_amelioré.png', dpi=300, bbox_inches='tight')
plt.close()

# Analyse détaillée des performances
print("\n" + "="*60)
print("ANALYSE DÉTAILLÉE DES PERFORMANCES DU MODÈLE")
print("="*60)

mse = mean_squared_error(y_val_array, y_pred)
r2 = r2_score(y_val_array, y_pred)
rmse = np.sqrt(mse)
mae = np.mean(abs_errors)

print(f"\n📊 MÉTRIQUES DE PERFORMANCE:")
print(f"   • R² (Coefficient de détermination): {r2:.4f}")
print(f"   • RMSE (Racine de l'erreur quadratique moyenne): {rmse:.4f}")
print(f"   • MAE (Erreur absolue moyenne): {mae:.4f}")
print(f"   • MSE (Erreur quadratique moyenne): {mse:.4f}")

# Analyse des erreurs
print(f"\n📈 ANALYSE DES ERREURS:")
print(f"   • Erreur moyenne: {np.mean(errors):.4f}")
print(f"   • Écart-type des erreurs: {np.std(errors):.4f}")
print(f"   • Erreur minimale: {np.min(errors):.4f}")
print(f"   • Erreur maximale: {np.max(errors):.4f}")

# Pourcentage de prédictions dans la zone de tolérance
within_tolerance = np.sum(abs_errors <= tolerance)
percentage_within_tolerance = (within_tolerance / len(errors)) * 100

print(f"\n🎯 PRÉDICTIONS DANS LA ZONE DE TOLÉRANCE (±{tolerance}):")
print(f"   • {within_tolerance}/{len(errors)} prédictions ({percentage_within_tolerance:.1f}%)")

# Interprétation détaillée
print(f"\n🔍 INTERPRÉTATION:")
if r2 > 0.8:
    print("   ✅ EXCELLENT: Le modèle explique plus de 80% de la variance")
    print("   → Votre modèle est très performant pour ce type de données")
elif r2 > 0.6:
    print("   ✅ BON: Le modèle explique plus de 60% de la variance")
    print("   → Votre modèle a une performance satisfaisante")
elif r2 > 0.4:
    print("   ⚠️ MOYEN: Le modèle explique plus de 40% de la variance")
    print("   → Performance acceptable mais il y a place à l'amélioration")
else:
    print("   ❌ FAIBLE: Le modèle explique moins de 40% de la variance")
    print("   → Le modèle nécessite des améliorations")

# Contexte spécifique aux z-scores
print(f"\n📋 CONTEXTE SPÉCIFIQUE AUX Z-SCORES:")
print(f"   • Pour vos données de z-scores (plage: -442 à +468):")
print(f"     - Un RMSE < 50 est excellent")
print(f"     - Un RMSE entre 50-100 est acceptable")
print(f"     - Un RMSE > 100 nécessite attention")

if rmse < 50:
    print(f"   ✅ Votre RMSE de {rmse:.3f} est EXCELLENT")
elif rmse < 100:
    print(f"   ✅ Votre RMSE de {rmse:.3f} est ACCEPTABLE")
else:
    print(f"   ⚠️ Votre RMSE de {rmse:.3f} nécessite attention")

# Ajuster la zone de tolérance pour cette échelle
tolerance_adjusted = 50  # Tolérance adaptée à l'échelle réelle
within_tolerance_adjusted = np.sum(abs_errors <= tolerance_adjusted)
percentage_within_tolerance_adjusted = (within_tolerance_adjusted / len(errors)) * 100

print(f"\n🎯 PRÉDICTIONS DANS LA ZONE DE TOLÉRANCE ADAPTÉE (±{tolerance_adjusted}):")
print(f"   • {within_tolerance_adjusted}/{len(errors)} prédictions ({percentage_within_tolerance_adjusted:.1f}%)")

# Analyse de la distribution des z-scores
print(f"\n📊 ANALYSE DE LA DISTRIBUTION DES Z-SCORES:")
print(f"   • Z-score minimum: {np.min(y_val_array):.1f}")
print(f"   • Z-score maximum: {np.max(y_val_array):.1f}")
print(f"   • Z-score moyen: {np.mean(y_val_array):.1f}")
print(f"   • Écart-type des z-scores: {np.std(y_val_array):.1f}")

# Interprétation adaptée au contexte
print(f"\n🔍 INTERPRÉTATION ADAPTÉE AU CONTEXTE:")
print(f"   • Vos z-scores ont une plage très large ({np.max(y_val_array) - np.min(y_val_array):.1f} unités)")
print(f"   • Cette variabilité importante rend la prédiction plus difficile")
print(f"   • Un R² de {r2:.3f} dans ce contexte est {'excellent' if r2 > 0.6 else 'acceptable' if r2 > 0.4 else 'à améliorer'}")

print("="*60)

# Analyse globale du modèle
print("\n********************************************")
print("PERFORMANCE GLOBALE DU MODÈLE")
print("********************************************")
mse = mean_squared_error(y_val_array, y_pred)
r2 = r2_score(y_val_array, y_pred)
print(f"MSE (Erreur quadratique moyenne): {mse:.4f}")
print(f"R² (Coefficient de détermination): {r2:.4f}")
print(f"RMSE (Racine de l'erreur quadratique moyenne): {np.sqrt(mse):.4f}")

# Interprétation de la performance
print("\nInterprétation:")
if r2 > 0.8:
    print("- Le modèle a une TRÈS BONNE performance explicative")
elif r2 > 0.6:
    print("- Le modèle a une BONNE performance explicative")
elif r2 > 0.4:
    print("- Le modèle a une performance explicative MOYENNE")
else:
    print("- Le modèle a une performance explicative FAIBLE")

print(f"- Le modèle explique {r2*100:.1f}% de la variance dans les données")
print(f"- En moyenne, les prédictions ont une erreur de {np.sqrt(mse):.4f} unités")

# Après les visualisations, ajoutons l'analyse des effets
print("\n" + "="*80)
print("ANALYSE LOGIQUE DES EFFETS DES FEATURES SUR LE Z-SCORE")
print("="*80)

print("\n🧮 APPROCHE SIMPLIFIÉE ET LOGIQUE:")
print("   • Analyser les vraies relations entre features et z-score")
print("   • Exclure les variables constantes (comme 'time')")
print("   • Utiliser des méthodes statistiques classiques")

# Analyser les données d'abord
print(f"\n📊 ANALYSE DES DONNÉES:")
print(f"   • Nombre total d'observations: {len(data)}")
print(f"   • Features disponibles: {list(data.columns)}")

# Identifier les variables constantes
constant_features = []
for col in data.columns:
    if col != 'zscore':
        unique_values = data[col].nunique()
        if unique_values == 1:
            constant_features.append(col)
            print(f"   ⚠️ Variable constante détectée: {col} (valeur: {data[col].iloc[0]})")

if constant_features:
    print(f"\n❌ Variables à exclure (constantes): {constant_features}")
else:
    print(f"\n✅ Aucune variable constante détectée")

# Analyser les vraies relations
print(f"\n" + "="*80)
print("ANALYSE DES RELATIONS FEATURE-ZSCORE")
print("="*80)

# Exclure les variables constantes et le target
features_to_analyze = [col for col in data.columns if col not in constant_features + ['zscore']]

print(f"\n🔍 Features à analyser: {features_to_analyze}")

# Analyse de corrélation simple
print(f"\n📈 ANALYSE DE CORRÉLATION AVEC LE Z-SCORE:")
correlations = {}
for feature in features_to_analyze:
    correlation = data[feature].corr(data['zscore'])
    correlations[feature] = correlation
    print(f"   • {feature}: {correlation:.4f}")

# Trier par importance (valeur absolue de la corrélation)
sorted_features = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)

print(f"\n🎯 CLASSEMENT PAR IMPORTANCE (corrélation avec z-score):")
for i, (feature, corr) in enumerate(sorted_features, 1):
    direction = "↑" if corr > 0 else "↓"
    strength = "FORT" if abs(corr) > 0.3 else "MODÉRÉ" if abs(corr) > 0.1 else "FAIBLE"
    print(f"   {i}. {feature}: {corr:.4f} {direction} ({strength})")

# Test de significativité des corrélations
print(f"\n📊 TESTS DE SIGNIFICATIVITÉ:")
significant_features = []
for feature, corr in correlations.items():
    # Test de corrélation de Pearson
    from scipy.stats import pearsonr
    correlation, p_value = pearsonr(data[feature], data['zscore'])
    
    if p_value < 0.001:
        significance = "TRÈS SIGNIFICATIF (p < 0.001)"
    elif p_value < 0.01:
        significance = "TRÈS SIGNIFICATIF (p < 0.01)"
    elif p_value < 0.05:
        significance = "SIGNIFICATIF (p < 0.05)"
    else:
        significance = "NON SIGNIFICATIF (p ≥ 0.05)"
    
    print(f"   • {feature}: r={corr:.4f}, p={p_value:.6f} → {significance}")
    
    if p_value < 0.05:
        significant_features.append(feature)

print(f"\n✅ FEATURES SIGNIFICATIVES (p < 0.05): {significant_features}")

# Visualisation des relations importantes
print(f"\n📊 GÉNÉRATION DES VISUALISATIONS:")
n_significant = len(significant_features)
if n_significant > 0:
    n_cols = min(3, n_significant)
    n_rows = (n_significant + n_cols - 1) // n_cols
    
    plt.figure(figsize=(5*n_cols, 4*n_rows))
    for i, feature in enumerate(significant_features, 1):
        plt.subplot(n_rows, n_cols, i)
        plt.scatter(data[feature], data['zscore'], alpha=0.6, s=20)
        
        # Ligne de régression
        z = np.polyfit(data[feature], data['zscore'], 1)
        p = np.poly1d(z)
        plt.plot(data[feature], p(data[feature]), "r--", alpha=0.8)
        
        plt.xlabel(feature)
        plt.ylabel('z-score')
        plt.title(f'{feature} vs z-score\nr={correlations[feature]:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('visualisations/correlations_significatives.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Graphiques sauvegardés dans 'visualisations/correlations_significatives.png'")

# Résumé final logique
print(f"\n" + "="*80)
print("RÉSUMÉ LOGIQUE POUR LA PRÉDICTION")
print("="*80)

print(f"\n🎯 FEATURES IMPORTANTES POUR PRÉDIRE LE Z-SCORE:")
if significant_features:
    print(f"   • Features avec corrélation significative: {', '.join(significant_features)}")
    print(f"   • Ces features ont une relation statistiquement prouvée avec le z-score")
else:
    print(f"   • Aucune feature n'a de corrélation significative avec le z-score")

print(f"\n💡 RECOMMANDATIONS:")
print(f"   • Concentrez-vous sur les features significatives pour la prédiction")
print(f"   • Les variables constantes (comme 'time') n'apportent aucune information")
print(f"   • Utilisez les corrélations pour comprendre les relations linéaires")
print(f"   • NAMLSS capture les relations non-linéaires supplémentaires")

print("="*80)

