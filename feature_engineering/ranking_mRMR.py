import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import mutual_info_regression as MI
from skfda.preprocessing.dim_reduction.variable_selection import MinimumRedundancyMaximumRelevance
from skfda.representation.grid import FDataGrid
from dcor import u_distance_covariance_sqr, u_distance_correlation_sqr
from scipy.stats import pearsonr

def mutualInformation(X,y):
    return MI(X, y.ravel())[0]

def covariance_correlation(X, y):
    return np.cov(X.flatten(), y.flatten())[0,1]

def pearson_dependence(x, y):
    """Calcula la correlación de Pearson (valor absoluto)"""
    return np.abs(pearsonr(x.ravel(), y.ravel())[0])

# Número de características a seleccionar por el método
feature_counts = [25, 30, 35, 40, 45, 50, 55, 60, 66]

dependence_measures = {
    'pearson_correlation': 'pearson',
    'distance_covariance': u_distance_covariance_sqr,
    'distance_correlation': u_distance_correlation_sqr,
    'mutual_information': 'mutual_info'
}
results = []

X = pd.read_csv("../data_processed/X_processed_multiclass.csv")
y = pd.read_csv("../data_processed/y_processed_multiclass.csv")

y = np.squeeze(y)
X_fda = FDataGrid(X.values.reshape(X.shape[0], X.shape[1]))

for measure_name, measure_func in dependence_measures.items():
    print(f"\n---- Evaluando Dependence Measure: {measure_name} ----")
    if measure_func == 'pearson':
        dependence_function = pearson_dependence
    elif measure_func == 'mutual_info':
        dependence_function = mutualInformation
    else:
        dependence_function = measure_func

    for n_features in feature_counts:
        print(f"Número de características: {n_features}")
        
        f = MinimumRedundancyMaximumRelevance(n_features_to_select = n_features, dependence_measure=dependence_function, criterion='quotient')
        f.fit(X_fda, y)
        vars_selected = f.get_support()

        
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)  # Repite 5 veces

        # KNN
        knn = KNeighborsClassifier()
        scores_knn = cross_val_score(knn, X.iloc[:, vars_selected], y, cv=cv, scoring='accuracy')
        mean_knn = scores_knn.mean()

        # QDA
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        scores_qda = cross_val_score(qda, X.iloc[:, vars_selected], y, cv=cv, scoring='accuracy')
        mean_qda = scores_qda.mean()

        # Se guardan los resultados
        avg_accuracy = (mean_knn + mean_qda) / 2

        results.append({
            'dependence_measure': measure_name,
            'n_features': n_features,
            'selected_features': vars_selected,
            'accuracy_knn': mean_knn,
            'accuracy_qda': mean_qda,
            'avg_accuracy': avg_accuracy
        })



results_df = pd.DataFrame(results)
ranking = results_df.sort_values(by='avg_accuracy', ascending=False)
ranking.to_csv("../timeseries/ranking_mrmr_repeated_kfold.csv", index=False)


print("\nRanking de número de variables según accuracy promedio:")
print(ranking[['dependence_measure', 'n_features', 'accuracy_knn', 'accuracy_qda', 'avg_accuracy']])