import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from scipy.stats import pearsonr
from dcor import u_distance_covariance_sqr, u_distance_correlation_sqr

# Definimos las funciones de dependencia
def pearson_dependence(X, y):
    return np.array([np.abs(pearsonr(X[:, i], y)[0]) for i in range(X.shape[1])])

# Número de características a seleccionar por el método
feature_counts = [25, 30, 35, 40, 45, 50, 55, 60, 66]
dependence_measures = {
    'pearson_correlation': pearson_dependence,
    'mutual_information': mutual_info_classif,
    'f_statistic': f_classif
}
results = []

# Cargar datos
X = pd.read_csv("../data_processed/X_processed_multiclass.csv")
y = pd.read_csv("../data_processed/y_processed_multiclass.csv")
y = np.squeeze(y)

# Se itera sobre las medidas de dependencia
for measure_name, measure_func in dependence_measures.items():
    print(f"\n---- Evaluando Dependence Measure: {measure_name} ----")

    for n_features in feature_counts:
        print(f"Número de características: {n_features}")

        # Uso del método SelectKBest con la medida de dependencia
        selector = SelectKBest(score_func=measure_func, k=n_features)
        X_selected = selector.fit_transform(X, y)
        selected_indices = selector.get_support(indices=True)

        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

        # KNN
        knn = KNeighborsClassifier()
        scores_knn = cross_val_score(knn, X.iloc[:, selected_indices], y, cv=cv, scoring='accuracy')
        mean_knn = scores_knn.mean()

        # QDA
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)
        scores_qda = cross_val_score(qda, X.iloc[:, selected_indices], y, cv=cv, scoring='accuracy')
        mean_qda = scores_qda.mean()

        avg_accuracy = (mean_knn + mean_qda) / 2

        # Guardamos resultados
        results.append({
            'dependence_measure': measure_name,
            'n_features': n_features,
            'selected_features': selected_indices,
            'accuracy_knn': mean_knn,
            'accuracy_qda': mean_qda,
            'avg_accuracy': avg_accuracy
        })


results_df = pd.DataFrame(results)
ranking = results_df.sort_values(by='avg_accuracy', ascending=False)
ranking.to_csv("../timeseries/ranking_selectkbest_repeated_kfold.csv", index=False)


print("\nRanking de número de variables según accuracy promedio:")
print(ranking[['dependence_measure', 'n_features', 'accuracy_knn', 'accuracy_qda', 'avg_accuracy']])
