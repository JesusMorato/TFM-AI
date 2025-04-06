import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

lasso_penalties = [0.0001, 0.001, 0.01, 0.1]  # Valores de C (inverso de lambda)
results = []

# Carga de datos
X = pd.read_csv("../data_processed/X_processed_multiclass.csv")
y = pd.read_csv("../data_processed/y_processed_multiclass.csv")
y = np.squeeze(y)

cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)

# Se itera sobre los valores de C para Lasso
print("---- Evaluando Lasso ----")
for c_value in lasso_penalties:
    print(f"\n---- Evaluando Lasso con C = {c_value} ----")

    # Lasso
    lasso = LogisticRegression(penalty='l1', C=c_value, solver='liblinear', multi_class='ovr', random_state=42)
    selector = SelectFromModel(lasso, prefit=False)  # `prefit=False` para ajustarlo dentro de `fit`
    X_selected = selector.fit_transform(X, y)
    selected_indices = selector.get_support(indices=True)

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
        'lasso_penalty_C': c_value,
        'n_features_selected': len(selected_indices),
        'selected_features': selected_indices,
        'accuracy_knn': mean_knn,
        'accuracy_qda': mean_qda,
        'avg_accuracy': avg_accuracy
    })


results_df = pd.DataFrame(results)
ranking = results_df.sort_values(by='avg_accuracy', ascending=False)
ranking.to_csv("../timeseries/ranking_lasso_repeated_kfold.csv", index=False)


print("\nRanking de número de variables según accuracy promedio:")
print(ranking[['lasso_penalty_C', 'n_features_selected', 'accuracy_knn', 'accuracy_qda', 'avg_accuracy']])
