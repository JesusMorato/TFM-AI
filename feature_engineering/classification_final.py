import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_regression as MI
from sklearn.feature_selection import SelectFromModel
from boruta import BorutaPy
from skfda.preprocessing.dim_reduction.variable_selection import MinimumRedundancyMaximumRelevance
from skfda.representation.grid import FDataGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone

def mutualInformation(X,y):
    return MI(X, y.ravel())[0]

X = pd.read_csv("../data_processed/X_processed_multiclass.csv")
y = pd.read_csv("../data_processed/y_processed_multiclass.csv").values.ravel()

# Validación cruzada
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=1, random_state=42)

# Configuración de métodos de selección de características
feature_selectors = {
    "mRMR (Mutual Info, 45 vars)": MinimumRedundancyMaximumRelevance(n_features_to_select=45, dependence_measure=mutualInformation, criterion='quotient'),
    "SelectKBest (Mutual Info, 55 vars)": SelectKBest(score_func=mutual_info_classif, k=55),
    "Lasso (C=0.001)": SelectFromModel(LogisticRegression(penalty='l1', C=0.001, solver='liblinear', multi_class='ovr', random_state=42))   
}

results = []

# Primero evaluamos Boruta ya que no utiliza medidas de dependencia y su configuración es diferente
print("\nEvaluando método: Boruta")

knn_scores = []
qda_scores = []

for train_idx, test_idx in cv.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Creamos nueva instancia de Boruta en cada fold
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced', random_state=42)
    boruta_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=42)


    boruta_selector.fit(X_train.values, y_train)
    selected_indices = np.where(boruta_selector.support_)[0]  # Obtener características seleccionadas


    X_train_selected = X_train.iloc[:, selected_indices]
    X_test_selected = X_test.iloc[:, selected_indices]

    # KNN y QDA
    knn = KNeighborsClassifier()
    qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

    # Evaluación en cada fold
    knn.fit(X_train_selected, y_train)
    qda.fit(X_train_selected, y_train)

    knn_scores.append(knn.score(X_test_selected, y_test))
    qda_scores.append(qda.score(X_test_selected, y_test))

# Obtenemos la media y desviación estándar de las puntuaciones
mean_knn, std_knn = np.mean(knn_scores), np.std(knn_scores)
mean_qda, std_qda = np.mean(qda_scores), np.std(qda_scores)


results.append({
    "Method": "Boruta",
    "Accuracy KNN (Mean)": mean_knn,
    "Accuracy KNN (Std)": std_knn,
    "Accuracy QDA (Mean)": mean_qda,
    "Accuracy QDA (Std)": std_qda
})


# Se evalúan el resto de métodos con las mejores configuraciones obtenidas de los scripts de ranking
for method_name, selector in feature_selectors.items():
    print(f"\nEvaluando método: {method_name}")

    knn_scores = []
    qda_scores = []

    # Aplicamos validación cruzada
    for train_idx, test_idx in cv.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Aplicar selección de características en cada fold
        selector_clone = clone(selector)  # Clonar el selector para evitar sobreescritura
        
        if method_name.startswith("mRMR"):
            X_train_fda = FDataGrid(X_train.values.reshape(X_train.shape[0], X_train.shape[1]))
            selector_clone.fit(X_train_fda, y_train)
            selected_indices = selector_clone.get_support()

        else:
            selector_clone.fit(X_train, y_train)
            selected_indices = selector_clone.get_support(indices=True)

        X_train_selected = X_train.iloc[:, selected_indices]
        X_test_selected = X_test.iloc[:, selected_indices]

        # KNN y QDA
        knn = KNeighborsClassifier()
        qda = QuadraticDiscriminantAnalysis(reg_param=0.1)

        # Evaluación en cada fold
        knn.fit(X_train_selected, y_train)
        qda.fit(X_train_selected, y_train)

        knn_scores.append(knn.score(X_test_selected, y_test))
        qda_scores.append(qda.score(X_test_selected, y_test))

    # Calcular media y desviación estándar
    mean_knn, std_knn = np.mean(knn_scores), np.std(knn_scores)
    mean_qda, std_qda = np.mean(qda_scores), np.std(qda_scores)

    # Guardar resultados
    results.append({
        "Method": method_name,
        "Accuracy KNN (Mean)": mean_knn,
        "Accuracy KNN (Std)": std_knn,
        "Accuracy QDA (Mean)": mean_qda,
        "Accuracy QDA (Std)": std_qda
    })

# Obtenemos la media y desviación estándar de las puntuaciones
results_df = pd.DataFrame(results)

# Guardar y mostrar resultados
results_df.to_csv("../comparison_results_2.csv", index=False)

print("\nTabla de Resultados:")
print(results_df)
