import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from pickle import dump

# 1. CARREGAR E LIMPAR OS DADOS
# Detecta automaticamente separador (',' ou ';') e trata '?' como valor ausente (padrão UCI)
dados = pd.read_csv('breast-cancer.csv', sep=None, engine='python', na_values=['?', ' ?'])
dados = dados.dropna().reset_index(drop=True)

coluna_alvo = 'Class' if 'Class' in dados.columns else 'class'
dados_atributos = dados.drop(columns=[coluna_alvo])
dados_classes = dados[coluna_alvo]

# 2. NORMALIZAR OS DADOS (QUANTITATIVOS E CATEGÓRICOS)
colunas_num = dados_atributos.select_dtypes(include=['number']).columns.tolist()
colunas_cat = dados_atributos.select_dtypes(exclude=['number']).columns.tolist()

# 2.1 Normalizar atributos quantitativos (MinMaxScaler)
if len(colunas_num) > 0:
    normalizador = MinMaxScaler()
    modelo_normalizador = normalizador.fit(dados_atributos[colunas_num])
    with open('normalizador_breast_cancer.model', 'wb') as f:
        dump(modelo_normalizador, f)
    dados_num_norm = pd.DataFrame(
        modelo_normalizador.transform(dados_atributos[colunas_num]),
        columns=colunas_num
    )
else:
    dados_num_norm = pd.DataFrame(index=dados_atributos.index)

# 2.2 Normalizar atributos categóricos (get_dummies)
if len(colunas_cat) > 0:
    dados_cat_norm = pd.get_dummies(
        dados_atributos[colunas_cat].astype(str),
        prefix=colunas_cat,
        dummy_na=False,
        dtype=int
    )
else:
    dados_cat_norm = pd.DataFrame(index=dados_atributos.index)

dados_atributos_normalizados = dados_num_norm.join(dados_cat_norm)

# 3. SEPARAR TREINO E TESTE (80% TREINO / 20% TESTE ESTRATIFICADO)
X_train, X_test, y_train, y_test = train_test_split(
    dados_atributos_normalizados,
    dados_classes,
    test_size=0.2,
    random_state=42,
    stratify=dados_classes
)

# 4. BALANCEAR OS DADOS DE TREINO (SMOTE)
resampler = SMOTE(random_state=42)
X_train_bal, y_train_bal = resampler.fit_resample(X_train, y_train)

# 5. COMPARAÇÃO DE CLASSIFICADORES E HIPERPARAMETRIZAÇÃO (RandomizedSearchCV)
modelos = {
    'Decision Tree': (
        DecisionTreeClassifier(random_state=42),
        {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'splitter': ['best', 'random'],
            'min_samples_split': [int(x) for x in np.linspace(start=2, stop=10, num=5)],
            'max_depth': [None, 3, 5, 8, 10, 15]
        }
    ),
    'Random Forest': (
        RandomForestClassifier(random_state=42),
        {
            'n_estimators': [50, 100, 150, 200],
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 5, 10, 15],
            'min_samples_split': [2, 4, 6, 8]
        }
    ),
    'KNN': (
        KNeighborsClassifier(),
        {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
    ),
    'SVM': (
        SVC(random_state=42, probability=True),
        {
            'C': [0.1, 1, 5, 10],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto']
        }
    ),
    'Logistic Regression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            'C': [0.01, 0.1, 1, 10, 100]
        }
    )
}

resultados = []
melhor_nome = None
melhor_modelo = None
melhor_acuracia_teste = -1.0
melhor_acuracia_cv = -1.0
melhores_params = None

print("=== COMPARAÇÃO DE CLASSIFICADORES (HIPERPARAMETRIZAÇÃO) ===\n")

for nome, (estimador, grade) in modelos.items():
    total_combinacoes = int(np.prod([len(v) for v in grade.values()]))
    n_iter = min(20, total_combinacoes)

    busca = RandomizedSearchCV(
        estimator=estimador,
        param_distributions=grade,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        random_state=42,
        n_jobs=-1
    )
    busca.fit(X_train_bal, y_train_bal)

    modelo_otimizado = busca.best_estimator_
    acuracia_cv = busca.best_score_
    predicoes_teste = modelo_otimizado.predict(X_test)
    acuracia_teste = accuracy_score(y_test, predicoes_teste)

    resultados.append({
        'Modelo': nome,
        'Acuracia_CV (%)': round(acuracia_cv * 100, 2),
        'Acuracia_Teste (%)': round(acuracia_teste * 100, 2)
    })

    # Seleciona o modelo com maior acurácia no teste (desempate por maior CV)
    if (acuracia_teste > melhor_acuracia_teste) or (
        acuracia_teste == melhor_acuracia_teste and acuracia_cv > melhor_acuracia_cv
    ):
        melhor_acuracia_teste = acuracia_teste
        melhor_acuracia_cv = acuracia_cv
        melhor_nome = nome
        melhor_modelo = modelo_otimizado
        melhores_params = busca.best_params_

tabela_resultados = pd.DataFrame(resultados).sort_values(
    by=['Acuracia_Teste (%)', 'Acuracia_CV (%)'], ascending=False
)
print(tabela_resultados.to_string(index=False))

# 6. AVALIAÇÃO DETALHADA E SALVAMENTO DO MODELO VENCEDOR
print(f"\n>>> MELHOR MODELO SELECIONADO: {melhor_nome}")
print(f"Melhores hiperparâmetros: {melhores_params}")
print(f"Acurácia média na Validação Cruzada (CV): {melhor_acuracia_cv*100:.2f}%")
print(f"Acurácia no Conjunto de Teste:            {melhor_acuracia_teste*100:.2f}%")

pred_vencedor = melhor_modelo.predict(X_test)
print("\n--- MATRIZ DE CONFUSÃO (CONJUNTO DE TESTE) ---")
print(confusion_matrix(y_test, pred_vencedor))

print("\n--- RELATÓRIO DE CLASSIFICAÇÃO (CONJUNTO DE TESTE) ---")
print(classification_report(y_test, pred_vencedor))

with open('melhor_modelo_breast_cancer.model', 'wb') as f:
    dump(melhor_modelo, f)

metadados = {
    'nome_melhor_modelo': melhor_nome,
    'colunas_originais_atributos': dados_atributos.columns.tolist(),
    'colunas_num': colunas_num,
    'colunas_cat': colunas_cat,
    'colunas_finais_modelo': dados_atributos_normalizados.columns.tolist(),
    'exemplo_paciente': dados_atributos.iloc[0].to_dict()
}
with open('metadados_breast_cancer.model', 'wb') as f:
    dump(metadados, f)

print("Modelo vencedor salvo com sucesso como 'melhor_modelo_breast_cancer.model'!")