# Avaliação Semestral - Data Science (Classificador Breast Cancer)

**Aluno:** Matheus Eduardo Ducati  
**RA:** 47614137  
**Disciplina:** Data Science  

## Estrutura do Repositório
* `treinamento.py`: Limpeza de valores ausentes (`?`), normalização quantitativa (`MinMaxScaler`) e categórica (`get_dummies`), divisão treino/teste (80/20), balanceamento com `SMOTE`, comparação de 5 classificadores com `RandomizedSearchCV` (Decision Tree, Random Forest, KNN, SVM e Logistic Regression) e salvamento do modelo de maior acurácia (`Logistic Regression` — 71,43% no teste / 76,38% em CV).
* `inferencia.py`: Carregamento do modelo vencedor, normalização dos dados de uma nova paciente e predição do atributo `Class` com grau de confiança.
* `breast-cancer.csv`: Base de dados Breast Cancer (UCI).
* `*.model`: Modelo vencedor, normalizador e metadados persistidos com `pickle`.
