import pandas as pd
from pickle import load

# 1. CARREGAR OS MODELOS E METADADOS SALVOS NO TREINAMENTO
with open('metadados_breast_cancer.model', 'rb') as f:
    metadados = load(f)

with open('melhor_modelo_breast_cancer.model', 'rb') as f:
    modelo_classificador = load(f)

colunas_originais = metadados['colunas_originais_atributos']
colunas_num = metadados['colunas_num']
colunas_cat = metadados['colunas_cat']
colunas_finais_modelo = metadados['colunas_finais_modelo']

# 2. RECEBER OS DADOS DE UMA NOVA PACIENTE
dados_nova_paciente = [metadados['exemplo_paciente']]
nova_paciente_df = pd.DataFrame(dados_nova_paciente, columns=colunas_originais)

print("--- DADOS DA NOVA PACIENTE ---")
print(nova_paciente_df.to_string(index=False))

# 3. NORMALIZAR OS DADOS DA NOVA PACIENTE
if len(colunas_num) > 0:
    with open('normalizador_breast_cancer.model', 'rb') as f:
        normalizador = load(f)
    nova_num_norm = pd.DataFrame(
        normalizador.transform(nova_paciente_df[colunas_num]),
        columns=colunas_num
    )
else:
    nova_num_norm = pd.DataFrame(index=nova_paciente_df.index)

if len(colunas_cat) > 0:
    nova_cat_norm = pd.get_dummies(
        nova_paciente_df[colunas_cat].astype(str),
        prefix=colunas_cat,
        dtype=int
    )
else:
    nova_cat_norm = pd.DataFrame(index=nova_paciente_df.index)

nova_paciente_normalizada = nova_num_norm.join(nova_cat_norm)
nova_paciente_normalizada = nova_paciente_normalizada.reindex(
    columns=colunas_finais_modelo, fill_value=0
)

# 4. DETERMINAR O INDICATIVO DE DIAGNÓSTICO (ATRIBUTO Class)
diagnostico_predito = modelo_classificador.predict(nova_paciente_normalizada)[0]

print("\n--- RESULTADO DA INFERÊNCIA ---")
print(f"Modelo utilizado: {metadados['nome_melhor_modelo']}")
print(f"Indicativo de diagnóstico (Atributo Class): {diagnostico_predito}")

if hasattr(modelo_classificador, 'predict_proba'):
    probabilidades = modelo_classificador.predict_proba(nova_paciente_normalizada)[0]
    confianca = max(probabilidades) * 100
    print(f"Confiança da predição: {confianca:.2f}%")