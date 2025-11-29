import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Configuração da Página
st.set_page_config(page_title="Classificação de Vinhos - IA", layout="wide")

# Títulos e Introdução
st.title("🍷 Sistema Inteligente de Classificação de Vinhos")
st.markdown("""
Este sistema utiliza um algoritmo de **Machine Learning (Random Forest)** para classificar vinhos em três categorias
baseado em suas características químicas.
*Trabalho Final de Inteligência Artificial - UNESP*
**Alunos:** Gabriel Mazer e Matheus Shoda
""")

# 1. Carregamento do Dataset
@st.cache_data
def load_data():
    wine = load_wine()
    df = pd.DataFrame(wine.data, columns=wine.feature_names)
    df['target'] = wine.target
    return df, wine.target_names

df, target_names = load_data()

# Sidebar para Inputs do Usuário
st.sidebar.header("Parâmetros do Vinho")
st.sidebar.markdown("Ajuste os valores para simular um novo vinho:")

def user_input_features():
    # Criando sliders para as principais features
    alcohol = st.sidebar.slider('Álcool', 11.0, 15.0, 13.0)
    malic_acid = st.sidebar.slider('Ácido Málico', 0.7, 6.0, 2.3)
    ash = st.sidebar.slider('Cinzas', 1.3, 3.3, 2.3)
    alcalinity = st.sidebar.slider('Alcalinidade das Cinzas', 10.0, 30.0, 19.0)
    magnesium = st.sidebar.slider('Magnésio', 70.0, 165.0, 100.0)
    color_intensity = st.sidebar.slider('Intensidade da Cor', 1.0, 13.0, 5.0)
    flavanoids = st.sidebar.slider('Flavanóides', 0.3, 5.1, 2.0)
    
    # Para simplificar a demo, vamos usar valores médios para as outras colunas menos impactantes
    data = {
        'alcohol': alcohol,
        'malic_acid': malic_acid,
        'ash': ash,
        'alcalinity_of_ash': alcalinity,
        'magnesium': magnesium,
        'total_phenols': df['total_phenols'].mean(),
        'flavanoids': flavanoids,
        'nonflavanoid_phenols': df['nonflavanoid_phenols'].mean(),
        'proanthocyanins': df['proanthocyanins'].mean(),
        'color_intensity': color_intensity,
        'hue': df['hue'].mean(),
        'od280/od315_of_diluted_wines': df['od280/od315_of_diluted_wines'].mean(),
        'proline': df['proline'].mean()
    }
    features = pd.DataFrame(data, index=[0])
    return features

df_user = user_input_features()

# 2. Divisão em Treino e Teste
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Modelo de Machine Learning
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predição do Usuário
prediction = clf.predict(df_user)
prediction_proba = clf.predict_proba(df_user)

# Layout de Colunas
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Visualização dos Dados")
    # Gráfico de dispersão simples
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='alcohol', y='color_intensity', hue='target', palette='viridis', ax=ax)
    plt.title("Relação: Álcool vs Intensidade da Cor")
    st.pyplot(fig)
    
    st.markdown("---")
    st.subheader("📈 Métricas de Desempenho")
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"**Acurácia do Modelo:** {acc:.2%}")
    st.info("A acurácia indica a porcentagem de vinhos que o modelo classificou corretamente no conjunto de teste.")

with col2:
    st.subheader("🔍 Resultado da Predição")
    st.markdown("Com base nos valores do menu lateral:")
    
    # Resultado formatado
    vinho_tipo = target_names[prediction[0]]
    st.success(f"Classificação: **{vinho_tipo.upper()}**")
    
    st.write("Probabilidade de cada classe:")
    st.bar_chart(prediction_proba[0])
    
    st.warning("Nota: O modelo usa Random Forest, uma técnica que cria múltiplas árvores de decisão para garantir precisão.")

# Rodapé obrigatório
st.markdown("---")
st.markdown("Desenvolvido para a disciplina de Inteligência Artificial - Novembro/2025")
