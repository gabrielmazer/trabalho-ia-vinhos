# 🍷 Sistema Inteligente de Classificação de Vinhos

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![Status](https://img.shields.io/badge/Status-Concluído-success)

> Trabalho Final da disciplina de Inteligência Artificial - UNESP

## 📋 Sobre o Projeto

Este projeto consiste em um sistema de **Machine Learning** desenvolvido para classificar amostras de vinhos em três categorias distintas (cultivares), baseando-se em suas propriedades químicas.

Utilizamos o algoritmo **Random Forest** (Floresta Aleatória) devido à sua alta precisão e robustez para dados tabulares. O sistema foi implantado em nuvem utilizando o **Streamlit**, permitindo que qualquer usuário interaja com o modelo preditivo através de uma interface amigável.

### 🎯 Objetivo
Demonstrar a aplicação prática de técnicas de IA para resolução de problemas de classificação, cumprindo os requisitos de:
- Entrada de novos dados para predição.
- Visualização de dados.
- Deploy da aplicação online.

---

## 🚀 Acesse o Projeto Online

A aplicação está rodando em tempo real na nuvem. Clique no link abaixo para testar:

### [👉 Acessar Classificador de Vinhos](https://trabalho-ia-vinhos-gmazer-mshoda.streamlit.app/)

---

## 🛠 Tecnologias Utilizadas

* **Linguagem:** Python
* **Interface/Deploy:** [Streamlit](https://streamlit.io/)
* **Manipulação de Dados:** Pandas & NumPy
* **Machine Learning:** Scikit-learn (Random Forest Classifier)
* **Visualização:** Matplotlib & Seaborn
* **Dataset:** Wine Dataset (UCI Machine Learning Repository)

---

## 👥 Autores

| Aluno | Função |
|-------|--------|
| **Gabriel Mazer** | Desenvolvimento & Documentação |
| **Matheus Shoda** | Desenvolvimento & Análise de Dados |

---

## 📊 Como Rodar Localmente (Opcional)

Se você quiser rodar este projeto na sua própria máquina para testes, siga os passos abaixo no seu terminal:

1. Clone o repositório:
   ```bash
   git clone [https://github.com/gabrielmazer/trabalho-ia-vinhos.git](https://github.com/gabrielmazer/trabalho-ia-vinhos.git)

2. Entre na pasta do projeto:
   ```bash
   cd trabalho-ia-vinhos

3. Instale as dependências necessárias:
   ```bash
   pip install -r requirements.txt

4. Execute a aplicação:
   ```bash
   streamlit run app.py

Nota: Este projeto foi desenvolvido para fins acadêmicos em Novembro/2025.
