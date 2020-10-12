import pandas as pd
import streamlit as st
from os.path import exists
from scrub import *
from model import *

dfOcorrencias = None
dfPessoas = None


def import_bases():
    global dfOcorrencias, dfPessoas
    print('IMPORTAÇÃO DA BASE DE OCORRENCIAS')
    if exists('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfOcorrencias.csv'):
        dfOcorrencias = pd.read_csv('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfOcorrencias.csv')
    else:
        print('LIMPEZA DA BASE DE OCORRENCIAS')
        dfOcorrencias = scrub_ocorrencias()
        dfOcorrencias.to_csv('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfOcorrencias.csv', index=False)

    print('IMPORTAÇÃO DA BASE DE PESSOAS')
    if exists('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfPessoas.csv'):
        dfPessoas = pd.read_csv('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfPessoas.csv')
    else:
        print('LIMPEZA DA BASE DE PESSOAS')
        dfPessoas = scrub_pessoas()
        dfPessoas.to_csv('D:/Users/Lucas Santos/Desktop/Repositórios/registrosPRF/Bases Limpas/dfPessoas.csv', index=False)


st.title('Dados abertos sobre Acidentes da Polícia Rodoviária Federal a partir da aplicação do processo OSEMN')
st.sidebar.title('Sobre o trabalho:')
app_mode = st.sidebar.selectbox('Qual opção deseja?', ['Introdução', 'Titulo 1', 'Titulo 2', 'Titulo 3', 'Titulo 4'])

if app_mode == 'Introdução':
    st.markdown('# **Introdução**')
    st.write('''Olá mundo!''')

    import_bases()

    H3P = Hipotese3Pessoas(dfPessoas)
    st.write(H3P.regressoes())
elif app_mode == 'Titulo 1':
    st.header('Análise Estatística')
