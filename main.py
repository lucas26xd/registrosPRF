import streamlit as st
from os.path import exists
from scrub import *
from predicoes import *
from introducao import *


# @st.cache
def import_bases():
    print('IMPORTAÇÃO DA BASE DE OCORRENCIAS')
    if exists('./Bases Limpas/dfOcorrencias.csv'):
        dfOcorrencias = pd.read_csv('./Bases Limpas/dfOcorrencias.csv')
    else:
        print('LIMPEZA DA BASE DE OCORRENCIAS')
        dfOcorrencias = scrub_ocorrencias()
        dfOcorrencias.to_csv('./Bases Limpas/dfOcorrencias.csv', index=False)

    print('IMPORTAÇÃO DA BASE DE PESSOAS')
    if exists('./Bases Limpas/dfPessoas.csv'):
        dfPessoas = pd.read_csv('./Bases Limpas/dfPessoas.csv')
    else:
        print('LIMPEZA DA BASE DE PESSOAS')
        dfPessoas = scrub_pessoas()
        dfPessoas.to_csv('./Bases Limpas/dfPessoas.csv', index=False)
    return dfOcorrencias, dfPessoas


st.title('Dados abertos sobre Acidentes da Polícia Rodoviária Federal a partir da aplicação do processo OSEMN')
st.sidebar.title('Sobre o trabalho:')
app_mode = st.sidebar.selectbox('Capítulo', ['Introdução', 'Desempenho dos regressores', 'Predições por Faixa Etária', 'Predições por Causas de Acidentes'])

dfOcorrencias, dfPessoas = import_bases()

if app_mode == 'Introdução':
    Introducao()
elif app_mode == 'Desempenho dos regressores':
    pass
elif app_mode == 'Predições por Faixa Etária':
    st.sidebar.title('Predições por mes/ano em cada faixa etária')
    st.write('### Legenda das Faixas Etárias:')
    st.write('- Criança: Pessoa com idade entre **0** e **12** anos')
    st.write('- Jovem: Pessoa com idade entre **12** e **24** anos')
    st.write('- Adulto: Pessoa com idade entre **24** e **60** anos')
    st.write('- Idoso: Pessoa com idade superior a **60** anos')

    pred = Predicoes(['Acidentes', 'Envolvidos', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos'])

    ppi = PredicoesPessoasIdade(pred, dfPessoas)
    df = ppi.predicts()

    # Pegando apenas as conlunas requisitadas
    df = df[(ppi.x + [pred.qtds])]

    # Pegando apenas as datas requisitadas
    df = pred.filter_dates(df)

    #Ordenando
    df = pred.sort(df)

    st.write(df)

    # Plotagem
    fig = pred.create_plot(df, 'Faixa Etária', 'Quantidades por Mês/Ano em cada Faixa Etária')
    st.write(fig)
elif app_mode == 'Predições por Causas de Acidentes':
    st.sidebar.title('Predições por Causas de Acidentes')

    pred = Predicoes(['Acidentes', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos', 'Feridos', 'Veículos'])

    poca = PredicoesOcorrenciasCausasAcidentes(pred, dfOcorrencias)
    df = poca.predicts()

    # Pegando apenas as conlunas requisitadas
    df = df[(poca.x + [pred.qtds])]

    # Pegando apenas as datas requisitadas
    df = pred.filter_dates(df)

    # Ordenando
    df = pred.sort(df)

    st.write(df)

    #Plotagem
    fig = pred.create_plot(df, 'Causas de Acidentes', 'Quantidades por Mês/Ano em cada Causa de Acidentes')
    fig.update_layout(width=1000, height=800)
    st.write(fig)
