import pandas as pd
import streamlit as st
from os.path import exists
from scrub import *
from model import *
from predicoes import *


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
app_mode = st.sidebar.selectbox('Capítulo', ['Predições Idade', 'Introdução', 'Titulo 1', 'Titulo 2', 'Titulo 3', 'Titulo 4'])

dfOcorrencias, dfPessoas = import_bases()

if app_mode == 'Introdução':
    st.markdown('# **Introdução**')
    st.write('''Olá mundo!''')

    H3P = Hipotese3Pessoas(dfPessoas)
    st.write(H3P.regressoes())
elif app_mode == 'Titulo 1':
    st.header('Análise Estatística')
elif app_mode == 'Predições Idade':
    st.sidebar.title('Predições por mes/ano em cada faixa etária')
    st.write('### Legenda das Faixas Etárias:')
    st.write('- Criança: Pessoa com idade entre **0** e **12** anos')
    st.write('- Jovem: Pessoa com idade entre **12** e **24** anos')
    st.write('- Adulto: Pessoa com idade entre **24** e **60** anos')
    st.write('- Idoso: Pessoa com idade superior a **60** anos')

    range_dates = [f'{mes + 1:0>2}/{2017 + ano}' for ano in range(5) for mes in range(12)]
    start_date, end_date = st.sidebar.select_slider('Intervalo de datas', options=range_dates, value=('01/2020', '12/2020'))

    options_qtd = ['Acidentes', 'Envolvidos', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos']
    qtds = st.sidebar.multiselect('Quantidade', options=options_qtd, default=['Envolvidos'])

    ppi = PredicoesPessoasIdade(dfPessoas)
    df_predicts, df = ppi.predicts()

    # Pegando apenas as conlunas requisitadas
    df = df[(ppi.x + qtds)]

    # Pegando apenas as datas requisitadas
    start_year = int(start_date[3:])
    start_month = int(start_date[:2])
    end_year = int(end_date[3:])
    end_month = int(end_date[:2])

    years = list()
    for y in range(end_year - start_year + 1):
        years.append(f'{start_year + y}')

    months = list()
    qtd_months = (len(years) * 12) - (start_month - 1) - (12 - end_month)
    for m in range(qtd_months):
        months.append(f'{ppi.month_names[(start_month + m - 1) % 12]}')

    df = df[(df['Ano'].isin(years)) & (df['Mês'].isin(months))]
    #ISSUE: Pegar apenas o conjunto de meses correspondentes de cada ano e não apenas o ISIN no DF

    #Ordenando
    sorterIndex = dict(zip(ppi.month_names, range(len(ppi.month_names))))
    df['mes_rank'] = df['Mês'].map(sorterIndex)
    df = df.sort_values(['Ano', 'mes_rank'])
    df = df.drop(columns=['mes_rank'])

    st.write(df)

    # Plotagem
    # ISSUE: Utilizar também os qtds especificados não só 'Envolvidos'
    st.write(ppi.create_plot(df, 'Envolvidos', 'Quantidades por Mês/Ano em cada Faixa Etária'))
