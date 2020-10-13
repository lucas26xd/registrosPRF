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

def calc_intervalo_meses_ano(df, months, years):
    months_ = months
    list_months = ['Hi!'] * len(years)
    cont = 0
    i = 0
    for y in years: 
        #st.write(y)
        months_ = months_[cont:]
        cont = 0
        #st.write("months_")
        #st.write(months_)    
        for m in months_: 
            #st.write(m)
            cont= cont+1
            if m == 'December':
                break
        list_months[i] = months_[0:cont]
        i = i+1
        #st.write("list_months")
        #st.write(list_months)    
    
    i = 0
    df_ = df[(df['Ano'].isin([years[i]])) & (df['Mês'].isin(list_months[i]))]
    #st.write(years[i])
    #st.write(df_)
    if len(years)>1:
        for y in years[1:]:
            i = i+1
            df_1 = df[(df['Ano'].isin([y])) & (df['Mês'].isin(list_months[i]))]
            #st.write(y)
            #st.write(df_1)
            frames = [df_, df_1]
            df_ = pd.concat(frames)
    
    return df_


st.title('Dados abertos sobre Acidentes da Polícia Rodoviária Federal a partir da aplicação do processo OSEMN')
st.sidebar.title('Sobre o trabalho:')
app_mode = st.sidebar.selectbox('Capítulo', ['Predições Idade', 'Introdução', 'Titulo 1', 'Predições por Causas de Acidentes', 'Titulo 3', 'Titulo 4'])

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


elif app_mode == 'Predições por Causas de Acidentes':
    st.sidebar.title('Predições por Causas de Acidentes')

    range_dates = [f'{mes + 1:0>2}/{2017 + ano}' for ano in range(5) for mes in range(12)]
    start_date, end_date = st.sidebar.select_slider('Intervalo de datas', options=range_dates, value=('01/2020', '12/2020'))

    options_qtd = ['Acidentes', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos', 'Feridos', 'Veículos']
    #qtds = st.sidebar.multiselect('Quantidade', options=options_qtd, default=['Acidentes'])
    qtds = st.sidebar.selectbox('Quantidade', options_qtd)

    ppi = PredicoesOcorrenciasCausasAcidentes(dfOcorrencias)
    df_predicts, df = ppi.predicts()

    # Pegando apenas as conlunas requisitadas
    #df = df[(ppi.x + qtds)]
    df = df[(ppi.x + [qtds])]

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
  
    df = calc_intervalo_meses_ano(df, months, years)
    
    #Ordenando
    sorterIndex = dict(zip(ppi.month_names, range(len(ppi.month_names))))
    df['mes_rank'] = df['Mês'].map(sorterIndex)
    df = df.sort_values(['Ano', 'mes_rank'])
    df = df.drop(columns=['mes_rank'])

    st.write(df)
    
    fig = ppi.create_plot(df, qtds, 'Quantidades por Mês/Ano em cada Causa de Acidentes')
    fig.update_layout(width=1000, height=800)

    #st.plotly_chart(fig,width = 1200, height=800)
    st.write(fig)
 
    #st.write(ppi.create_plot(df, qtds, 'Quantidades por Mês/Ano em cada Causa de Acidentes'))