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


def calc_intervalo_meses_ano(df, months, years):
    months_ = months
    list_months = ['Hi!'] * len(years)
    cont = 0
    i = 0
    for y in years:
        months_ = months_[cont:]
        cont = 0
        for m in months_:
            cont += 1
            if m == 'December':
                break
        list_months[i] = months_[0:cont]
        i += 1
    
    i = 0
    df_ = df[(df['Ano'].isin([years[i]])) & (df['Mês'].isin(list_months[i]))]
    if len(years) > 1:
        for y in years[1:]:
            i += 1
            df_1 = df[(df['Ano'].isin([y])) & (df['Mês'].isin(list_months[i]))]
            frames = [df_, df_1]
            df_ = pd.concat(frames)
    
    return df_


st.title('Dados abertos sobre Acidentes da Polícia Rodoviária Federal a partir da aplicação do processo OSEMN')
st.sidebar.title('Sobre o trabalho:')
app_mode = st.sidebar.selectbox('Capítulo', ['Introdução', 'Desempenho dos regressores', 'Predições por Faixa Etária', 'Predições por Causas de Acidentes'])

dfOcorrencias, dfPessoas = import_bases()

if app_mode == 'Introdução':
    Introducao()
elif app_mode == 'Desempenho dos regressores':
    pass
elif app_mode == 'Predições Idade':
    st.sidebar.title('Predições por mes/ano em cada faixa etária')
    st.write('### Legenda das Faixas Etárias:')
    st.write('- Criança: Pessoa com idade entre **0** e **12** anos')
    st.write('- Jovem: Pessoa com idade entre **12** e **24** anos')
    st.write('- Adulto: Pessoa com idade entre **24** e **60** anos')
    st.write('- Idoso: Pessoa com idade superior a **60** anos')

    range_dates = [f'{mes + 1:0>2}/{2017 + ano}' for ano in range(5) for mes in range(12)]
    start_date, end_date = st.sidebar.select_slider('Intervalo de datas', options=range_dates, value=('01/2020', '03/2021'))

    options_qtd = ['Acidentes', 'Envolvidos', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos']
    qtds = st.sidebar.selectbox('Quantidade', options=options_qtd, index=1)

    ppi = PredicoesPessoasIdade(dfPessoas)
    df_predicts, df = ppi.predicts()

    # Pegando apenas as conlunas requisitadas
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

    # df = df[(df['Ano'].isin(years)) & (df['Mês'].isin(months))]
    df = calc_intervalo_meses_ano(df, months, years)

    #Ordenando
    sorterIndex = dict(zip(ppi.month_names, range(len(ppi.month_names))))
    df['mes_rank'] = df['Mês'].map(sorterIndex)
    df = df.sort_values(['Ano', 'mes_rank'])
    df = df.drop(columns=['mes_rank'])

    st.write(df)

    # Plotagem
    fig = ppi.create_plot(df, qtds, 'Quantidades por Mês/Ano em cada Faixa Etária')
    # fig.update_layout(width=1000, height=800)
    st.write(fig)
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
