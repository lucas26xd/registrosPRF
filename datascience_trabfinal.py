import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image


st.title("Dados abertos sobre Acidentes da Polícia Rodoviária Federal a partir da aplicação do processo OSEMN")
st.sidebar.title("Sobre o trabalho:")
app_mode = st.sidebar.selectbox("Qual opção deseja?",["Introdução","Titulo 1","Titulo 2 ", "Titulo 3", "Titulo 4"])

if app_mode == "Introdução":

    #add imagem no "cabeçalho"
    image = Image.open('cabe.jpg')
    st.image(image, use_column_width=True)
    
    st.markdown(' **_Qual meu problema?_**')
    st.write("""xxxxxxxx""")

elif app_mode == "Titulo 1":
    st.header("Análise Estatística")
    
    #Lendo base de dados
    def load_data (nrows): 
        dados = pd.read_excel('passagens_spia_s25_jan_jun_2019.xlsx', nrows = nrows ) 
        return dados
    st.text('Carregando a base de dados ...') 
    data = load_data(31280) 
    st.write(data)
    st.write("Escolha uma das opções a respeito do dataset.")
    #opçoes de escolha do usuario
    options = ['Descrição dos Dados', 'Informações sobre Velocidade', 'Informações sobre Sensores','Dados Nulos']
    #exibi a mensagem desejada acima do selectbox e as opções definidas acima
    distOpt = st.selectbox('Qual distribuição deseja?', options)

    if distOpt == 'Descrição dos Dados':
        x = data.describe()
        st.write(x)
    elif distOpt == 'Informações sobre tal coisa':
        plt.hist(data['velocidade'], bins=50)
        st.pyplot()

    elif distOpt == 'Informações sobre Sensores':
        datamaior = data[(data["velocidade"] > 10)]
        st.write('Número de sensores com velocidade maior que 10:', len( datamaior["geos25"].unique() ))

    else:
        st.write(data.isnull().sum(), 'Quantidade de observações NULL existentes no dataset por atributo')

    #st.bar_chart(chart_data)

elif app_mode == "Titulo 2":
    st.header("Saúde dos Sensores")
    st.markdown("""texto""")
    #Sensores com 1 registro
    st.markdown('**Sensores com apenas 1 registro, assim considerados "doentes"**')
    map_data_sick = pd.DataFrame({
    'latitude' : [-4.577809,-3.755487, -3.771947, -4.27414 ,-3.73194, -5.948694, -7.13527, -3.692833,
    -7.294109, -3.776468, -6.105134, -6.399444, -3.82662, -3.73342, -4.167702, -7.634639,-12.91,
    -6.326777,-7.10122, -6.775611,-15.515278,-7.08251,-5.751425,-3.466361,-7.316369,-7.235413,
    -7.10524,-7.213732,-4.948527,-5.909567,-4.62007,-5.73981,-3.742038,-4.121139,-3.817403,-4.928326,-7.332778,
    -7.180356,-3.766036,-3.530056,-6.068566, -4.501139,-5.09561,-3.806264, -4.121778, -6.374889,-5.579469,-3.761703,
    -7.234636,-7.14599,-7.20775,-3.881327,-5.96023,-6.080622,-6.93703,-7.12565,-3.61875,-7.093778,-6.776638],
    'longitude': [-38.365423, -38.526142, -38.534878,  -38.726830, -38.580476, -39.524361, -34.8826 , -40.975667,
    -39.037187, -38.616611, -49.868306, -39.168083, -38.472544, -38.592519, -38.447957, -39.263778,-38.458361,-39.321,
    -34.86139,-39.292361,-41.238139,-34.83333,-39.645275,-40.39525,-38.780111,-35.879475,-34.83577, -35.922895,
    -37.989267,-40.024019,-37.65019,-39.62175,-38.513683,-38.660361,-38.600794,-37.96346,-39.149028,-39.309897,
    -38.599459,-40.410056, -49.903386, -37.800389,-38.07436,-38.508375,-38.660583,-39.265127,-39.584792,-38.515535,
    -35.917579,-34.87483,-35.875543,-38.419264,-39.082338,-49.883345,-39.56503,-34.83903,-40.420639,-39.730028,-39.292638]})
    st.map(map_data_sick)

    

elif app_mode == "Titulo 3":	
    st.header("Divisão por Lote de 24h")



elif app_mode == "Titulo 4":
    st.write("Para Técnicas de Predição, acesse: ")

	
#st.line_chart(chart_data)
#st.area_chart(chart_data)
#st.bar_chart(chart_data)





