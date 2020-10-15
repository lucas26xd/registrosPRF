import pandas as pd
import streamlit as st
from PIL import Image


class Introducao():
    def __init__(self):
        self.image = Image.open('./Imagens/logo.png')
        self.__cabecalho()
        self.__agg_ocorrencias()
        self.__agg_pessoas()
        self.__rodape()

    def __cabecalho(self):
        st.image(self.image, use_column_width=True)
        st.markdown(' ')
        st.markdown('## **Introdução**')
        st.markdown('A Polícia Rodoviária Federal disponibiliza em seu site os dados gerais de acidentes nas rodovias '
                    'federais brasileiras, como segue neste [link](https://portal.prf.gov.br/dados-abertos-acidentes).')
        st.markdown('''Segundo a **Agência CNT Transporte Atual** : "Os números mostram queda de **2,6%** nas ocorrências 
        quando comparado os anos de **2018** e **2019**. Já os acidentes com vítimas (mortos e feridos), por sua vez, 
        tiveram elevação de **3,3%**, subindo de **53.963**, em **2018**, para **55.756**. Foram **2.526** feridos a 
        mais em **2019**. Além disso, em **2019**, o número de mortes cresceu **1,2%**, passando para **5.332** (**63** 
        óbitos a mais que em **2018**). Foi o primeiro aumento em sete anos. De **2012** a **2018**, as mortes nas 
        rodovias federais tiveram queda de **39,2%**, com sucessivas reduções a cada ano".''')
        st.markdown('Baseados nesse estudo e nos dados disponibilizados no site, como mostrado anteriormente, o '
                    'trabalho a seguir faz a análise estatística sobre as bases de dados: **Agrupados por ocorrência** '
                    'e **Agrupados por pessoas**, entre os anos de **2017** e **2020**. Em seguida mostraremos também '
                    'algumas possíveis hipóteses futuras e sua aplicação em alguns regressores.')

    def __agg_ocorrencias(self):
        st.markdown('## **Agrupados por Ocorrências**')
        options = st.selectbox('Selecione a opção desejada a cerca das informações sobre a Base Ocorrências.',
                               ['', 'Atributos de Ocorrências', 'Quantidade de Registros de Ocorrências'])

        if options == 'Atributos de Ocorrências':
            table_dfOcorrencias = pd.DataFrame({
                'Atributos': ['id', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km', 'municipio',
                              'causa_acidente', 'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'sentido_via',
                              'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'ano', 'pessoas',
                              'mortos', 'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos',
                              'latitude', 'longitude', 'regional', 'delegacia', 'uop'],
                'Status': (['Não apagado'] * 11 + ['Apagado'] * 2 + ['Não apagado'] * 3 + ['Apagado'] +
                           ['Não apagado'] * 9 + ['Apagado'] * 5)})
            st.write(table_dfOcorrencias)
        elif options == 'Quantidade de Registros de Ocorrências':
            qtd_dfOcorrencias = pd.DataFrame({
                'Ano': ['2017', '2018', '2019', '2020'],
                'Quantidade de registros': ['89563', '69295', '67446', '34084']})
            st.write(qtd_dfOcorrencias)

    def __agg_pessoas(self):
        st.markdown('## **Agrupados por Pessoas**')
        optionPessoas = st.selectbox('Selecione a opção desejada a cerca das informações sobre a Base Pessoas.',
                                     ['', 'Atributos de Pessoas', 'Quantidade de Registros de Pessoas'])

        if optionPessoas == 'Atributos de Pessoas':
            table_dfPessoas = pd.DataFrame({
                'Atributos': ['id', 'pesid', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km', 'municipio',
                              'causa_acidente', 'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'sentido_via',
                              'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'id_veiculo',
                              'tipo_veiculo', 'marca', 'ano_fabricacao_veiculo', 'tipo_envolvido', 'estado_fisico',
                              'idade', 'sexo', 'ilesos', 'feridos leves', 'feridos graves', 'mortos', 'latitude',
                              'longitude', 'regional', 'delegacia', 'uop'],
                'Status': (['Não apagado'] * 12 + ['Apagado'] * 2 + ['Não apagado'] * 3 + ['Apagado'] * 2 +
                           ['Não apagado'] + ['Apagado'] * 2 + ['Não apagado'] * 8 + ['Apagado'] * 5)})
            st.write(table_dfPessoas)
        elif optionPessoas == 'Quantidade de Registros de Pessoas':
            qtd_dfPessoas = pd.DataFrame({
                'Ano': ['2017', '2018', '2019', '2020'],
                'Quantidade de registros': ['91270', '162274', '164803', '204396']})
            st.write(qtd_dfPessoas)

    def __rodape(self):
        st.markdown('## **Quem somos:**')
        st.markdown('- Ângela Magalhães')
        st.markdown('- Hugo Sousa')
        st.markdown('- Kamila Gomes')
        st.markdown('- Lucas Costa')
        st.markdown('- Lucinara Fernandes')
        st.markdown('- Thaís Félix')
