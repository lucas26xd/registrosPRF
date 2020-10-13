import pandas as pd
import streamlit as st
from PIL import Image


class Introducao():
    def __init__(self):
        self.image = Image.open('./Imagens/logo.jpg')
        self.__cabecalho()
        self.__agg_ocorrencias()
        self.__agg_pessoas()
        self.__rodape()

    def __cabecalho(self):
        st.image(self.image, use_column_width=True)
        st.markdown(' ')
        st.markdown('**Introdução**')
        st.markdown("""A Polícia Rodoviária Federal disponibiliza em seu site os dados gerais de acidentes nas rodovias federais brasileiras, como segue no link a seguir: https://portal.prf.gov.br/dados-abertos-acidentes. 
                Segundo a _Agência CNT Transporte Atual_  " os números mostram queda de 2,6% nas ocorrências quando comparado os anos de 2018 e 2019.
                 Já os acidentes com vítimas (mortos e feridos), por sua vez, tiveram elevação de 3,3%, subindo de 53.963, em 2018, para 55.756. Foram 2.526 feridos a mais em 2019.
                Além disso, em 2019, o número de mortes cresceu 1,2%, passando para 5.332 (63 óbitos a mais que em 2018). Foi o primeiro aumento em sete anos. De 2012 a 2018, as mortes nas rodovias federais
                tiveram queda de 39,2%, com sucessivas reduções a cada ano".""")
        st.markdown("""Baseados nesse estudo e nos dados disponibilizados no site, como mostrado anteriormente, o trabalho a seguir faz a análise estatística sobre a
                as bases de dados: Agrupados por ocorrência e Agrupados por pessoas, entre os anos de 2017 em diante. Em seguida mostraremos também algumas possíveis hipóteses futuras e a aplicação em dos dados em alguns regressores.""")

    def __agg_ocorrencias(self):
        st.markdown(' **Agrupados por Ocorrências**')
        st.write("Selecione a opção desejada a cerca das informações sobre a Base Ocorrências.")
        options = st.selectbox("Qual opção deseja?", ["Atributos de Ocorrências", "Quantidade de Registros de Ocorrências"])

        if options == "Atributos de Ocorrências":
            table_dfOcorrencia = pd.DataFrame({
                'atributos': ['id', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km', 'municipio',
                              'causa_acidente', 'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'sentido_via',
                              'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'ano', 'pessoas', 'mortos',
                              'feridos_leves', 'feridos_graves', 'ilesos', 'ignorados', 'feridos', 'veiculos', 'latitude',
                              'longitude', 'regional', 'delegacia', 'uop'],
                'status': ['não apagado', 'não apagado', 'não apagado', 'não apagado', 'não apagado', 'não apagado',
                           'não apagado', 'não apagado', 'não apagado', 'não apagado', 'não apagado', 'apagado', 'apagado',
                           'não apagado', 'não apagado', ' nao apagado', 'apagado', 'não apagado', 'não apagado',
                           'não apagado', 'não apagado', 'não apagado', 'não apagado', 'não apagado', 'não apagado',
                           'nao apagado', 'apagado', 'apagado', 'apagado', 'apagado', 'apagado']})
            st.write(table_dfOcorrencia)
        elif options == "Quantidade de Registros de Ocorrências":
            qtd_dfOcorrencia = pd.DataFrame({
                'ano': ['2017', '2018', '2019', '2020'],
                'quantidade de registros': ['89563', '69295', '67446', '34084']})
            st.write(qtd_dfOcorrencia)

    def __agg_pessoas(self):
        st.markdown(' **Agrupados por Pessoas**')
        st.write("Selecione a opção desejada a cerca das informações sobre a Base Pessoas.")
        optionPessoas = st.selectbox("Qual opção deseja?", ["Atributos de Pessoas", "Quantidade de Registros de Pessoas"])

        if optionPessoas == "Atributos de Pessoas":
            table_dfPessoas = pd.DataFrame({
                'atributos': ['id', 'pesid', 'data_inversa', 'dia_semana', 'horario', 'uf', 'br', 'km', 'municipio',
                              'causa_acidente', 'tipo_acidente', 'classificacao_acidente', 'fase_dia', 'sentido_via',
                              'condicao_metereologica', 'tipo_pista', 'tracado_via', 'uso_solo', 'id_veiculo',
                              'tipo_veiculo', 'marca', 'ano_fabricacao_veiculo', 'tipo_envolvido', 'estado_fisico', 'idade',
                              'sexo', 'ilesos', 'feridos leves', 'feridos graves', 'mortos', 'latitude', 'longitude',
                              'regional', 'delegacia', 'uop'],
                'status': ['nao apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado',
                           'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado',
                           'apagado', 'apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'apagado', 'apagado',
                           'nao apagado', 'apagado', 'apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado',
                           'nao apagado', 'nao apagado', 'nao apagado', 'nao apagado', 'apagado', 'apagado', 'apagado',
                           'apagado', 'apagado']})
            st.write(table_dfPessoas)
        elif optionPessoas == "Quantidade de Registros de Pessoas":
            qtd_dfPessoas = pd.DataFrame({
                'ano': ['2017', '2018', '2019', '2020'],
                'quantidade de registros': ['91270', '162274', '164803', '204396']})
            st.write(qtd_dfPessoas)

    def __rodape(self):
        st.markdown('**Quem somos:**')
        st.markdown('Ângela Magalhães, Hugo Sousa, Kamila Gomes, Lucas Costa, Lucinaria Fernandes, Thaís Félix.')
