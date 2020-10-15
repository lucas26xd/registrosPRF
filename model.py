from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


class Regressores():
    def hold_out(self, X, Y, size=0.3):  # Separação da base em treino em teste usando a técnica holdout
        return train_test_split(X, Y, test_size=size)

    def label_encoder(self, df, attrs):
        LE = LabelEncoder()
        for i in attrs:
            df[i] = LE.fit_transform(df[i])
        return df

    def standard_scaler(self, X, Y):  # Escalonamento padrão de toda base
        SS = StandardScaler()
        X = SS.fit_transform(X)
        Y = SS.fit_transform(Y)
        return X, Y

    def regressao(self, regressor, X_train, Y_train, X_test, Y_test):  # Treina o regressor, imprime e retorna suas métricas
        regressor.fit(X_train, Y_train)

        score_train = regressor.score(X_train, Y_train)
        score_test = regressor.score(X_test, Y_test)
        print('Score Treino: ', score_train, '\nScore Teste: ', score_test)

        previsoes = regressor.predict(X_test)
        mae = mean_absolute_error(Y_test, previsoes)
        print('Erro médio absoluto: ', mae, '\n')

        return score_train, score_test, mae

    def regressao_linear(self, X_train, Y_train, X_test, Y_test):  # Cria o modelo de regressão linear
        regressor = LinearRegression()
        return self.regressao(regressor, X_train, Y_train, X_test, Y_test)

    def regressao_polinomial(self, X_train, Y_train, X_test, Y_test, dg=3):  # Cria o modelo de regressão polinomial
        poly = PolynomialFeatures(degree=dg)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.fit_transform(X_test)

        return self.regressao_linear(X_poly_train, Y_train, X_poly_test, Y_test)

    def arvore_decisao(self, X_train, Y_train, X_test, Y_test):  # Cria o modelo de árvore de decisão
        regressor = DecisionTreeRegressor()
        return self.regressao(regressor, X_train, Y_train, X_test, Y_test)

    def floresta_randomica(self, X_train, Y_train, X_test, Y_test, n=10):  # Cria o modelo de floresta randômica
        regressor = RandomForestRegressor(n_estimators=n)
        return self.regressao(regressor, X_train, Y_train, X_test, Y_test)

    def mlp(self, X_train, Y_train, X_test, Y_test, hls=(100,)):  # Cria o modelo de MLP
        regressor = MLPRegressor(hidden_layer_sizes=hls)
        return self.regressao(regressor, X_train, Y_train, X_test, Y_test)

    def knr(self, X_train, Y_train, X_test, Y_test, k=2):  # Cria o modelo de regressão dos k vizinhos
        regressor = KNeighborsRegressor(n_neighbors=k)
        return self.regressao(regressor, X_train, Y_train, X_test, Y_test)

    def plot_scores_bar(self, regressors, scs1, scs2, title):  # Plota o gráfico de barras dos scores de cada regressor
        fig = go.Figure(data=[go.Bar(name='Scores de Treino', x=regressors, y=scs1),
                              go.Bar(name='Scores de Teste', x=regressors, y=scs2)])
        fig.update_layout(barmode='group', title_text=title)
        # fig.show()
        return fig


class Hipotese1Pessoas():
    def __init__(self, dfPessoas):
        st.markdown('## **Número de Acidentes por Faixa Etária**')
        st.markdown('A partir da análise comparativa entre os scores e erros médio absolutos, de acordo com a abordagem'
        ' da hipótese de predição do Número de Acidentes por Faixa Etária, os regressores que mais '
        'se destacaram com melhores resultados foram: **Árvore de Decisão**, **Random Forest** e **Regressão Polinomial**')
        self.reg = Regressores()
        self.agrupa_dados(dfPessoas)
        self.divisao_dados()
        self.regressoes()
        self.regressoes2()


    def agrupa_dados(self, dfPessoas):
        self.x = ['ano', 'mes', 'faixa-idade']
        self.y = ['qtd_acidentes', 'qtd_pessoas', 'qtd_ilesos', 'qtd_feridos_leves', 'qtd_feridos_graves', 'qtd_mortos']

        self.df = dfPessoas.copy()

        self.df.loc[dfPessoas['idade'] >= 0, 'idade'] = 'Criança'
        self.df.loc[dfPessoas['idade'] >= 13, 'idade'] = 'Jovem'
        self.df.loc[dfPessoas['idade'] >= 25, 'idade'] = 'Adulto'
        self.df.loc[dfPessoas['idade'] >= 60, 'idade'] = 'Idoso'

        self.df['data_inversa'] = pd.to_datetime(self.df['data_inversa'])

        self.df = self.df.groupby(
            [self.df['data_inversa'].dt.strftime('%Y'), self.df['data_inversa'].dt.strftime('%B'), 'idade'])
        self.df = self.df.agg(
                {'id': 'nunique', 'pesid': 'count', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum',
                 'mortos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.y
        self.df = self.df.reset_index()

    def divisao_dados(self):
        # Label Encoder
        df_pessoas_tratado = self.reg.label_encoder(self.df.copy(), ['mes', 'faixa-idade'])

        # Divisão da base
        X = df_pessoas_tratado[self.x]
        Y = df_pessoas_tratado[self.y]

        # Standart Scaler
        # X, Y = standard_scaler(X, Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.reg.hold_out(X, Y)

    def regressoes(self):
        sc1_rl, sc2_rl, mae_rl = self.reg.regressao_linear(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rp, sc2_rp, mae_rp = self.reg.regressao_polinomial(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_ad, sc2_ad, mae_ad = self.reg.arvore_decisao(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rf, sc2_rf, mae_rf = self.reg.floresta_randomica(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rn, sc2_rn, mae_rn = self.reg.mlp(self.X_train, self.Y_train, self.X_test, self.Y_test, hls=(200, 100))
        sc1_kn, sc2_kn, mae_kn = self.reg.knr(self.X_train, self.Y_train, self.X_test, self.Y_test, k=5)

        regressores = ['Linear', 'Polinomial', 'Decision Tree', 'Random Forest', 'MLP', 'KNN']
        scores_treino = [sc1_rl, sc1_rp, sc1_ad, sc1_rf, sc1_rn, sc1_kn]
        scores_teste = [sc2_rl, sc2_rp, sc2_ad, sc2_rf, sc2_rn, sc2_kn]
        maes = [mae_rl, mae_rp, mae_ad, mae_rf, mae_rn, mae_kn]

        return self.reg.plot_scores_bar(regressores, scores_treino, scores_teste, '')

    def regressoes2(self):
        sc1_rl, sc2_rl, mae_rl = self.reg.regressao_linear(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rp, sc2_rp, mae_rp = self.reg.regressao_polinomial(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_ad, sc2_ad, mae_ad = self.reg.arvore_decisao(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rf, sc2_rf, mae_rf = self.reg.floresta_randomica(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rn, sc2_rn, mae_rn = self.reg.mlp(self.X_train, self.Y_train, self.X_test, self.Y_test, hls=(200, 100))
        sc1_kn, sc2_kn, mae_kn = self.reg.knr(self.X_train, self.Y_train, self.X_test, self.Y_test, k=5)

        regressores = ['Linear', 'Polinomial', 'Decision Tree', 'Random Forest', 'MLP', 'KNN']
        scores_treino = [sc1_rl, sc1_rp, sc1_ad, sc1_rf, sc1_rn, sc1_kn]
        scores_teste = [sc2_rl, sc2_rp, sc2_ad, sc2_rf, sc2_rn, sc2_kn]
        maes = [mae_rl, mae_rp, mae_ad, mae_rf, mae_rn, mae_kn]

        return px.bar(x=regressores, y=maes, color=maes, labels={'x': 'Regressores', 'y': 'MAE'},
            title='')


class Hipotese3Ocorrencias():
    def __init__(self, dfOcorrencias):
        st.markdown('## **Número de Acidentes por Causa de Acidente**')
        st.markdown('A partir da análise comparativa entre os scores e erros médio absolutos, de acordo com a abordagem'
        ' da hipótese de predição do Número de Acidentes por Causa de Acidente, os regressores que mais '
        'se destacaram com melhores resultados foram: **Árvore de Decisão**, **Random Forest** e **Regressão Polinomial**')
        self.reg = Regressores()
        self.agrupa_dados(dfOcorrencias)
        self.divisao_dados()
        self.regressoes()

    def agrupa_dados(self, dfOcorrencias):
        self.x = ['ano', 'mes', 'causa_acidente']
        self.y = ['qtd_acidentes', 'qtd_ilesos', 'qtd_feridos_leves', 'qtd_feridos_graves', 'qtd_mortos', 'qtd_feridos',
         'qtd_veiculos']

        self.df = dfOcorrencias.copy()

        self.df['data_inversa'] = pd.to_datetime(self.df['data_inversa'])
        self.df['horario'] = pd.to_datetime(self.df['horario'])

        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%Y'), self.df['data_inversa'].dt.strftime('%B'), 'causa_acidente'])
        self.df = self.df.agg({'id': 'nunique', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum', 'feridos': 'sum', 'veiculos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.y
        self.df = self.df.reset_index()

    def divisao_dados(self):
        # Label Encoder
        df_pessoas_tratado = self.reg.label_encoder(self.df.copy(), ['mes', 'causa_acidente'])

        # Divisão da base
        X = df_pessoas_tratado[self.x]
        Y = df_pessoas_tratado[self.y]

        # Standart Scaler
        # X, Y = standard_scaler(X, Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.reg.hold_out(X, Y)

    def regressoes(self):
        sc1_rl, sc2_rl, mae_rl = self.reg.regressao_linear(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rp, sc2_rp, mae_rp = self.reg.regressao_polinomial(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_ad, sc2_ad, mae_ad = self.reg.arvore_decisao(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rf, sc2_rf, mae_rf = self.reg.floresta_randomica(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rn, sc2_rn, mae_rn = self.reg.mlp(self.X_train, self.Y_train, self.X_test, self.Y_test, hls=(200, 100))
        sc1_kn, sc2_kn, mae_kn = self.reg.knr(self.X_train, self.Y_train, self.X_test, self.Y_test, k=5)

        regressores = ['Linear', 'Polinomial', 'Decision Tree', 'Random Forest', 'MLP', 'KNN']
        scores_treino = [sc1_rl, sc1_rp, sc1_ad, sc1_rf, sc1_rn, sc1_kn]
        scores_teste = [sc2_rl, sc2_rp, sc2_ad, sc2_rf, sc2_rn, sc2_kn]
        maes = [mae_rl, mae_rp, mae_ad, mae_rf, mae_rn, mae_kn]

        return self.reg.plot_scores_bar(regressores, scores_treino, scores_teste,'')
        #

    def regressoes2(self):
        sc1_rl, sc2_rl, mae_rl = self.reg.regressao_linear(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rp, sc2_rp, mae_rp = self.reg.regressao_polinomial(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_ad, sc2_ad, mae_ad = self.reg.arvore_decisao(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rf, sc2_rf, mae_rf = self.reg.floresta_randomica(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rn, sc2_rn, mae_rn = self.reg.mlp(self.X_train, self.Y_train, self.X_test, self.Y_test, hls=(200, 100))
        sc1_kn, sc2_kn, mae_kn = self.reg.knr(self.X_train, self.Y_train, self.X_test, self.Y_test, k=5)

        regressores = ['Linear', 'Polinomial', 'Decision Tree', 'Random Forest', 'MLP', 'KNN']
        scores_treino = [sc1_rl, sc1_rp, sc1_ad, sc1_rf, sc1_rn, sc1_kn]
        scores_teste = [sc2_rl, sc2_rp, sc2_ad, sc2_rf, sc2_rn, sc2_kn]
        maes = [mae_rl, mae_rp, mae_ad, mae_rf, mae_rn, mae_kn]

        return px.bar(x=regressores, y=maes, color=maes, labels={'x': 'Regressores', 'y': 'MAE'},
                title='')

class Gidade():
    def __init__(self,d1,d2):
        self.graficos(d1,d2)

    def graficos(self, d1, d2):
        options = st.selectbox('Confira os Gráficos:',
                                   ['', 'Scores de TREINO/TESTE (Mês, Ano, Faixa Etária)', 'Erro Médio Absoluto (Mês, Ano, Faixa Etária)'])
        if options == 'Scores de TREINO/TESTE (Mês, Ano, Faixa Etária)':
            st.write(
                    'Scores de TREINO/TESTE dos regressores na base de pessoas agrupadas (Mês, Ano, Faixa Etária).')
            st.write(d1)
        elif options == 'Erro Médio Absoluto (Mês, Ano, Faixa Etária)':
            st.write('ERRO MÉDIO ABSOLUTO dos regressores na base de pessoas agrupadas (Mês, Ano, Faixa Etária).')
            st.write(d2)

class Gcausa():
    def __init__(self,d1,d2):
        self.graficos(d1,d2)

    def graficos(self, d1, d2):
        options = st.selectbox('Confira os Gráficos:',
                                   ['', 'Scores de TREINO/TESTE (Mês, Ano, Causa Acidente)', 'Erro Médio Absoluto (Mês, Ano, Causa Acidente)'])
        if options == 'Scores de TREINO/TESTE (Mês, Ano, Causa Acidente)':
            st.write(
                    'Scores de TREINO/TESTE dos regressores na base de pessoas agrupadas (Mês, Ano, Causa Acidente).')
            st.write(d1)
        elif options == 'Erro Médio Absoluto (Mês, Ano, Causa Acidente)':
            st.write('ERRO MÉDIO ABSOLUTO dos regressores na base de pessoas agrupadas (Mês, Ano, Causa Acidente).')
            st.write(d2)