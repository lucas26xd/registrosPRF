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


class Regressores:
    def __init__(self):
        self.y = ['qtd_acidentes', 'qtd_pessoas', 'qtd_ilesos', 'qtd_feridos_leves', 'qtd_feridos_graves', 'qtd_mortos']
        self.regressors = ['Linear', 'Polinomial', 'Decision Tree', 'Random Forest', 'MLP', 'KNN']

    @staticmethod
    def __hold_out(X, Y, size=0.3):  # Separação da base em treino em teste usando a técnica holdout
        return train_test_split(X, Y, test_size=size)

    @staticmethod
    def __label_encoder(df, attrs):
        LE = LabelEncoder()
        for a in attrs:
            df[a] = LE.fit_transform(df[a])
        return df

    @staticmethod
    def __standard_scaler(X, Y):  # Escalonamento padrão de toda base
        SS = StandardScaler()
        X = SS.fit_transform(X)
        Y = SS.fit_transform(Y)
        return X, Y

    @staticmethod
    def __regression(regressor, X_train, Y_train, X_test, Y_test):  # Treina o regressor, imprime e retorna suas métricas
        regressor.fit(X_train, Y_train)

        score_train = regressor.score(X_train, Y_train)
        score_test = regressor.score(X_test, Y_test)
        # print('Score Treino: ', score_train, '\nScore Teste: ', score_test)

        previsoes = regressor.predict(X_test)
        mae = mean_absolute_error(Y_test, previsoes)
        # print('Erro médio absoluto: ', mae, '\n')

        return score_train, score_test, mae

    def __linear(self, X_train, Y_train, X_test, Y_test):  # Cria o modelo de regressão linear
        regressor = LinearRegression()
        return self.__regression(regressor, X_train, Y_train, X_test, Y_test)

    def __polinomial(self, X_train, Y_train, X_test, Y_test, dg=3):  # Cria o modelo de regressão polinomial
        poly = PolynomialFeatures(degree=dg)
        X_poly_train = poly.fit_transform(X_train)
        X_poly_test = poly.fit_transform(X_test)

        return self.__linear(X_poly_train, Y_train, X_poly_test, Y_test)

    def __decision_tree(self, X_train, Y_train, X_test, Y_test):  # Cria o modelo de árvore de decisão
        regressor = DecisionTreeRegressor()
        return self.__regression(regressor, X_train, Y_train, X_test, Y_test)

    def __random_forest(self, X_train, Y_train, X_test, Y_test, n=10):  # Cria o modelo de floresta randômica
        regressor = RandomForestRegressor(n_estimators=n)
        return self.__regression(regressor, X_train, Y_train, X_test, Y_test)

    def __mlp(self, X_train, Y_train, X_test, Y_test, hls=(100,)):  # Cria o modelo de MLP
        regressor = MLPRegressor(hidden_layer_sizes=hls)
        return self.__regression(regressor, X_train, Y_train, X_test, Y_test)

    def __knr(self, X_train, Y_train, X_test, Y_test, k=2):  # Cria o modelo de regressão dos k vizinhos
        regressor = KNeighborsRegressor(n_neighbors=k)
        return self.__regression(regressor, X_train, Y_train, X_test, Y_test)

    def __plot_scores_bar(self, scs1, scs2):  # Plota o gráfico de barras dos scores de cada regressor
        fig = go.Figure(data=[go.Bar(name='Scores de Treino', x=self.regressors, y=scs1),
                              go.Bar(name='Scores de Teste', x=self.regressors, y=scs2)])
        fig.update_layout(barmode='group')
        return fig

    def __plot_mae_bar(self, maes):
        return px.bar(x=self.regressors, y=maes, color=maes, labels={'x': 'Regressores', 'y': 'MAE'})

    def agg_data(self, df, x):
        self.x = x
        self.df['data_inversa'] = pd.to_datetime(df['data_inversa'])

        agg_atrrs = {'id': 'nunique'}
        if 'pesid' in self.df.columns:
            agg_atrrs['pesid'] = 'count'
        else:
            self.y.remove('qtd_pessoas')
        agg_atrrs.update({'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum'})

        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%Y'),
                                   self.df['data_inversa'].dt.strftime('%B'),
                                   self.x[-1]])
        self.df = self.df.agg(agg_atrrs)
        self.df.index.names = self.x
        self.df.columns = self.y
        self.df = self.df.reset_index()

    def split_data(self):
        # Label Encoder
        df_pessoas_tratado = self.__label_encoder(self.df.copy(), ['mes', self.x[-1]])

        # Divisão da base
        X = df_pessoas_tratado[self.x]
        Y = df_pessoas_tratado[self.y]

        # Standart Scaler
        # X, Y = self.__standard_scaler(X, Y)

        self.X_train, self.X_test, self.Y_train, self.Y_test = self.__hold_out(X, Y)

    def regressions(self):
        sc1_rl, sc2_rl, mae_rl = self.__linear(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rp, sc2_rp, mae_rp = self.__polinomial(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_ad, sc2_ad, mae_ad = self.__decision_tree(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rf, sc2_rf, mae_rf = self.__random_forest(self.X_train, self.Y_train, self.X_test, self.Y_test)
        sc1_rn, sc2_rn, mae_rn = self.__mlp(self.X_train, self.Y_train, self.X_test, self.Y_test, hls=(200, 100))
        sc1_kn, sc2_kn, mae_kn = self.__knr(self.X_train, self.Y_train, self.X_test, self.Y_test, k=5)

        scores_treino = [sc1_rl, sc1_rp, sc1_ad, sc1_rf, sc1_rn, sc1_kn]
        scores_teste = [sc2_rl, sc2_rp, sc2_ad, sc2_rf, sc2_rn, sc2_kn]
        maes = [mae_rl, mae_rp, mae_ad, mae_rf, mae_rn, mae_kn]

        return (self.__plot_scores_bar(scores_treino, scores_teste),
                self.__plot_mae_bar(maes))


class Hipotese1Pessoas(Regressores):
    def __init__(self, dfPessoas):
        super().__init__()
        self.x = ['ano', 'mes', 'faixa-etaria']
        st.markdown('## **Número de Acidentes por Faixa Etária**')
        st.markdown('''A partir da análise comparativa entre os scores e erros médio absolutos, de acordo com a 
        abordagem da hipótese de predição do Número de Acidentes por Faixa Etária, os regressores que mais se destacaram
         com melhores resultados foram **Árvore de Decisão** e **Random Forest**''')
        self.__agg_data(dfPessoas)
        self.split_data()

    def __agg_data(self, dfPessoas):
        self.df = dfPessoas.copy()

        self.df.loc[dfPessoas['idade'] >= 0, 'idade'] = 'Criança'
        self.df.loc[dfPessoas['idade'] >= 13, 'idade'] = 'Jovem'
        self.df.loc[dfPessoas['idade'] >= 25, 'idade'] = 'Adulto'
        self.df.loc[dfPessoas['idade'] >= 60, 'idade'] = 'Idoso'
        self.df = self.df.rename(columns={'idade': self.x[-1]})

        self.agg_data(self.df, self.x)


class Hipotese3Ocorrencias(Regressores):
    def __init__(self, dfOcorrencias):
        super().__init__()
        self.x = ['ano', 'mes', 'causa_acidente']
        st.markdown('## **Número de Acidentes por Causa de Acidente**')
        st.markdown('''A partir da análise comparativa entre os scores e erros médio absolutos, de acordo com a 
        abordagem da hipótese de predição do Número de Acidentes por Causa de Acidente, os regressores que mais se 
        destacaram com melhores resultados foram **Árvore de Decisão** e **Random Forest**''')
        self.__agg_data(dfOcorrencias)
        self.split_data()

    def __agg_data(self, dfOcorrencias):
        self.df = dfOcorrencias.copy()

        self.agg_data(self.df, self.x)
