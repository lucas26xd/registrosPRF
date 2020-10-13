import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go


class PredicoesPessoasIdade():
    def __init__(self, dfPessoas):
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.y = ['Acidentes', 'Envolvidos', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos']
        self.x = ['Mês', 'Ano', 'Faixa Etária']

        dfPessoas['data_inversa'] = pd.to_datetime(dfPessoas['data_inversa'])
        self.df = dfPessoas.copy()

        self.agg_data(dfPessoas)
        self.agg_new_data()

    def agg_data(self, dfPessoas):
        self.df.loc[dfPessoas['idade'] >= 0, 'idade'] = 'Criança'
        self.df.loc[dfPessoas['idade'] >= 13, 'idade'] = 'Jovem'
        self.df.loc[dfPessoas['idade'] >= 25, 'idade'] = 'Adulto'
        self.df.loc[dfPessoas['idade'] >= 60, 'idade'] = 'Idoso'

        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%B'), self.df['data_inversa'].dt.strftime('%Y'), 'idade'])
        self.df = self.df.agg({'id': 'nunique', 'pesid': 'count', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.y
        self.df = self.df.reset_index()

    def agg_new_data(self):
        month_names_missing = ['July', 'August', 'September', 'October', 'November', 'December']
        faixa_etaria_unique = self.df['Faixa Etária'].unique().tolist()

        mesh = np.array(np.meshgrid(month_names_missing, ['2020'], faixa_etaria_unique)).T.reshape(-1, 3)
        df_new2020 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Faixa Etária'])

        mesh = np.array(np.meshgrid(self.month_names, ['2021'], faixa_etaria_unique)).T.reshape(-1, 3)
        df_new2021 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Faixa Etária'])

        self.df_new = pd.concat([df_new2020, df_new2021])
        self.df_new.index = [i for i in range(len(self.df_new))]

        self.df_concatenated = pd.concat([self.df, self.df_new])

        # Encode meses
        for month in self.month_names:
            self.df_concatenated.loc[self.df_concatenated['Mês'] == month, 'Mês'] = self.month_names.index(month) + 1

        # Encode faixa etaria
        for faixa in faixa_etaria_unique:
            self.df_concatenated.loc[self.df_concatenated['Faixa Etária'] == faixa, 'Faixa Etária'] = faixa_etaria_unique.index(faixa) + 1

        self.df_concatenated.index = [i for i in range(len(self.df_concatenated))]
        # print(self.df_concatenated)

    def predicts(self):
        X_train = self.df_concatenated[:len(self.df)][self.x]
        X_test = self.df_concatenated[len(self.df):][self.x]
        Y_train = self.df[self.y]

        # poly = PolynomialFeatures(degree=3)
        # X_poly_train = poly.fit_transform(X_train)
        # X_poly_test = poly.fit_transform(X_test)
        #
        # regressor = LinearRegression()
        # regressor.fit(X_poly_train, Y_train)
        #
        # previsoes = regressor.predict(X_poly_test)

        regressor = RandomForestRegressor()
        regressor.fit(X_train, Y_train)
        previsoes = regressor.predict(X_test)

        previsoes = pd.DataFrame(previsoes)
        previsoes.columns = self.y
        for c in previsoes.columns:
            previsoes[c] = previsoes[c].astype(int)

        previsoes = pd.concat([self.df_new, previsoes], axis=1)

        return previsoes, pd.concat([self.df, previsoes])

    def create_plot(self, df, attrs, title):
        fig = go.Figure()
        for faixa in df['Faixa Etária'].unique():
            data = df[df['Faixa Etária'] == faixa]
            meses = [f'{t[0][:3]} {t[1]}' for t in zip(data['Mês'].values, data['Ano'].values)]
            fig.add_trace(go.Scatter(x=meses, y=data[attrs].values, mode='lines+markers', name=faixa))
        fig.update_layout(title_text=title)
        return fig


class PredicoesOcorrenciasCausasAcidentes():
    def __init__(self, dfOcorrencias):
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
        self.y = ['Acidentes', 'Ilesos', 'Feridos Leves', 'Feridos Graves', 'Mortos', 'Feridos', 'Veículos']
        self.x = ['Mês', 'Ano', 'Causas de Acidentes']

        dfOcorrencias['data_inversa'] = pd.to_datetime(dfOcorrencias['data_inversa'])
        self.df = dfOcorrencias.copy()

        self.agg_data(dfOcorrencias)
        self.agg_new_data()

    def agg_data(self, dfOcorrencias):
        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%B'), self.df['data_inversa'].dt.strftime('%Y'), 'causa_acidente'])
        self.df = self.df.agg({'id': 'nunique', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum', 'feridos': 'sum', 'veiculos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.y
        self.df = self.df.reset_index()

    def agg_new_data(self):
        month_names_missing = ['July', 'August', 'September', 'October', 'November', 'December']
        causa_acidente_unique = self.df['Causas de Acidentes'].unique().tolist()

        mesh = np.array(np.meshgrid(month_names_missing, ['2020'], causa_acidente_unique)).T.reshape(-1, 3)
        df_new2020 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Causas de Acidentes'])

        mesh = np.array(np.meshgrid(self.month_names, ['2021'], causa_acidente_unique)).T.reshape(-1, 3)
        df_new2021 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Causas de Acidentes'])

        self.df_new = pd.concat([df_new2020, df_new2021])
        self.df_new.index = [i for i in range(len(self.df_new))]

        self.df_concatenated = pd.concat([self.df, self.df_new])

        # Encode meses
        for month in self.month_names:
            self.df_concatenated.loc[self.df_concatenated['Mês'] == month, 'Mês'] = self.month_names.index(month) + 1

        # Encode faixa etaria
        for causa in causa_acidente_unique:
            self.df_concatenated.loc[self.df_concatenated['Causas de Acidentes'] == causa, 'Causas de Acidentes'] = causa_acidente_unique.index(causa) + 1

        self.df_concatenated.index = [i for i in range(len(self.df_concatenated))]
        # print(self.df_concatenated)

    def predicts(self):
        X_train = self.df_concatenated[:len(self.df)][self.x]
        X_test = self.df_concatenated[len(self.df):][self.x]
        Y_train = self.df[self.y]

        regressor = RandomForestRegressor()
        regressor.fit(X_train, Y_train)
        previsoes = regressor.predict(X_test)

        previsoes = pd.DataFrame(previsoes)
        previsoes.columns = self.y

        previsoes = pd.concat([self.df_new, previsoes], axis=1)

        return previsoes, pd.concat([self.df, previsoes])

    def create_plot(self, df, attrs, title):
        fig = go.Figure()
        for causa in df['Causas de Acidentes'].unique():
            data = df[df['Causas de Acidentes'] == causa]
            meses = [f'{t[0][:3]} {t[1]}' for t in zip(data['Mês'].values, data['Ano'].values)]
            fig.add_trace(go.Scatter(x=meses, y=data[attrs].values, mode='lines+markers', name=causa))
        fig.update_layout(title_text=title)
        return fig