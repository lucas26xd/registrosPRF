import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor


class Predicoes():
    def __init__(self, options):
        self.month_names_missing = ['July', 'August', 'September', 'October', 'November', 'December']
        self.month_names = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
                            'October', 'November', 'December']
        self.y = options

        self.range_dates = [f'{mes + 1:0>2}/{2017 + ano}' for ano in range(5) for mes in range(12)]
        self.start_date, self.end_date = st.sidebar.select_slider('Intervalo de datas', options=self.range_dates,
                                                                  value=('01/2020', '03/2021'))
        self.qtds = st.sidebar.selectbox('Quantidade', options=self.y, index=1)

    def filter_dates(self, df):
        start_year = int(self.start_date[3:])
        start_month = int(self.start_date[:2])
        end_year = int(self.end_date[3:])
        end_month = int(self.end_date[:2])

        years = list()
        for y in range(end_year - start_year + 1):
            years.append(f'{start_year + y}')

        months = list()
        qtd_months = (len(years) * 12) - (start_month - 1) - (12 - end_month)
        for m in range(qtd_months):
            months.append(f'{self.month_names[(start_month + m - 1) % 12]}')

        return self.__calc_intervalo_meses_ano(df, months, years)

    def __calc_intervalo_meses_ano(self, df, months, years):
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

    def sort(self, df):
        sorter_index = dict(zip(self.month_names, range(len(self.month_names))))
        df['mes_rank'] = df['Mês'].map(sorter_index)
        df = df.sort_values(['Ano', 'mes_rank'])
        return df.drop(columns=['mes_rank'])

    def create_plot(self, df, by, title):
        fig = go.Figure()
        for faixa in df[by].unique():
            data = df[df[by] == faixa]
            meses = [f'{t[0][:3]} {t[1]}' for t in zip(data['Mês'].values, data['Ano'].values)]
            fig.add_trace(go.Scatter(x=meses, y=data[self.qtds].values, mode='lines+markers', name=faixa))
        fig.update_layout(title_text=title)
        return fig

    def predict(self, df, df_new, X_train, X_test):
        Y_train = df[self.y]

        regressor = RandomForestRegressor()
        regressor.fit(X_train, Y_train)
        previsoes = regressor.predict(X_test)

        previsoes = pd.DataFrame(previsoes)
        previsoes.columns = self.y

        for c in previsoes.columns:
            previsoes[c] = previsoes[c].astype(int)

        previsoes = pd.concat([df_new, previsoes], axis=1)

        return pd.concat([df, previsoes])


class PredicoesPessoasIdade():
    def __init__(self, pred, dfPessoas):
        self.x = ['Mês', 'Ano', 'Faixa Etária']
        self.pred = pred

        dfPessoas['data_inversa'] = pd.to_datetime(dfPessoas['data_inversa'])
        self.df = dfPessoas.copy()

        self.__agg_data(dfPessoas)
        self.__agg_new_data()

    def __agg_data(self, dfPessoas):
        self.df.loc[dfPessoas['idade'] >= 0, 'idade'] = 'Criança'
        self.df.loc[dfPessoas['idade'] >= 13, 'idade'] = 'Jovem'
        self.df.loc[dfPessoas['idade'] >= 25, 'idade'] = 'Adulto'
        self.df.loc[dfPessoas['idade'] >= 60, 'idade'] = 'Idoso'

        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%B'), self.df['data_inversa'].dt.strftime('%Y'), 'idade'])
        self.df = self.df.agg({'id': 'nunique', 'pesid': 'count', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.pred.y
        self.df = self.df.reset_index()

    def __agg_new_data(self):
        faixa_etaria_unique = self.df['Faixa Etária'].unique().tolist()

        mesh = np.array(np.meshgrid(self.pred.month_names_missing, ['2020'], faixa_etaria_unique)).T.reshape(-1, 3)
        df_new2020 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Faixa Etária'])

        mesh = np.array(np.meshgrid(self.pred.month_names, ['2021'], faixa_etaria_unique)).T.reshape(-1, 3)
        df_new2021 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Faixa Etária'])

        self.df_new = pd.concat([df_new2020, df_new2021])
        self.df_new.index = [i for i in range(len(self.df_new))]

        self.df_concatenated = pd.concat([self.df, self.df_new])

        # Encode meses
        for month in self.pred.month_names:
            self.df_concatenated.loc[self.df_concatenated['Mês'] == month, 'Mês'] = self.pred.month_names.index(month) + 1

        # Encode faixa etaria
        for faixa in faixa_etaria_unique:
            self.df_concatenated.loc[self.df_concatenated['Faixa Etária'] == faixa, 'Faixa Etária'] = faixa_etaria_unique.index(faixa) + 1

        self.df_concatenated.index = [i for i in range(len(self.df_concatenated))]

    def predicts(self):
        X_train = self.df_concatenated[:len(self.df)][self.x]
        X_test = self.df_concatenated[len(self.df):][self.x]

        return self.pred.predict(self.df, self.df_new, X_train, X_test)


class PredicoesOcorrenciasCausasAcidentes():
    def __init__(self, pred, dfOcorrencias):
        self.x = ['Mês', 'Ano', 'Causas de Acidentes']
        self.pred = pred

        dfOcorrencias['data_inversa'] = pd.to_datetime(dfOcorrencias['data_inversa'])
        self.df = dfOcorrencias.copy()

        self.__agg_data()
        self.__agg_new_data()

    def __agg_data(self):
        self.df = self.df.groupby([self.df['data_inversa'].dt.strftime('%B'), self.df['data_inversa'].dt.strftime('%Y'), 'causa_acidente'])
        self.df = self.df.agg({'id': 'nunique', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum', 'feridos': 'sum', 'veiculos': 'sum'})
        self.df.index.names = self.x
        self.df.columns = self.pred.y
        self.df = self.df.reset_index()

    def __agg_new_data(self):
        causa_acidente_unique = self.df['Causas de Acidentes'].unique().tolist()

        mesh = np.array(np.meshgrid(self.pred.month_names_missing, ['2020'], causa_acidente_unique)).T.reshape(-1, 3)
        df_new2020 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Causas de Acidentes'])

        mesh = np.array(np.meshgrid(self.pred.month_names, ['2021'], causa_acidente_unique)).T.reshape(-1, 3)
        df_new2021 = pd.DataFrame(mesh, columns=['Mês', 'Ano', 'Causas de Acidentes'])

        self.df_new = pd.concat([df_new2020, df_new2021])
        self.df_new.index = [i for i in range(len(self.df_new))]

        self.df_concatenated = pd.concat([self.df, self.df_new])

        # Encode meses
        for month in self.pred.month_names:
            self.df_concatenated.loc[self.df_concatenated['Mês'] == month, 'Mês'] = self.pred.month_names.index(month) + 1

        # Encode causa acidentes
        for causa in causa_acidente_unique:
            self.df_concatenated.loc[self.df_concatenated['Causas de Acidentes'] == causa, 'Causas de Acidentes'] = causa_acidente_unique.index(causa) + 1

        self.df_concatenated.index = [i for i in range(len(self.df_concatenated))]

    def predicts(self):
        X_train = self.df_concatenated[:len(self.df)][self.x]
        X_test = self.df_concatenated[len(self.df):][self.x]

        return self.pred.predict(self.df, self.df_new, X_train, X_test)
