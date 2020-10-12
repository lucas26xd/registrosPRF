import pandas as pd

dfPessoas = pd.read_csv('./Bases Limpas/dfPessoas.csv')

dfPessoas['data_inversa'] = pd.to_datetime(dfPessoas['data_inversa'])
dfPessoas['horario'] = pd.to_datetime(dfPessoas['horario'])

dfPessoas = dfPessoas.groupby([dfPessoas['data_inversa'].dt.strftime('%Y'), dfPessoas['data_inversa'].dt.strftime('%B'), 'sexo'])
dfPessoas = dfPessoas.agg({'id': 'nunique', 'pesid': 'nunique', 'ilesos': 'sum', 'feridos_leves': 'sum', 'feridos_graves': 'sum', 'mortos': 'sum'})

dfPessoas.index.names = ['ano', 'mes', 'faixa_idade']
dfPessoas.columns = ['qtd_acidentes', 'qtd_pessoas', 'qtd_ilesos', 'qtd_feridos_leves', 'qtd_feridos_graves', 'qtd_mortos']
dfPessoas = dfPessoas.reset_index()

print(dfPessoas)
