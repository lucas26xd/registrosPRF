import pandas as pd

url = 'https://raw.githubusercontent.com/lucas26xd/registrosPRF/master/Bases'
bases = {'Ocorrencias': {'folder': 'Agrupados%20por%20ocorrencia',
                         'files': ['datatran2017.csv', 'datatran2018.csv', 'datatran2019.csv', 'datatran2020.csv']},
         'Pessoas': {'folder': 'Agrupados%20por%20pessoa',
                     'files': ['acidentes2017.csv', 'acidentes2018.csv', 'acidentes2019.csv', 'acidentes2020.csv']},
         }


def import_base(base):
    dfs = list()
    for file in bases[base]['files']:
        dfs.append(pd.read_csv(f'{url}/{bases[base]["folder"]}/{file}', encoding='ISO-8859-1', sep=';'))

    return pd.concat(dfs)


def converte_srt_to_float(df, atrib):
    for i in range(0, len(atrib)):
        df[atrib[i]] = df[atrib[i]].apply(lambda x: x.replace(",", "."))
        df[atrib[i]] = df[atrib[i]].astype(float)


def converte_float_to_int(df, atrib):
    for i in range(0, len(atrib)):
        df[atrib[i]] = df[atrib[i]].astype(int)


def scrub_ocorrencias():
    dfOcorrencias = import_base('Ocorrencias')

    # Remoção de atributos considerados irrelevantes
    dfOcorrencias = dfOcorrencias.drop(columns=['fase_dia', 'sentido_via', 'uso_solo', 'latitude', 'longitude',
                                                'regional', 'delegacia', 'uop'])

    #
    # e regitros NaN nos atributos br e km
    t = len(dfOcorrencias.id)
    dfOcorrencias = dfOcorrencias[dfOcorrencias['br'].notna()]
    dfOcorrencias = dfOcorrencias[dfOcorrencias['km'].notna()]
    print(f'Removidos {t - len(dfOcorrencias.id)} registros.')

    # Conversões
    atributos_int = ["id", "br", "pessoas", "mortos", "feridos_leves", "feridos_graves", "ilesos", "ignorados",
                     "feridos", "veiculos"]
    converte_float_to_int(dfOcorrencias, atributos_int)
    converte_srt_to_float(dfOcorrencias, ["km"])

    dfOcorrencias['data_inversa'] = pd.to_datetime(dfOcorrencias['data_inversa'])
    dfOcorrencias['horario'] = pd.to_datetime(dfOcorrencias['horario'])

    return dfOcorrencias


def scrub_pessoas():
    dfPessoas = import_base('Pessoas')

    # Remoção de atributos considerados irrelevantes
    dfPessoas = dfPessoas.drop(columns=['fase_dia', 'sentido_via', 'uso_solo', 'id_veiculo', 'marca', 'delegacia',
                                        'ano_fabricacao_veiculo', 'latitude', 'longitude', 'regional', 'uop'])

    # Remove regitros NaN nos atributos br e km
    t = len(dfPessoas.id)
    dfPessoas = dfPessoas[dfPessoas['br'].notna()]
    dfPessoas = dfPessoas[dfPessoas['km'].notna()]
    print(f'Removidos {t - len(dfPessoas.id)} registros.')

    # Pessoas com sexo = 'Ignorado' virarão sexo = 'Não Informado'
    dfPessoas.loc[dfPessoas['sexo'] == 'Ignorado', 'sexo'] = 'Não Informado'

    # Conversões
    dfPessoas['id'] = dfPessoas['id'].astype(int)
    dfPessoas['data_inversa'] = pd.to_datetime(dfPessoas['data_inversa'])
    dfPessoas['horario'] = pd.to_datetime(dfPessoas['horario'])

    # Substituindo valores NaN por PESIDs novos
    qtd_pesid_nan = len(dfPessoas[dfPessoas['pesid'].isna()]['pesid'])
    dfPessoas.loc[dfPessoas['pesid'].isna(), 'pesid'] = [dfPessoas['pesid'].max() + 1 + i for i in range(qtd_pesid_nan)]
    dfPessoas['pesid'] = dfPessoas['pesid'].astype(int)

    # Preenchendo idades faltantes ou outliers com a média dos valores válidos
    media_idade = dfPessoas.loc[dfPessoas['idade'] <= 110, 'idade'].mean()
    dfPessoas.loc[(dfPessoas['idade'].isna()) | (dfPessoas['idade'] > 110), 'idade'] = media_idade
    dfPessoas['idade'] = dfPessoas['idade'].astype(int)

    return dfPessoas