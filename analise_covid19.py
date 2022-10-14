# Importando as bibliotecas necessárias para o desenvolvimento:
import pandas as pd                     # Para gerir os datasets/dataframes
import numpy as np                      # Calculos de computação cientifica 
from datetime import datetime           # Manipulação de datas
import plotly.express as px             # Visualização e exportação de Gráficos dinâmicos
import plotly.graph_objects as go
import plotly.offline as py
import plotly.io as pio
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt         # Plotagem dos gráficos estáticos
from pmdarima.arima import auto_arima   # Para predições usando ARIMA
from prophet import Prophet             # Para predições usando Prophet
import re                               # Biblioteca RE (Regular Expressions) para agilizar as formatações

py.init_notebook_mode(connected=True)

# Importando dados para o projeto:
# url = url do arquivo no github | como usei um arquivo local na minha máquina, usei a linha 21. Quando empurrar os arquivos para o GH basta comentar a linha 21 e descomentar as linha 19 e 20
# url = "urldoarquivonogithub"
# df = pd.read_csv(url, parse_dates =["ObservationDate", "Last Update"])
df = pd.read_csv("C:/Users/GeffDesk/Documents/Geração Tech Unimed BH - Ciência de Dados/Py - Códigos em aula/Modelo para prever evolução do Covid19 no Brasil/covid_19_data.csv", parse_dates =["ObservationDate", "Last Update"])
print(df)
print(df.shape)

# Conferir os tipos de dados de cada coluna
print(df.dtypes)

# Formatando as informações importadas para melhor visualisação e manipulação

def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()           # Substituindo um padrão por um valor | Sempre que houver a / ou espaços vazio eles serão substituidos por nada

print(corrige_colunas("AdgE/P"))                            # Testando função

# Corrigindo todas as colunas do DF
df.columns = [corrige_colunas(col) for col in  df.columns]
print(df)

# Selecionando apenas os dados do Brasil para investigação
df.loc[df.countryregion == "Brazil"]                         # Se não souber se o Brasil esta na lista, o comando df.countryregion.unique() retorna todos os paises que estáo no DF 
print(df.loc[df.countryregion == "Brazil"])

# Filtrando novamente para os dados que contenham pelo menos 1 caso por dia
brasil = df.loc[
    (df.countryregion == "Brazil") & 
    (df.confirmed > 0)
]
print(brasil)

# Investigando casos cofirmados (Gráficos)
casos_confirmados = px.line(brasil, "observationdate", "confirmed", title = "Casos confirmados no Brasil")
#casos_confirmados.show(renderer = "browser")                # Abre uma janela do navegador com o gráfico interativo

# Novos casos por dia (com gráficos)
# Tecnica de programação funcional criando uma função para uma nova coluna de novos casos | map, lambda(função anônima) np.arrange
brasil["novoscasos"] = list(map(
    lambda x: 0 if (x == 0) else brasil["confirmed"].iloc[x] -brasil["confirmed"].iloc[x - 1],      # Iterando cada dia por indice
    np.arange(brasil.shape[0])
))
# Criando o Gráfico:
novos_casos_por_dia = px.line(brasil, x = "observationdate", y = "novoscasos", title = "Novos Casos por dia")
#novos_casos_por_dia.show(renderer = "browser")

# Análise das mortes:

fig = go.Figure()                   # Criando um tipo figura para fazer os gráficos com linhas e pontos

fig.add_trace(
    go.Scatter(x = brasil.observationdate, y = brasil.deaths, name = "Mortes",
    mode = "lines + markers", line = {"color": "red"})
)
# Editando o Layout do gráfico:
fig.update_layout(title = "Mortes por COVID - 19 no Brasil")
#fig.show(renderer = "browser")

# Taxa de crescimento (implementando um função para calcular a taxa)
# taxa_crescimento = (presente / passado) ** (1/n) - 1
def taxa_crescimento(data, variable, data_inicio = None, data_fim = None):
    # Se data_inicio for None, define como a primeira data disponível
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
    else:
        data_fim = pd.to_datetime(data_fim)
    
    # Define os valores do presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]

    # Define o número de pontos no tempo que vamos avaliar
    n = (data_fim - data_inicio).days

    # Calcular a taxa
    taxa = (presente / passado) ** (1/n) - 1

    return taxa * 100

# Taxa de crescimento médio do COVIV no Brasil em todo o periodo
print(taxa_crescimento(brasil, "confirmed"),"%")

# Taxa de crescimento diario:
def taxa_crescimento_diaria(data, variable, data_inicio = None, data_fim = None):
    # Se data_inicio for None, define como a primeira data disponível
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)

    data_fim = data.observationdate.max()
    # Define o número de pontos no tempo que vamos avaliar
    n = (data_fim - data_inicio).days

    # Taxa calculada de um dia para o outro
    taxas = list(map(
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x - 1]) / data[variable].iloc[x - 1],
        range(1, n + 1)
    ))
    return np.array(taxas) * 100

tx_dia = taxa_crescimento_diaria(brasil, "confirmed")
print(tx_dia)

# Definindo os dias e plotando um gráfico da tx_dia
primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()

tx_dia_grafico = px.line(x = pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y = tx_dia, title = "Taxa de crescimento de casos confirmados no Brasil")
#tx_dia_grafico.show(renderer = "browser")

# Predições:
novoscasos = brasil.novoscasos
novoscasos.index = brasil.observationdate

res = seasonal_decompose(novoscasos)

fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.scatter(novoscasos.index, res.resid)
ax4.axhline(0, linestyle='dashed', c='black')
plt.show()

# Decompondo a serie confirmados
confirmados = brasil.confirmed
confirmados.index = brasil.observationdate
print(confirmados)

res2 = seasonal_decompose(confirmados)       # Retorna algum erro em relação a leitura do datetime

fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize = (10, 8))

ax1.plot(res.observed)
ax2.plot(res.trend)
ax3.plot(res.seasonal)
ax4.plot(confirmados.index, res.resid)
ax4.axhline(0, linestyle = "dashed", c = "black")
#plt.show()

# Modelo de predição usando a biblioteca ARIMA
# ARIMA: Modelagem por modelo de series temporais ARIMA = Média móvel integrada auto regressiva
# Integra o futuro como ação do passado para estimativa

modelo = auto_arima(confirmados)

fig = go.Figure(go.Scatter(
    x = confirmados.index, y = confirmados, name = "Observados"
))

fig.add_trace(go.Scatter(
    x = confirmados.index, y = modelo.predict_in_sample(), name = "Preditos"
))

fig.add_trace(go.Scatter(
    x = pd.date_range("2020-05-20", "2020-06-20"), y = modelo.predict(31), name = "Forecast"        # Definindo o Range da predição para 1 mês pd.date_range(inicio, fim)
))

fig.update_layout(title = "Previsão de casos confirmados no Brasil para os próximos 30 dias")
#fig.show("browser")

# Modelo de crescimento | Usando a biblioteca fbprophet


# Preprocessamentos
train = confirmados.reset_index()[:-5]
test = confirmados.reset_index()[-5:]

# Renomeando colunas | A Biblioteca Prophet tem alguns requisitos especiais para formatação de colunas segundo documentação: https://facebook.github.io/prophet/docs/installation.html
train.rename(columns = {"observationdate": "ds", "confirmed": "y"}, inplace = True)
test.rename(columns = {"observationdate": "ds", "confirmed": "y"}, inplace = True)

# Definir o modelo de crescimento
profeta = Prophet(growth = "logistic", changepoints = ["2020-03-21", "2020-03-30", "2020-04-25",    # Datas com pontos de mudanças bruscas
                                                        "2020-05-03", "2020-05-10"])


pop = 211463256     # Projeção da população Brasileira pelo IBGE
train["cap"] = pop

# Treina o modelo
profeta.fit(train)

# Construir previsões para o futuro 
future_dates = profeta.make_future_dataframe(periods = 200)
future_dates["cap"] = pop
forecast = profeta.predict(future_dates) 

# Gráfico para mostrar essa predição: | Considerando o pior cenário onde toda população Brasileira seria contaminada
fig = go.Figure()

fig.add_trace(go.Scatter(x = forecast.ds, y = forecast.yhat, name = "Predição"))
#fig.add_trace(go.Scatter(x = test.index, y = test, name = "Teste"))
fig.add_trace(go.Scatter(x = train.ds, y = train.y, name = "Treino"))
fig.update_layout(title ="Predições de casos confirmados no Brasil")
#fig.show("browser")

# Para outros cenários, só mudar a população (pop) na linha  187:
"""
! pop = 211463256     mudar esse valor para o valor de infectados que se deseja analisar 
e segue com o restante do código para treinamento e exibição do gráfico
"""
