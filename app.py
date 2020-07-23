#CREATED BY POR JUAN MINANGO
#email: jcarlosminango@gmail.com

import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor

# funcion para cargar la dataset
@st.cache
def get_data():
    return pd.read_csv("model/data.csv")


# função para treinar o modelo
def train_model():
    data = get_data()
    x = data.drop("MEDV",axis=1)
    y = data["MEDV"]
    rf_regressor = RandomForestRegressor(n_estimators=200, max_depth=7, max_features=3)
    rf_regressor.fit(x, y)
    return rf_regressor


def main():
    # generando un dataframe
    data = get_data()

    # entrenando el modelo
    model = train_model()

    # título
    st.title("Data App - Prediciendo el Valor de Inmuebles de la Ciudad de Boston")

    # subtítulo
    st.info("Este es un App de Predicción de Machine Learning utilizado para exibir el problema de predicción de valores de inmuebles de la ciudad de Boston.")

    # verificando el dataset
    st.subheader("Seleccione el conjunto de características de la base de datos")

    # atributos que son exibidos por default
    defaultcols = ["RM","PTRATIO","LSTAT","MEDV"]

    # definiendo atributos a partir de multiselect
    cols = st.multiselect("Características", data.columns.tolist(), default=defaultcols)

    # exibiendo los top 10 registros del dataframe
    st.dataframe(data[cols].head(10))


    st.subheader("Distribución de inmuebles por precio")

    # definienndo el rango de valores
    faixa_valores = st.slider("Rango de precios", float(data.MEDV.min()), 150., (10.0, 100.0))

    # filtrando los datos
    dados = data[data['MEDV'].between(left=faixa_valores[0],right=faixa_valores[1])]

    # plot la distribuicion de los datos
    f = px.histogram(dados, x="MEDV", nbins=100, title="Distribución de Precios")
    f.update_xaxes(title="MEDV")
    f.update_yaxes(title="Total de Inmuebles")
    st.plotly_chart(f)


    st.sidebar.subheader("Defina los atributos del inmueble para predicción")

    # mapeando datos de usuário para cada atributo
    crim = st.sidebar.number_input("Tasa de Criminalidad", value=data.CRIM.mean())
    indus = st.sidebar.number_input("Proporción de Hectares de Negócio", value=data.CRIM.mean())
    chas = st.sidebar.selectbox("Tiene límite con el río?",("Si","No"))

    # transformando los datos de entrada en valor binário
    chas = 1 if chas == "Si" else 0

    nox = st.sidebar.number_input("Concentración de óxido nítrico", value=data.NOX.mean())

    rm = st.sidebar.number_input("Número de Cuartos", value=1)

    ptratio = st.sidebar.number_input("Índice de alunos para profesores",value=data.PTRATIO.mean())

    b = st.sidebar.number_input("Proporción de personar de descendencia afro-americana",value=data.B.mean())

    lstat = st.sidebar.number_input("Porcentaje de status bajo",value=data.LSTAT.mean())

    # insertando un boton en la pantalla
    btn_predict = st.sidebar.button("Realizar Predicción")

    #Agradecimiento
    st.sidebar.info("Desarrollado por Juan Minango")

    # verifica se o botão foi acionado
    if btn_predict:
        result = model.predict([[crim,indus,chas,nox,rm,ptratio,b,lstat]])
        st.subheader("El valor previsto para el inmueble con las caracteristicas escojidas es:")
        result = "US $ "+str(round(result[0]*10,2))
        st.write(result)


if __name__ == '__main__':
    main()