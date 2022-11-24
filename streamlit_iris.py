import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing as pp
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Config streamlit
#-----------------------------------------------------------------------------------
st.set_page_config(page_title="Predict heart", page_icon=":tada:", layout="wide")



st.sidebar.header("Les parametres d'entrée")

def user_input():
    sepal_length=st.sidebar.slider('La longeur du Sepal',4.3,7.9,5.3)
    sepal_width=st.sidebar.slider('La largeur du Sepal',2.0,4.4,3.3)
    petal_length=st.sidebar.slider('La longeur du Petal',1.0,6.9,2.3)
    petal_width=st.sidebar.slider('La largeur du Petal',0.1,2.5,1.3)
    data={'Longueur de sépal':sepal_length,
    'Largeur de sépal':sepal_width,
    'Longueur de pétale':petal_length,
    'Largeur de pétale':petal_width
    }
    fleur_parametres=pd.DataFrame(data,index=[0])
    return fleur_parametres

param= user_input()

st.subheader('Les caractéristiques de l\'iris recherché')
st.write(param)


# ---------------------------------------------------------------------
# Chargement du dataset
#-----------------------------------------------------------------------------------
data =pd.read_csv("iris.csv")
data["variety_label"]=pp.LabelEncoder().fit_transform(data["variety"])
data_iris=data.iloc[:,:-2]
data_target=data["variety_label"]


# ---------------------------------------------------------------------
# Creation du model
#-----------------------------------------------------------------------------------
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)
model.fit(data_iris,data_target)

prediction=model.predict(param)

st.subheader(f"La catégorie de la fleur d'iris est : {data.loc[data['variety_label']==prediction[0],'variety'].unique()[0]}")

if st.button('Graphique des caractéristiques'):
    fig= sns.pairplot(data=data.iloc[:,:-1],hue="variety")
    st.pyplot(fig)
