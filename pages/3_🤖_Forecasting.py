# coding=utf-8
""" """
# Copyright 2023, Swiss Statistical Design & Innovation Sàrl

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.metrics import mean_absolute_error as mae
from sklearn.tree import DecisionTreeRegressor

st.set_page_config(
    page_title="Forecasting",
    page_icon="🤖",
)

st.title('Prédisez !')

st.markdown("""
Nous disposons à présent d'un ensemble de données nettoyé et nous l'avons analysé.
Nous sommes donc prêts à créer des modèles et à lancer des prévisions.
""")

st.divider()
st.header('Séparer le jeu de données: train et test')

st.markdown("""
Selon le principe de base du machine learning, un modèle doit s'entraîner sur ensemble de données dit _train set_.
Celui-ci est construit en sélectionnant une partie des données pour l'apprentissage.

Les autres données non utilisées pour l'apprentissage sont dites de _test_.
Nous allons les utiliser pour comparer les résultats de notre modèles et la réalité des ventes.

Il y a deux questions principales à se poser au moment de séparer le jeu de données.
- Combien de données souhaite-t-on réserver à l'entrainement ? et à la validation ?
- Comment sélectionnons-nous les données pour l'entraînement ?

**Questions**:
- Pouvez-vous devinez l'effet du nombre de données utilisées pour l'entrainement ? (Pensez aux cas extrêmes)
- Essayez d'imaginer différentes façons de sélectionner des données pour l'entraînement.
""")

st.divider()

st.markdown("""
Sélectionnez le **pourcentage** de données que vous souhaitez utiliser pour l'entrainement ainsi que la **méthodologie** de séparation des données.
""")


@st.cache_data
def load_data():
    df = pd.read_csv('data/bakery_sales_cleaned.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.index = df['datetime']

    df = df.resample(rule='1D').sum()
    df['datetime'] = df.index
    df.index = range(len(df))

    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day_of_week'] = df['datetime'].dt.day_of_week

    df['revenue_prev_day'] = df['full_price'].shift(1)
    df['quantity_prev_day'] = df['quantity'].shift(1)

    df['revenue_prev_week'] = df['full_price'].shift(7)
    df['quantity_prev_week'] = df['quantity'].shift(7)

    df = df.dropna(axis=0)

    return df


df = load_data()

perc_training = st.slider(
    "Pourcentage des données d'entrainement", 0, 100, 50)
split_methodo = st.selectbox(
    "Méthodologie pour séparer les données:",
    ['testing after training', 'testing before training', 'random'],
    label_visibility='visible',
)

if split_methodo == 'random':
    df['train'] = np.random.rand(len(df)) <= perc_training/100
elif split_methodo == 'testing after training':
    df['train'] = df.index <= perc_training/100*len(df)
elif split_methodo == 'testing before training':
    df['train'] = ~(df.index <= (1-perc_training/100)*len(df))

df['Split'] = df['train'].apply(lambda x: 'Train' if x else 'Test')

color_discrete_map = {'Train': '#636EFA',
                      'Test': '#EF553B', 'Predicted': '#00CC96'}

fig = go.Figure()
fig.add_trace(go.Scatter(x=df[df['train']]['datetime'], y=df[df['train']]
                         ['full_price'], mode='markers', name='Train data', marker_color='#636EFA'))
fig.add_trace(go.Scatter(x=df[~df['train']]['datetime'], y=df[~df['train']]
                         ['full_price'], mode='markers', name='Test data', marker_color='#EF553B'))
fig.update_layout(
    xaxis_title="Dates",
    yaxis_title="Revenue",
)
st.plotly_chart(fig, use_container_width=True)


with st.expander("Astuces"):
    st.markdown("""
- Dans le cas de série temporelles, il est généralement compliqué d'utiliser une méthode de séparation aléatoire.
En effet, s'il existe une corrélation entre deux dates proches, votre modèle pourrait s'appuyer sur des données non disponibles dans un contexte réel pour réaliser sa prédiction.
- Dans une même idée, utiliser des données plus récentes pour prédire des données plus anciennes ne fait pas de sens dans ce contexte.
- Il existe des méthodes plus évoluées de train/test dites de validation croisées.
- En pratique, nous réservons souvent 70 à 90 pourcent des données pour l'entraînement.
    """)

st.divider()

st.header('Entrainez un modèle et prédisez')

st.markdown("""
Pour entraîner un modèle de machine learning, vous devez généralement choisir les variables que vous utiliserez,
ainsi que le modèle que vous souhaitez entraîner.

Dans cet exemple, nous avons passablement allégé la partie liées aux variables appelées _features engineering_.
Celle-ci consiste à adapter, croiser et augmenter les variables présentes dans le jeu de données.

Nous vous passons également l'étape de sélection des paramètres du modèles car elle est assez technique.

Nous vous proposons ci-dessous de choisir le modèle ainsi que les variables  que vous souhaitez y mettre.
""")

features = st.multiselect(
    'Choisissez les variables:',
    ['year', 'month', 'day_of_week', 'quantity', 'revenue_prev_day',
        'quantity_prev_day', 'revenue_prev_week', 'quantity_prev_week'],
    ['day_of_week', 'month'])

ml_model = st.selectbox(
    "Choisissez le modèle de machine learning:",
    ['Linear Regression', 'Lasso', 'XGB', 'Decision Tree'],
    label_visibility='visible',
)

target = 'full_price'

X_train = df[df['train']][features]
y_train = df[df['train']][target]

X_test = df[~df['train']][features]
y_test = df[~df['train']][target]

if st.button('Train model', type='primary', use_container_width=True):

    if len(features) == 0:
        st.markdown("**Please choose at least one feature!**")
    else:

        if ml_model == 'Linear Regression':
            model = LinearRegression()
        elif ml_model == 'XGB':
            model = xgb.XGBRegressor(
                eta=0.1, gamma=5, max_depth=4, reg_lambda=0.2, reg_alpha=10, tree_method='exact')
        elif ml_model == 'Decision Tree':
            model = DecisionTreeRegressor()
        elif ml_model == 'Lasso':
            model = Lasso(alpha=0.5)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df[df['train']]['datetime'], y=df[df['train']]
                                 ['full_price'], mode='markers', name='Train data', marker_color='#636EFA'))
        fig.add_trace(go.Scatter(x=df[~df['train']]['datetime'], y=df[~df['train']]
                                 ['full_price'], mode='markers', name='Test data', marker_color='#EF553B'))
        fig.add_trace(go.Scatter(x=df[~df['train']]['datetime'], y=y_pred, mode='markers',
                                 marker_line_width=2, name='Predicted data', marker_color='#00CC96'))
        fig.update_layout(
            xaxis_title="Dates",
            yaxis_title="Revenue",
        )
        st.plotly_chart(fig, use_container_width=True)

        err_mae = mae(y_test, y_pred)

        st.markdown(f"""
        Vos prédictions confrontées à la réalité enregistrent une erreur MAE: **{err_mae:.2f}**.

        Cela signigie qu'en moyenne, vous réalisez une erreur d'environ {err_mae:.0f} € par jour (en positif ou en négatif).""")

        with st.expander("En savoir plus sur la MAE"):
            st.markdown("""
            Mean Absolute Error:            
            Cette erreur correspond à la somme de la valeur absolue des différences entre la valeur prédite et la réalité.
                        """)

        st.markdown("""    

       L'étape suivante consiste à répondre à certaines des questions suivantes et à revenir au début de cette page pour améliorer les performances du modèle :
        - Êtes-vous satisfait de vos résultats ? 
        - Pouvez-vous expliquer pourquoi vous avez obtenu ces résultats ? 
        - Avez-vous des idées pour améliorer le modèle ? 

        **Défi:** Notez votre score, changez les paramètres et trouvez un meilleure modèle que vos collègues!
        """)

with st.expander("Astuce"):
    st.markdown("""
Vous souhaitez en savoir plus sur un des modèles ? N'hésitez pas à consulter Chat GPT ou Wikipédia qui est une source riche sur le sujet.
""")

st.divider()

st.header('Notes et remarques finales')
st.markdown("""
- Il est possible de choisir une autre fonction d'erreur.
Pour cela, il faut discuter avec les experts métiers qui vous donneront leurs ressentis sur les erreurs les plus graves.
- Un grand défi dans le cadre d'analyse de séries temporelles est de n'utiliser que des données réellement disponibles au temps t.
    """)

st.divider()

st.header('**Félicitations**!')
st.markdown("""
Vous voici arrivé au terme de cette analyse guidée.
Nous espérons que cette expérience vous aura plu et que vous comprenez dorénvant mieux ce qui se passe dans la tête d'un data scientist.

Si vous souhaitez en savoir plus sur le sujet, n'hésitez pas à prendre contact avec nous sur notre
[page web.](https://swiss-sdi.ch/contact-swiss-sdi/)

Merci de nous avoir accompagné lors de cette analyse et à bientôt!
    """)
