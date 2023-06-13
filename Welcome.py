# coding=utf-8
""" """
# Copyright 2023, Swiss Statistical Design & Innovation Sàrl

import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Bienvenue",
)

st.title("Projet guidé de data science")

st.markdown("""
Dans cette page web interactive, vous avez la possibilité de jouer le rôle de data scientist à travers un exemple guidé.

Votre mission du jour est de **prévoir les ventes d'une boulangerie en France**.

Vous découvrirez les  étapes clés d'un projet de data science et apprendrez quelques points d'attention à ne pas rater lors d'une analyse.""")

st.image("https://www.parisperfect.com/blog/wp-content/uploads/2017/04/dcp-2016030716268-1024x645.jpg")

st.markdown("""
En particulier, pour mener à bien votre mission, vous traverserez les pas suivants:
1. **Data Cleaning**: Charger l'ensemble des données et éliminer les incohérences.
2. **Data Analysis**: Comprendre les données en les examinants.
3. **Forecasting**: Essayer différents modèles d'apprentissage automatique et différentes caractéristiques pour trouver le meilleur modèle possible.

Curieux de vous lancer dans votre première mission de data scientist ?

Dirigiez-vous sans attendre vers la première étape! 
""")


if st.button('Aller vers 🧹 Data Cleaning', type='primary', use_container_width=True):
    switch_page('Data_Cleaning')
