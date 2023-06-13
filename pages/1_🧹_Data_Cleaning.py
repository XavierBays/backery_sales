# coding=utf-8
""" """
# Copyright 2023, Swiss Statistical Design & Innovation Sàrl

import inspect

import numpy as np
import pandas as pd
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Data Cleaning",
    page_icon="🧹",
)

st.title('Nettoyez les données !')

st.markdown("""
Dans cette première partie, vous commencerez à travailler avec un nouveau jeu de données. Vous découvrirez quelles sont les premières étapes:
1. Charger l'ensemble de données
2. Nettoyer les données pour conserver les colonnes intéressantes
3. Détecter les problèmes dans l'ensemble de données et les résoudre

Avant decharger les données disponibles, il est judicieux de réfléchir à la problématique et aux données nécessaires.

**Questions:**
- Avez-vous bien compris ce que vous cherchez à prédire ?
- Quelles données pensez-vous nécessaires pour répondre à cette problématique ?
""")

with st.expander("Astuce"):
    st.markdown("""
    Prédire les ventes peut-être interprété de différentes manières. Cette problématiques n'est pas bien définie.
    - Cherche-t-on à prédire le chiffre d'affaire ?
    - Souhaite-t-on prédire la quantité de produits vendus ?
    - Par panier d'achat ou en absolu ? ou alors, par type d'article ?
    - ...

    Pour la suite, nous allons viser la prédiction du chiffre d'affaire.
    """)

st.divider()
st.header('Chargez les données')
st.markdown("""
La première étape consiste à charger les données.
On parle généralement d'un processus ETL pour _Extract_ , _Transform_, _Load_ .
Il faut en effet extraire la données (par exemple depuis son format excel), pour ensuite la transformer (cleaning) puis la placer au bon endroit.

Ne vous faites pas de souci pour l'extraction, on s'en est occupé pour vous!

Vous pouvez déjà visualiser les données grâce au _DataFrame_ ci-dessous.""")


@st.cache_data
def load_data():
    df = pd.read_csv('data/bakery_sales.csv')
    df = df.drop(columns='Unnamed: 0')

    return df


df = load_data()

st.dataframe(df, use_container_width=True)

st.markdown("""
Il est essentiel de commencer par observer vos données. Répondez aux **questions** suivantes:
- Quelles colonnes avez vous à disposition ?
- Combien y a-t-il de lignes ?
- Manque-t-il des données ?
- Pensiez-vous avoir plus d'information ?
- Quelle est la variable à prédire (variable réponse) ?
""")

st.divider()
st.header('Préparez les données')

st.markdown("""
Un des critères principaux de la qualité des données est le format.
Il est important que le manière de représenter l'information (type) soit cohérent avec ce qu'elle signifie.
Par exemple, nous souhaitons que les quantités soient des chiffres (et non des charactères).

Observons le type de chacune de nos colonnes:""")

st.dataframe(dict(df.dtypes))


with st.expander("Que signifient ces types ?"):
    st.markdown("""
    - float64: un chiffre à virgule (représenté par 64 bytes)
    - object: un texte ou un format non numérique indéfini
    """)

st.markdown("""
**Questions**:
- Qu'avez-vous remarqué ?
- Y a-t-il des problèmes de format ?""")

with st.expander("Réponse:"):
    st.markdown("""
    Il y a effectivement quelques problèmes à résoudre :
    - Le prix unitaire est au format "string", c'est-à-dire qu'il ne s'agit pas d'un nombre. Nous devons donc le modifier.
    - Les variables `date` et `time` doivent être dans un format `datetime` spécifique pour être analysées correctement.""")

st.markdown("""
Nous allons en profitez pour ajouter une colonne qui nous semble nécessaire: le **prix total**. On peut le faire en multipliant la variable `Quantity` par la variable `unit_price`.

Nous allons également supprimer une colonne qui n'est pas nécessaire à l'analyse: `ticket_number`.

Voici le résultat:
""")


@st.cache_data
def clean_df(data):
    """
    Clean the DataFrame for visualization

    Parameters
    ----------
    data : pd.DataFrame
        Original data

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for visualization
    """

    # Create a datetime variable and use the right type
    data['datetime'] = data['date'] + ' ' + data['time']
    data['datetime'] = pd.to_datetime(data['datetime'])

    # Change unit_price to a float
    data['unit_price'] = data['unit_price'].apply(
        lambda x: float(x[:-2].replace(',', '.')))

    # Compute the full price
    data['full_price'] = data['Quantity'] * data['unit_price']

    # Rename a column
    data = data.rename(columns={'Quantity': 'quantity'})

    # Select all final columns
    data = data[['datetime', 'article',
                 'quantity', 'unit_price', 'full_price']]

    return data


df = clean_df(df)

with st.expander("Vous voulez savoir à quoi ressemble le code python pour cette étape ? Cliquez ici."):
    st.code(inspect.getsource(clean_df), language='python')

st.dataframe(df, use_container_width=True)

st.markdown("""
Selon vous, la variable `ticket_number` était-elle vraiment inutile ? A quoi pourrait-elle nous servir ? 
""")

st.divider()
st.header('Repérez des problèmes dans les données')

st.markdown("""
Nous nous sommes beaucoup concentré sur les colonnes. Il est maintenant temps de nous attaquer aux lignes.
Y a-t-il des observations problématiques ? Essayez d'anticiper des erreurs potentielles:
- Quelles valeurs pourraient être erronnées pour la variable `quantity` ? 
- et pour la variable `unit_price` ?


Un bon réflex consiste généralement à vérifier les statistiques principales telles que la _moyenne_, le _minimum_, le _maximum_, etc.

La table ci-dessous vous présente pour trois colonnes les statistiques suivantes:
le nombre d'observations, la moyenne, la déviation standard, le miniumum, le quantile 25, la médiane, le quantile 75, le maximum.

Remarque: dans la tableau ci-dessous la séparation entre milliers et centaines est représentée par la virgule.""")

st.dataframe(df.describe(), use_container_width=True)

st.markdown("""
**Questions**:
- Que remarquez-vous ?
- Y a-t-il des valeurs incohérentes ?
- Vos hypothèses se sont-elles avérées ?
- La variable `full_price` est-elle problématique ?
""")

with st.expander("Réponse:"):
    st.markdown("""
    Grâce à ces quelques statistiques nous constatons les problèmes suivants :
    - La valeur minimale de la variable `quantité` est négative.
    - La valeur minimale de la variable `prix_unitaire` est de 0. Cela parait surprenant.

    Il est toutefois important de s'intéresser au contexte métier pour savoir si ces observations sont problématiques.
    - Quelles questions poseriez-vous au personnel de la boulagerie suite à ces observations ?

    """)

st.subheader("Nettoyez les problèmes de quantité")

st.markdown("""
Tout d'abord, abordons le problème de la `quantity`. La première idée pourrait être d'écarter simplement toutes les lignes si la `quantité` est négative.
Cependant, cela pourrait être une erreur.
Imaginez que le vendeur/la vendeuse en boulangerie souhaite simplement annuler une vente suite à une erreur de caisse.

Observons les données pour comprendre ce qu'il se passe.
""")


shown_samples = st.dataframe(df[df.quantity < 0].sample(
    10, random_state=13), use_container_width=True)

if st.button('Charger 10 autres exemples'):
    shown_samples.dataframe(
        df[df.quantity < 0].sample(10), use_container_width=True)

st.markdown("""
Vous pouvez maintenant sélectionner l'un des indices dans le DataFrame et le saisir dans la case ci-dessous.
Nous afficherons les 5 dernières ventes réalisées avant celle qui pose problème.""")

idx = st.number_input(
    'Insérez votre indice avec une quantité négative', value=158888)

st.dataframe(df.iloc[idx-5:idx+1], use_container_width=True)

st.markdown("""
**Questions**:
- Qu'observez-vous ?
- Quelle est votre hypothèse ?
- Comment peut-on résoudre ce problème ?""")

with st.expander("Réponse:"):
    st.markdown("""
    Vous l'aurez constaté, lorsqu'on trouve une vente avec une quantité négative, il existe une vente similaire avec une quantité positive juste avant.
    L'hypothèse de l'erreur de caisse semble s'avérérer. Il faudrait évidemment vérifier cela avec une personne de la boulangerie.

    Pour résoudre ce problème, nous devons donc annuler les deux ventes la positive et la négative.""")

st.markdown("""
**Question**:
- En quoi garder ces deux ventes pourrait-il poser problème pour l'analyse ? La somme de l'opération est finalement nulle...


Appliquons maintenant le nettoyage des données.""")


@st.cache_data
def cancel_negative_sales(data):
    """
    This function takes a DataFrame as an argument and removes any row
    where the quantity of sales is a negative number and its corresponding
    sale with a positive quantity.

    Parameters
    ----------
    data : pd.DataFrame
        Original data

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame without negative quantity
    """

    # List of indices we'll delete
    idx_to_del = []

    # Go through each row of the DF where the quantity is negative
    for i, row in data[data.quantity < 0].iterrows():

        # Get the rows before the current one
        tmp = data.iloc[0:i]

        # Get the rows with the same article and the same quantity as a positive number
        tmp = tmp[(tmp.article == row.article) &
                  (tmp.quantity == -row.quantity)]

        # Get the latest index, i.e. the closest to the current one
        idx = tmp.index[-1]

        # Save the indices that we will remove
        idx_to_del.append(idx)
        idx_to_del.append(i)

    # Remove all the rows we don't want
    data = data.drop(idx_to_del, axis=0)

    # Reset the indices
    data = data.reset_index(drop=True)

    return data


df = cancel_negative_sales(df)

with st.expander("Vous voulez savoir à quoi ressemble le code python pour cette étape ? Cliquez ici."):
    st.code(inspect.getsource(cancel_negative_sales), language='python')

st.markdown("""
Le problème de quantité négative devrait être résolu.
Pour en être absolument certain, observons à nouveau les statistiques principales.
""")

st.dataframe(df.describe(), use_container_width=True)

st.markdown("""
Nous voyons dorénavant que la quantité minimale est effectivement de 1.

Nous pouvons maintenant nous attaquer au problème du `unit_price`.
""")

st.subheader("Nettoyez les problèmes de prix")

st.markdown("""
Comme précédemment, observons les données quand le prix unitaire `unit_price` est égale à 0.
""")

st.dataframe(df[df.unit_price == 0], use_container_width=True)

nbr_sales_0 = len(df[df.unit_price == 0])

st.markdown(f"""
Il y a  {nbr_sales_0} ventes avec un prix unitaire de 0.

**Question**:
- Que feriez-vous de ces valeurs ?""")

with st.expander("Réponse:"):
    st.markdown("""
    - Certaines de ces valeurs concernent un article vide ou non défini représenté par un point. Nous pouvons les supprimer.
    - Le nombre de ventes nulles est négligeable par rapport à la taille du jeu de données. Nous pouvons également les supprimer.

    Il est tout de même important de relever ces observations à la boulangerie car derrière une petite anomalie de données se cache parfois un problème sérieux de processus.

    Supprimons donc ces données et affichons les statistiques une dernière fois.
    """)

st.markdown("""
Voici une dernière fois nos statistiques de base:
    """)

df = df[df.unit_price > 0]

# We can run this to save the cleaned file
# df.to_csv('data/bakery_sales_cleaned.csv', index=False)

st.dataframe(df.describe(), use_container_width=True)

st.markdown("""
**Félicitations!**
Le jeu de données est maintenant nettoyé de ses imprécisions.

En bonus, vous avez gagné en maturtié sur ces données. Vous savez dorénavant les données disponibles, leurs valeurs attendues, ...


Vous pouvez démarrer dans l'analyse des données!
""")

if st.button('Aller vers 📈 Data Analysis', type='primary', use_container_width=True):
    switch_page('Data_Analysis')
