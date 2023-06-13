# coding=utf-8
""" """
# Copyright 2023, Swiss Statistical Design & Innovation Sàrl

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_extras.switch_page_button import switch_page

st.set_page_config(
    page_title="Data Analysis",
    page_icon="📈",
)

st.title('Analysez les données !')

st.markdown("""
Maintenant que nous disposons d'un ensemble de données propres, nous pouvons commencer à les analyser.
Dans cette partie, il est essentiel de garder à l'esprit l'objectif final du projet: **prédire le chiffre d'affaire de la boulangerie**.
""")

st.divider()
st.header('Chargez les données nettoyées')
st.markdown("""
Nous chargeons d'abord les données nettoyées, puis nous examinons les distributions des différentes colonnes de données.

Observer tranquillement chacune des variables disponibles.""")


@st.cache_data
def load_data():
    df = pd.read_csv('data/bakery_sales_cleaned.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df


df = load_data()

possibilities = ['None']
possibilities.extend(df.columns)

variable_to_check = st.selectbox(
    "Sélectionner la variable que vous souhaitez voir?",
    possibilities,
    label_visibility='visible',
)

if variable_to_check == 'datetime':
    tmp = df.copy()

    # Plot the avg sales by hours
    tmp['hour'] = tmp['datetime'].dt.hour

    sales_per_hour = tmp.groupby('hour').size(
    ) / (tmp['datetime'].dt.date.nunique() * 24)
    sales_per_hour = sales_per_hour.reset_index(name='average_sales')

    fig_h = px.bar(sales_per_hour, x='hour', y='average_sales', title='Average Hourly sales', labels={
                   'hour': 'Hour', 'average_sales': 'Average Sales'})

    st.plotly_chart(fig_h, use_container_width=True)

    # Plot the avg sales by day of the week
    tmp['day_of_week'] = tmp['datetime'].dt.day_name()

    sales_per_day = tmp.groupby('day_of_week').size(
    ) / (tmp['datetime'].dt.date.nunique() * 7)
    sales_per_day = sales_per_day.reset_index(name='average_sales')

    days_of_week = ['Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday', 'Saturday', 'Sunday']
    sales_per_day['day_of_week'] = pd.Categorical(
        sales_per_day['day_of_week'], categories=days_of_week, ordered=True)
    sales_per_day = sales_per_day.sort_values('day_of_week')

    fig_d = px.bar(sales_per_day, x='day_of_week', y='average_sales', title='Average Sales per Day of the Week', labels={
                   'day_of_week': 'Day of the Week', 'average_sales': 'Average Sales'})

    st.plotly_chart(fig_d, use_container_width=True)

    # Plot the avg sales per month
    tmp['month'] = tmp['datetime'].dt.month_name()

    sales_per_month = tmp.groupby(
        'month').size() / tmp['datetime'].dt.year.nunique()
    sales_per_month = sales_per_month.reset_index(name='average_sales')

    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']
    sales_per_month['month'] = pd.Categorical(
        sales_per_month['month'], categories=months, ordered=True)
    sales_per_month = sales_per_month.sort_values('month')

    fig_h = px.bar(sales_per_month, x='month', y='average_sales', title='Average Sales per Month', labels={
                   'month': 'Month', 'average_sales': 'Average Sales'})

    st.plotly_chart(fig_h, use_container_width=True)

elif variable_to_check != 'None':
    fig = px.histogram(df, x=variable_to_check)

    fig.update_xaxes(categoryorder="total descending")
    fig.update_layout(bargap=0.1)

    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
**Questions**:
- Que constatez-vous à partir de ces différents graphes ?
- Voyez-vous une forme de saisonnalité dans les ventes ?
- Quel est l'article le plus vendu ?
- Combien de pièces d'un même article sont généralement achetées ensemble ?
- Dans quelle fourchette de prix se trouve l'article le plus vendu ?
- Quelle variable aura de l'influence sur le chiffre d'affaires ?
- Y a-t-il d'autres graphes que vous souhaiteriez voir ?
    """)

st.divider()

st.header('Concentration sur le prix')
st.markdown("""
Vu notre mission, nous devons évidemment nous concentrer un instant sur la variable du prix.

Pour les séries temporelles, nous commençons généralement par sélectionner un pas de temps pertinent. Quelle granularité souhaitez-vous fixer:
l'heure, le jour, la semaine, le mois, l'année ?

Vous pouvez explorer les différentes possibilités ci-dessous.
""")

dct = {
    'hour': '1H',
    'day': '1D',
    'week': '1W',
    'month': '1M',
    'year': '1Y'
}

time_steps = ['None']
time_steps.extend(dct.keys())

chosen_time_step = st.selectbox(
    "Quel pas de temps souhaitez-vous visualiser ?",
    time_steps,
    label_visibility='visible',
)

tmp = df[['datetime', 'full_price']].copy()
tmp.index = tmp['datetime']

if chosen_time_step != 'None':
    tmp = tmp.resample(rule=dct[chosen_time_step]).sum()
    tmp['datetime'] = tmp.index

    fig = px.scatter(tmp, x='datetime', y="full_price")
    fig.update_layout(
        xaxis_title="Dates",
        yaxis_title="Revenue",
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("""
Quel pas de temps choisiriez-vous ?
""")

with st.expander("Astuce"):
    st.markdown("""
    Il n'y a pas de réponse parfaite. Les ventes horaires semblent trop désordonnées pour être prédites correctement.
    D'autre part, les revenus annuels/mensuels ont trop peu de points de données pour être de bons candidats.
    Par conséquent, nous recommandons **les ventes quotidiennes ou hebdomadaires**.
    Elles semblent qu'il y ait un certain schéma.
    """)

st.divider()

st.markdown("""
    **Félicitations!**
    Vous avez maintenant une bonne intution sur vos données et les schémas sous jacents . Il est temps de passer à la **prédiction** !
    
    """)

if st.button('Aller vers 🤖 Forecasting', type='primary', use_container_width=True):
    switch_page('Forecasting')
