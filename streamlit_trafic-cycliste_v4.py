import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import date
import holidays
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from plotly.subplots import make_subplots
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf

#INTRO
st.image('velo3.jpg')
st.title("Trafic cycliste à Paris")
st.subheader("Variabilité spatio-temporelle et facteurs d'influence")
st.sidebar.title("Sommaire")
pages=["Introduction","Exploration", "DataVizualization", "Modélisation", "Conclusion et Perspéctives"]
page=st.sidebar.radio("Aller vers", pages)
st.sidebar.markdown("""
Auteurs : Sadio DIALLO et Taha EL FERRADI
Promo : DA Bootcamp janvier 2024.
                    """)

if page == pages[0] : 
  st.write("<span style='font-weight: bold; color: blue;'>CONTEXTE</span>", unsafe_allow_html=True)
  st.write("La Ville de Paris déploie depuis plusieurs années des compteurs à vélo permanents pour évaluer le développement de la pratique cycliste. Les compteurs sont situés sur des pistes cyclables et dans certains couloirs bus ouverts aux vélos."
         "Nous avons choisi d’utiliser ses données pour étudier la variabilité spatio-temporelle du trafic cycliste dans la capitale ainsi que les facteurs ayant un impact sur le trafic.")
  st.write("<span style='font-weight: bold; color: blue;'>OBJECTIF</span>", unsafe_allow_html=True)

  st.write("La mine d’informations à notre disposition avec ce dataset peut permettre d’étudier une multitude de sujets. Parmi ces thématiques, on peut citer:")
  st.markdown("""
  - Aménagement urbain et transport durable : planification des infrastructures cyclables.
  - Émissions de carbone et environnement : évaluation de l'impact environnemental.
  - Sécurité routière : identification des zones accidentogènes.
  - Politiques publiques : évaluation de l'impact des politiques en faveur du vélo.""")
  st.write("Toutes ces approches nous semblent intéressantes, c’est pourquoi nous avons choisi de compléter le dataset avec des variables complémentaires qui nous permettront de répondre aux questions suivantes :  Quels sont les axes les plus empruntés, quels jours, à quel moment de la journée, quel est l’impact des vacances, jours fériés et météo.")

  st.write("<span style='font-weight: bold; color: blue;'>CADRE</span>", unsafe_allow_html=True)
  st.write("Les données proviennent du site opendata.paris.fr et couvrent la période allant du <u><strong>1er janvier 2023</strong></u> au <u><strong>31 décembre 2023</strong></u>, avec 858 154 lignes pour 16 variables.", unsafe_allow_html=True)

#CALCULS DES DF
fr_holidays = holidays.France(years=2023)
vacances_scolaires = pd.to_datetime([
    '2023-01-01', '2023-01-02', 
    '2023-02-18', '2023-02-19', '2023-02-20', '2023-02-21', '2023-02-22', '2023-02-23', '2023-02-24', '2023-02-25', '2023-02-26', '2023-02-27', '2023-02-28',
    '2023-04-22', '2023-04-23', '2023-04-24', '2023-04-25', '2023-04-26', '2023-04-27', '2023-04-28', '2023-04-29', '2023-04-30',
    '2023-07-08', '2023-07-09', '2023-07-10', '2023-07-11', '2023-07-12', '2023-07-13', '2023-07-14', '2023-07-15', '2023-07-16', '2023-07-17', '2023-07-18', '2023-07-19', '2023-07-20', '2023-07-21', '2023-07-22', '2023-07-23', '2023-07-24', '2023-07-25', '2023-07-26', '2023-07-27', '2023-07-28', '2023-07-29', '2023-07-30', '2023-07-31',
    '2023-10-21', '2023-10-22', '2023-10-23', '2023-10-24', '2023-10-25', '2023-10-26', '2023-10-27', '2023-10-28', '2023-10-29', '2023-10-30',
    '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-27', '2023-12-28', '2023-12-29', '2023-12-30', '2023-12-31'
])
dates_vacances_noel = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-27', '2023-12-28', '2023-12-29', '2023-12-30', '2023-12-31'])

def map_season(month):
    if month in [12, 1, 2]:  # Décembre, Janvier, Février -> Hiver
        return 'Hiver'
    elif month in [3, 4, 5]:  # Mars, Avril, Mai -> Printemps
        return 'Printemps'
    elif month in [6, 7, 8]:  # Juin, Juillet, Août -> Été
        return 'Été'
    else:  # Septembre, Octobre, Novembre -> Automne
        return 'Automne'


@st.cache_data
def load_df(filepath, sep):
  df = pd.read_csv(filepath, sep=sep)
  df = df.rename(columns={'Comptage horaire': 'Comptage_horaire'})
  df = df.rename(columns={'Nom du site de comptage': 'Nom_du_site_de_comptage'})
  df = df.rename(columns={'Identifiant du compteur': 'Identifiant_du_compteur'})
  df = df.rename(columns={'Nom du compteur': 'Nom_du_compteur'})
  df = df.rename(columns={'Identifiant du site de comptage': 'Identifiant_du_site_de_comptage'})
  df = df.rename(columns={'Date et heure de comptage': 'Date_et_heure_de_comptage'})
  df = df.rename(columns={"Date d'installation du site de comptage": "Date_d_installation_du_site_de_comptage"})
  df = df.rename(columns={'Coordonnées géographiques': 'Coordonnées_géographiques'})
  df = df.rename(columns={'Identifiant technique compteur': 'Identifiant_technique_compteur'})
  df = df.rename(columns={'Lien vers photo du site de comptage': 'Lien_vers_photo_du_site_de_comptage'})
  df['Identifiant_du_site_de_comptage'] = df['Identifiant_du_site_de_comptage'].astype(str)
  df["Date_et_heure_de_comptage"]=pd.to_datetime(df["Date_et_heure_de_comptage"],utc=True)
  df=df.drop(["Lien_vers_photo_du_site_de_comptage", "Identifiant_technique_compteur","ID Photos", "test_lien_vers_photos_du_site_de_comptage_", "id_photo_1", "url_sites", "type_dimage"],axis=1)
  df[['Longitude', 'Latitude']] = df['Coordonnées_géographiques'].str.split(',', expand=True)
  df['Année'] = df['Date_et_heure_de_comptage'].dt.year
  df['Mois'] = df['Date_et_heure_de_comptage'].dt.month
  df['Jour'] = df['Date_et_heure_de_comptage'].dt.day
  df['Heure'] = df['Date_et_heure_de_comptage'].dt.hour
  df['Latitude'] = pd.to_numeric(df['Latitude'])
  df['Longitude'] = pd.to_numeric(df['Longitude'])
  df['Date'] = df['Date_et_heure_de_comptage'].dt.date
  df['Date'] = pd.to_datetime(df['Date'])
  df['jour_de_la_semaine'] = df['Date_et_heure_de_comptage'].dt.dayofweek
  df['week_end'] = df['jour_de_la_semaine'].apply(lambda x: 1 if x >= 5 else 0)
  df['Est_ferie'] = df['Date_et_heure_de_comptage'].dt.date.isin(fr_holidays)
  df['Vacances_scolaires'] = df['Date'].isin(vacances_scolaires)
  df['vacances_aout'] = (df['Date'].dt.month == 8).astype(int)
  return df


@st.cache_data     # A UTILISER AVANT DE DROP VACANCES SCOLAIRES POUR VACANCES NOEL
def load_df1(df1):
  df1=df.drop(['Vacances_scolaires'], axis = 1)
  df1['vacances_noel'] = df['Date'].isin(dates_vacances_noel).astype(int)
  df_meteo = pd.read_csv("C:\\Users\\sadio\\Desktop\\Data analyst\\Projet\\météo.csv", sep=",")
  df_meteo['Date'] = pd.to_datetime(df_meteo['Date'], format="%d/%m/%Y")
  df_meteo['Précipitations'] = df_meteo['Précipitations'].str.replace(',', '.', regex=False).astype(float)
  df_meteo['Température maximale'] = df_meteo['Température maximale'].str.replace(',', '.', regex=False).astype(float)
  df_meteo['Température minimale'] = df_meteo['Température minimale'].str.replace(',', '.', regex=False).astype(float)
  df_meteo['Température moyenne'] = df_meteo['Température moyenne'].str.replace(',', '.', regex=False).astype(float)
  df1=pd.merge(df1, df_meteo, on='Date', how='left')
  df1['Saison'] = df1['Mois'].apply(map_season)
  conditions = [
    (df1['Précipitations'] > 0.5) & (df1['Précipitations'] <= 2),
    (df1['Précipitations'] > 2) & (df1['Précipitations'] <= 7.6),
    (df1['Précipitations'] > 7.6)]
  values = ['pluie faible', 'pluie modérée', 'forte pluie']
  df1['pluie_intensity'] = np.select(conditions, values, default='Pas de pluie')
  df1 = df1.rename(columns={'Température maximale': 'Temperature_maximale'})
  df1 = df1.rename(columns={'Température minimale': 'Temperature_minimale'})
  df1 = df1.rename(columns={'Température moyenne': 'Temperature_moyenne'})
  df1['Saison'] = df1['Mois'].apply(map_season)
  df1['pluie_intensity'] = np.select(conditions, values, default='Pas de pluie')
  conditions = [
    (df1['Précipitations'] > 0.5) & (df1['Précipitations'] <= 2),
    (df1['Précipitations'] > 2) & (df1['Précipitations'] <= 7.6),
    (df1['Précipitations'] > 7.6)
]
  values = ['pluie faible', 'pluie modérée', 'forte pluie']
  return df1


df=load_df('C:\\Users\\sadio\\Desktop\\Data analyst\\Projet\\comptage-velo-donnees-compteurs.csv', sep =';')

df1=load_df1(df)

#EXPLORATION
if page == pages[1]:
    st.write("### Exploration des données")

    st.write("<span style='font-weight: bold; color: skyblue;'>Valeurs manquantes</span>", unsafe_allow_html=True)

    st.write("66 136 valeurs sont manquantes. Elles concernent 6 variables, toutes liées aux photos prises des compteurs. Nous décidons de supprimer les colonnes du DataFrame car elles ne nous sont pas utiles pour notre analyse.")
    st.write("\n")

    st.write("Après une étude rapide du dataset, il nous est rapidement venu à l'esprit d’abandonner les variables liées aux photos des compteurs car celles-ci n’avaient aucune pertinence pour notre étude, à savoir :")
    st.markdown(""" 
- "Lien vers photo du site de comptage"
- "ID Photos"unsafe_allow_html="
- "test_lien_vers_photos_du_site_de_comptage_"
- "id_photo_1"
- "url_sites"
- "type_dimage" """)

    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Harmonisation des données</span>", unsafe_allow_html=True)

    st.write("Nous avons décidé d’harmoniser les noms des variables en remplaçant tous les espaces par des underscores. Afin de pouvoir exploiter au mieux les variables temporelles, nous les avons retravaillées pour créer de nouvelles variables (jour, mois, année, heure,  jour de la semaine, week-end, jour férié, saison, vacances scolaires, etc..).")
    st.write("De même pour la variable “coordonnées_géographiques” que nous avons splité en 2 variables ; l’une “longitude” et l’autre “latitude”. Nous avons également modifié le type de la variable “identifiant_du_site” pour le définir en tant que “str” et avons converti la variable “Date_et_heure_de_comptage” sous format “datetime”.")
    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Extrait de notre DataFrame</span>", unsafe_allow_html=True)

    st.dataframe(df.head())


def show_data(df):
    st.dataframe(df.head())

    st.dataframe(df.head()) 


#DATAVIZ
if page == pages[2]:
  #st.write("### Visualisations")
  if st.checkbox("VISUALISATIONS GEOGRAPHIQUES") : 

    st.write("<span style='font-weight: bold; color: blue;'>VISUALISATIONS GEOGRAPHIQUES</span>", unsafe_allow_html=True)

    import folium
    from streamlit_folium import folium_static

    df_subset = df[['Nom_du_site_de_comptage', 'Latitude', 'Longitude']]

  # Supprimer les doublons pour obtenir les 75 compteurs uniques avec leurs coordonnées
    df_unique = df_subset.drop_duplicates(subset=['Nom_du_site_de_comptage'])
  # Créer une carte centrée sur Paris
    carte = folium.Map(location=[48.8566, 2.3522], zoom_start=12)

  # Ajouter des marqueurs pour chaque compteur
    for index, row in df_unique.iterrows():
        nom_compteur = row['Nom_du_site_de_comptage']
        longitude = row['Longitude']
        latitude = row['Latitude']
      
      # Ajouter un marqueur pour chaque compteur
        folium.Marker([longitude, latitude], popup=nom_compteur).add_to(carte)

  # Afficher la carte dans Streamlit
    st.write("<span style='font-weight: bold; color: skyblue;'>Carte des compteurs de vélo à Paris.</span>", unsafe_allow_html=True)
  
    folium_static(carte)

    st.write('Remarques:')
    st.markdown("""
    - Les compteurs sont répartis de manière plus ou moins dense. 
    - Les arrondissements du Nord (17, 18 et 19ème) sont peu couverts par le réseau de comptage.
    - Les grands axes des quais de Seine et du Nord au Sud sont biens couverts
                """)

      # Calculer le comptage horaire moyen pour chaque lieu de comptage
    mean_by_location = df[df['Année'] == 2023].groupby('Nom_du_site_de_comptage')['Comptage_horaire'].mean().nlargest(10)

    # Créer le graphique avec Matplotlib
    plt.figure(figsize=(10, 6))
    mean_by_location.plot(kind='bar', color='skyblue')

    # Titres et libellés
    plt.title('Top 10 des lieux de comptage avec le comptage horaire moyen le plus élevé')
    plt.xlabel('Lieu de comptage')
    plt.ylabel('Comptage horaire moyen')

    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')

    # Afficher les valeurs pour chaque barre
    for i, val in enumerate(mean_by_location):
        plt.text(i, val, str(round(val, 2)), ha='center', va='bottom')

    # Ajustement du layout
    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
        # Calculer le comptage horaire moyen pour chaque lieu de comptage
    mean_by_location = df[df['Année'] == 2023].groupby('Nom_du_site_de_comptage')['Comptage_horaire'].mean().nsmallest(10)

    # Créer le graphique avec Matplotlib
    plt.figure(figsize=(10, 6))
    mean_by_location.plot(kind='bar', color='red')

    # Titres et libellés
    plt.title('Top 10 des lieux de comptage avec le comptage horaire moyen le moins élevé')
    plt.xlabel('Lieu de comptage')
    plt.ylabel('Comptage horaire moyen')

    # Rotation des étiquettes de l'axe x pour une meilleure lisibilité
    plt.xticks(rotation=45, ha='right')

    # Afficher les valeurs pour chaque barre
    for i, val in enumerate(mean_by_location):
        plt.text(i, val, str(round(val, 2)), ha='center', va='bottom')

    # Ajustement du layout
    plt.tight_layout()

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    st.write("Les axes les plus empruntés traversent des quartiers dynamiques, tandis que les moins fréquentés sont situés près de zones résidentielles.")

    


  if st.checkbox("VISUALISATIONS TEMPORELLES") :
    st.write("<span style='font-weight: bold; color: blue;'>VISUALISATIONS TEMPORELLES</span>", unsafe_allow_html=True)
    st.write("<span style='font-weight: bold; color: skyblue;'>Visualisations des données sur une journée</span>", unsafe_allow_html=True)
    df_hourly_count = df[df['Année'] == 2023].groupby('Heure')['Comptage_horaire'].mean()

    # Tracer la répartition du comptage horaire par heure
    plt.figure(figsize=(10, 6))
    bars = plt.bar(df_hourly_count.index, df_hourly_count.values)
    plt.xlabel('Heure')
    plt.ylabel('Comptage horaire moyen')
    plt.title('Répartition du comptage horaire moyen')

    # Ajouter les valeurs au-dessus de chaque barre
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval, round(yval, 2), va='bottom')

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    st.write("Deux pics de trafic se distinguent entre 7-8 heures et 17-18 heures, correspondant aux déplacements domicile-travail.")

    # Heatmap des comptages moyens par heure et site
    pivot_table = df[df['Année'] == 2023].pivot_table(index='Nom_du_site_de_comptage', columns='Heure', values='Comptage_horaire', aggfunc='mean')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, cmap='coolwarm')
    plt.title('Heatmap des comptages moyens par heure et site')
    plt.xlabel('Heure')
    plt.ylabel('Nom du site de comptage')

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    st.write("La heatmap ci-dessus nous permet également de visualiser les horaires de fortes fréquentations. On constate que les pics journaliers se visualisent sur les différents sites de comptage.")
    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Visualisations des données sur une semaine</span>", unsafe_allow_html=True)
    
# Calculer la moyenne du comptage horaire par jour de la semaine
    mean_by_weekday = df.groupby('jour_de_la_semaine')['Comptage_horaire'].mean()

    # Définir l'ordre des jours de la semaine
    jours_semaine = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']

    # Tracer le graphique
    plt.figure(figsize=(10, 6))
    mean_by_weekday.plot(kind='bar', color='skyblue')
    plt.title('Circulation moyenne par jour de la semaine')
    plt.xlabel('Jour de la semaine')
    plt.ylabel('Comptage horaire moyen')

    # Ajouter des annotations pour afficher les valeurs de chaque jour
    for i, val in enumerate(mean_by_weekday):
        plt.text(i, val, str(round(val, 2)), ha='center', va='bottom')

    # Spécifier les étiquettes des ticks sur l'axe des x
    plt.xticks(range(len(jours_semaine)), jours_semaine, rotation=45)

    # Afficher le graphique
    st.pyplot(plt)

    st.write("L'analyse des données révèle une tendance intéressante concernant la fréquentation des lieux de comptage durant la semaine. Les week-ends affichent une fréquentation moindre par rapport aux autres jours, ce qui est attendu compte tenu des activités réduites pendant ces jours de repos. Par ailleurs, il est remarquable que les lundis et vendredis enregistrent également une fréquentation un peu plus basse. Cette observation pourrait être liée à une pratique accrue du télétravail en France, notamment les lundis et vendredis, comme le suggère une étude menée par la Fondation Jean-Jaurès en 2022.")
    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Différence de fréquentation entre les jours de semaines et le week-end</span>", unsafe_allow_html=True)




    st.write("Les week-ends sont souvent associés à un mode de vie plus détendu, avec moins de déplacements liés aux obligations professionnelles ou scolaires. Cela se traduit par une diminution du trafic de vélo pendant ces jours, surtout pendant les heures habituelles de travail.")

    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Comparaison entre jours fériés et jours non fériés</span>", unsafe_allow_html=True)
    st.write("Pour ajouter plus d’explications à l’argument de la tendance des personnes à utiliser le vélo pour se déplacer vers leur travail ou lieu d’études, nous décidons d'effectuer une comparaison entre les jours fériés et jours non fériés.")

    # Agréger les données par jour férié
    df_daily_mean = df[df['Année'] == 2023].groupby('Est_ferie')['Comptage_horaire'].mean()

    # Tracer les résultats avec Matplotlib
    plt.figure(figsize=(10, 6))
    bars = df_daily_mean.plot(kind='bar', color=['blue', 'green'])

    # Ajouter les valeurs au-dessus de chaque barre
    for bar in bars.patches:
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.05,  # Position X
                bar.get_height() + 0.1,                   # Position Y
                f"{bar.get_height():.2f}",                # Texte à afficher (avec deux décimales)
                ha='center', va='bottom',                 # Alignement horizontal et vertical
                fontsize=10)                              # Taille de la police

    plt.title('Comparaison des comptages horaires moyens entre jours fériés et jours non fériés')
    plt.xlabel('Type de jour (0: Jours non fériés, 1: Jours fériés)')
    plt.ylabel('Comptage horaire moyen')
    plt.xticks([0, 1], ['Jours non fériés', 'Jours fériés'], rotation=0)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    

    mean_weekday = df[df['Est_ferie'] == False]['Comptage_horaire'].mean()
    mean_holiday = df[df['Est_ferie'] == True]['Comptage_horaire'].mean()

    # Calculer la différence en pourcentage
    diff_percentage = ((mean_holiday - mean_weekday) / mean_weekday) * 100
    st.write("Différence en pourcentage de la circulation entre les jours fériés et les jours de la semaine :", diff_percentage)
    st.write("\n")
    st.write("La différence significative dans le comptage horaire moyen entre les jours de la semaine et entre les jours fériés et les jours non fériés accentue l’idée que les cyclistes utilisent leurs vélos pour les déplacements domicile - Travail / Ecole.")

    st.write("\n")
    st.write("<span style='font-weight: bold; color: skyblue;'>Comparaison du trafic de vélo entre les périodes de vacances scolaires et non vacances scolaires.</span>", unsafe_allow_html=True)
    st.write("Après avoir examiné les données par jour, notre attention se tourne désormais vers l'impact des vacances scolaires sur le comptage horaire.")
    # Agréger les données par période de vacances scolaires ou non
    aggregated_data = df[df['Année'] == 2023].groupby('Vacances_scolaires')['Comptage_horaire'].mean()

    # Tracer le graphique comparatif avec Matplotlib
    plt.figure(figsize=(10, 6))
    bars = aggregated_data.plot(kind='bar', color=['green', 'blue'])
    plt.title('Comparaison du trafic de vélo entre les périodes de vacances scolaires et non vacances scolaires (2023)')
    plt.xlabel('Période')
    plt.ylabel('Moyenne de comptage horaire')
    plt.xticks([0, 1], ['Hors vacances scolaires', 'Pendant vacances scolaires'], rotation=0)

    # Ajouter les annotations pour afficher les valeurs de chaque barre
    for i, val in enumerate(aggregated_data):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)
    st.write("Les vacances scolaires montrent une légère diminution du trafic, mais compensée par des loisirs en extérieur.")
     
    vacances_data = {
        'Date_debut': ['2023-01-01', '2023-02-18', '2023-04-22', '2023-07-08', '2023-10-21', '2023-12-23'],
        'Date_fin': ['2023-01-02', '2023-03-05', '2023-05-08', '2023-09-03', '2023-11-05', '2023-12-31']
    }
    vacances_df = pd.DataFrame(vacances_data)

    # Convertir les colonnes en objets datetime
    vacances_df['Date_debut'] = pd.to_datetime(vacances_df['Date_debut'])
    vacances_df['Date_fin'] = pd.to_datetime(vacances_df['Date_fin'])

    # Créer une liste de toutes les dates de vacances entre la date de début et de fin incluses
    dates_vacances = []
    for _, row in vacances_df.iterrows():
        dates_vacances.extend(pd.date_range(start=row['Date_debut'], end=row['Date_fin']).tolist())

    # Définir les périodes de vacances de Noël
    dates_vacances_noel = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-12-23', '2023-12-24', '2023-12-25', '2023-12-26', '2023-12-27', '2023-12-28', '2023-12-29', '2023-12-30', '2023-12-31'])

    # Créer une colonne 'vacances_noel' indiquant les jours de vacances de Noël
    df['vacances_noel'] = df['Date'].isin(dates_vacances_noel).astype(int)
  
    # Marquage des jours de vacances scolaires dans le DataFrame principal
    df['Jour_vacances_scolaires'] = df['Date'].apply(lambda x: 1 if x in dates_vacances else 0)
    df['Jour_vacances_scolaires'] = df['Date'].apply(lambda x: x in dates_vacances).astype(int)
    # Calculer la moyenne de comptage pour les jours de non vacances scolaires et les jours de vacances de Noël
    moyenne_non_vacances = df[df['Année'] == 2023][df[df['Année'] == 2023]['Jour_vacances_scolaires'] == 0]['Comptage_horaire'].mean()
    moyenne_vacances_noel = df[df['Année'] == 2023][df[df['Année'] == 2023]['vacances_noel'] == 1]['Comptage_horaire'].mean()

    # Créer un graphique en barres pour comparer les moyennes avec Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(['Non vacances scolaires', 'Vacances de Noël'], [moyenne_non_vacances, moyenne_vacances_noel], color=['green', 'red'])
    plt.title('Comparaison du comptage horaire moyen entre les jours de non vacances scolaires et les jours de vacances de Noël')
    plt.xlabel('Type de jour')
    plt.ylabel('Moyenne de comptage horaire')

    # Ajouter les annotations pour afficher les valeurs de chaque barre
    for i, val in enumerate([moyenne_non_vacances, moyenne_vacances_noel]):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    # Calculer la moyenne de comptage pour les jours de non vacances scolaires et les jours de vacances d'août
    moyenne_non_vacances = df[df['Année'] == 2023][df[df['Année'] == 2023]['Jour_vacances_scolaires'] == 0]['Comptage_horaire'].mean()
    moyenne_vacances_aout = df[df['Année'] == 2023][df[df['Année'] == 2023]['vacances_aout'] == 1]['Comptage_horaire'].mean()

    # Créer un graphique en barres pour comparer les moyennes avec Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(['Non vacances scolaires', 'Vacances d\'août'], [moyenne_non_vacances, moyenne_vacances_aout], color=['green', 'blue'])
    plt.title('Comparaison du comptage horaire moyen entre les jours de non vacances scolaires et les jours de vacances d\'août')
    plt.xlabel('Type de jour')
    plt.ylabel('Moyenne de comptage horaire')

    # Ajouter les annotations pour afficher les valeurs de chaque barre
    for i, val in enumerate([moyenne_non_vacances, moyenne_vacances_aout]):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    # Calculer la moyenne de comptage pour les jours de vacances de Noël et les jours de vacances d'août
    moyenne_vacances_noel = df[df['Année'] == 2023][df[df['Année'] == 2023]['vacances_noel'] == 1]['Comptage_horaire'].mean()
    moyenne_vacances_aout = df[df['Année'] == 2023][df[df['Année'] == 2023]['vacances_aout'] == 1]['Comptage_horaire'].mean()

    # Créer un graphique en barres pour comparer les moyennes avec Matplotlib
    plt.figure(figsize=(8, 6))
    plt.bar(['Vacances de Noël', 'Vacances d\'août'], [moyenne_vacances_noel, moyenne_vacances_aout], color=['red', 'blue'])
    plt.title('Comparaison du comptage horaire moyen entre les jours de vacances de Noël et les jours de vacances d\'août')
    plt.xlabel('Type de vacances')
    plt.ylabel('Moyenne de comptage horaire')

    # Ajouter les annotations pour afficher les valeurs de chaque barre
    for i, val in enumerate([moyenne_vacances_noel, moyenne_vacances_aout]):
        plt.text(i, val, f"{val:.2f}", ha='center', va='bottom', fontsize=10)

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    st.write("Une nette diminution du trafic est observée pendant les vacances de Noël par rapport à celles d'août, phénomène expliqué par divers facteurs.")
    st.write("La baisse de trafic durant les vacances de Noël comparé à août s'explique par les congés, la fermeture des commerces, et les conditions météorologiques plus froides.")
  
    st.write("\n")
    st.write("Par ailleurs, les conditions météorologiques peuvent également jouer un rôle, les températures étant généralement plus froides pendant les vacances de Noël par rapport au mois d'août.")




  if st.checkbox("VISUALISATIONS METEOROLOGIQUES") : 

    st.write("<span style='font-weight: bold; color: blue;'>VISUALISATIONS METEOROLOGIQUES</span>", unsafe_allow_html=True)
    st.write('\n')
    st.write("Après avoir pris en compte l'influence potentielle des conditions météorologiques sur le trafic cycliste, nous avons entrepris de recueillir et d'analyser les données météorologiques de Paris pour l'année 2023.")

    st.write("- Création d'une base de données météorologiques, comprenant les températures minimales, maximales et moyennes pour chaque jour de l'année 2023, ainsi que les données de précipitations. Nous la fusionnons avec notre base de donnée principale")
    st.write("<span style='font-weight: bold; color: skyblue;'>Extrait du nouveau DataFrame</span>", unsafe_allow_html=True)   
 
    st.dataframe(df1.head())
    st.write('\n')
    st.write("<span style='font-weight: bold; color: skyblue;'>Impact des températures</span>", unsafe_allow_html=True)
    
    # Calculer le comptage horaire moyen par saison
    count_by_season = df1[df1['Année'] == 2023].groupby('Saison')['Comptage_horaire'].mean()
    # Tracer le graphique avec Matplotlib
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = count_by_season.plot(kind='bar', color='skyblue', ax=ax)
    plt.title('Comptage horaire moyen par saison')
    plt.xlabel('Saison')
    plt.ylabel('Comptage horaire moyen')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Ajouter les annotations des valeurs au-dessus de chaque barre
    for bar in bars.patches:
        ax.text(bar.get_x() + bar.get_width() / 2 - 0.1,  # Position X
                bar.get_height() + 0.1,                   # Position Y
                f"{bar.get_height():.2f}",                # Texte à afficher (avec deux décimales)
                ha='center', va='bottom',                 # Alignement horizontal et vertical
                fontsize=10)                              # Taille de la police

    # Afficher le graphique dans Streamlit
    st.pyplot(fig)

    st.write("Les données météorologiques indiquent une préférence pour le vélo par temps plus chaud, corroborant le comportement observé.")
    st.write("Confirmée par le test de correlation entre le comptage horaire moyen par saison et la température moyenne par saison (0.6690313062940515)")
    st.write('\n')
  
# Calculer le comptage horaire moyen pour chaque valeur de l'intensité de la pluie
    mean_count_by_intensity = df1[df1['Année'] == 2023].groupby('pluie_intensity')['Comptage_horaire'].mean()

    # Trier les données en fonction du comptage horaire moyen
    sorted_intensity = mean_count_by_intensity.sort_values()

    # Tracer le graphique en utilisant seaborn
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=sorted_intensity.index, y=sorted_intensity.values, order=sorted_intensity.index, palette='Blues')
    plt.title('Comparaison des comptages en fonction de l\'intensité de la pluie')
    plt.xlabel('Intensité de la pluie')
    plt.ylabel('Comptage horaire moyen')

    # Ajouter les valeurs au-dessus des barres
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', xytext=(0, 10), textcoords='offset points')

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    st.write("Cette classification est basée sur les valeurs de la colonne 'Précipitations' de notre DataFrame, selon des seuils préétablis. Les seuils utilisés sont 0.5 pour la pluie faible, 2 pour la pluie modérée et 7.6 pour la forte pluie, conformément aux normes de l'Organisation météorologique mondiale.")
    st.write("Constatation cohérente avec l'idée commune selon laquelle les cyclistes sont moins enclins à utiliser leur vélo par temps de pluie en raison des conditions météorologiques défavorables et des risques accrus pour leur sécurité.")


  

  if st.checkbox("VISUALISATION GLOBALE") : 

    st.write("<span style='font-weight: bold; color: blue;'>VISUALISATION GLOBALE</span>", unsafe_allow_html=True)
    from datetime import datetime
    import matplotlib.pyplot as plt
    from matplotlib.dates import MonthLocator, DateFormatter
    import locale
    locale.setlocale(locale.LC_TIME, 'fr_FR.UTF-8')
    # Convertir les dates en objets datetime
    date_august = datetime.strptime('2023-08-15', '%Y-%m-%d')
    date_christmas = datetime.strptime('2023-12-25', '%Y-%m-%d')
    date_summer_start = datetime.strptime('2023-06-01', '%Y-%m-%d')
    date_summer_end = datetime.strptime('2023-08-31', '%Y-%m-%d')
    date_fall_start = datetime.strptime('2023-09-01', '%Y-%m-%d')
    date_fall_end = datetime.strptime('2023-11-30', '%Y-%m-%d')
    df_daily_count = df1[df1['Année'] == 2023].groupby(df1['Date_et_heure_de_comptage'].dt.date)['Comptage_horaire'].sum()

    # Tracer la tendance temporelle
    plt.figure(figsize=(10, 6))
    plt.plot(df_daily_count.index, df_daily_count.values)
    plt.xlabel('Date')
    plt.ylabel('Comptage horaire total')
    plt.title("Tendance temporelle du comptage horaire de l'année 2023")
    plt.xticks(rotation=45)

    # Affichage du nom des mois
    plt.gca().xaxis.set_major_locator(MonthLocator())
    plt.gca().xaxis.set_major_formatter(DateFormatter('%b'))

    plt.grid(True)

    # Annotation pour les vacances d'août
    plt.annotate('Vacances d\'août', xy=(date_august, 70000), xytext=(date_august, 45000),
                arrowprops=dict(facecolor='red', arrowstyle='->'))

    # Annotation pour Noël
    plt.annotate('Vacances Noël', xy=(date_christmas, 60000), xytext=(date_christmas, 40000),
                arrowprops=dict(facecolor='blue', arrowstyle='->'))

    plt.annotate("Période d'Été", xy=(date_summer_start, 300000), xytext=(date_summer_start, 300000))

    # Annotation pour l'automne
    plt.annotate('Automne', xy=(date_fall_start, 300000), xytext=(date_fall_start, 300000))

    # Afficher le graphique dans Streamlit
    st.pyplot(plt)

    st.write("Ce graphique, basé sur les données de l'année 2023, met en évidence plusieurs tendances du trafic de vélo à Paris:")
    st.write("\n")
       
    st.markdown("""
    - Intensité du trafic aux beaux jours (mai à septembre) : Les conditions météorologiques clémentes incitent plus de gens à faire du vélo, augmentant ainsi le trafic cycliste.
    - Baisse marquée du trafic en août (vacances d'été) : La diminution du trafic pendant les vacances estivales résulte de nombreux départs en vacances hors de la ville.
    - Baisse du trafic pendant les fêtes de fin d'année : Les activités urbaines réduites pendant les fêtes de fin d'année se traduisent par une baisse du trafic cycliste.
    - Variations du trafic selon les jours de la semaine : Une différence significative entre les jours de semaine et les week-ends, indiquant une utilisation plus fréquente du vélo en semaine, probablement due aux déplacements professionnels.""")
    st.write("\n")
    st.write("Après cette analyse, la prochaine étape consiste à passer à la modélisation avec des algorithmes de machine learning.")  






#MODELISATION
if page == pages[3] : 
  st.markdown("---")
  st.write("### Modélisations")
  st.write("<span style='font-weight: bold; color: blue;'>Objectif :</span> Prédire la densité du trafic.", unsafe_allow_html=True)
  st.write("#### Régression linéaire")
  st.write("<span style='font-weight: bold; color: blue;'>Variables :</span> Toutes celles jugées pertinentes et non redondantes.", unsafe_allow_html=True)
  st.write("<span style='font-weight: bold; color: blue;'>Entraînement</span> sur les relevés de l'année 2023 et prédictions pour janvier 2024.", unsafe_allow_html=True)
  st.write("<span style='font-weight: bold; color: blue;'>Performance :</span> **R²**=0,10, **MAE** = 63 et **RMSE** =98.", unsafe_allow_html=True)
  
 
  #Nettoyage
  df1_LR=df1.drop(["Nom_du_compteur", "Identifiant_du_site_de_comptage", "Nom_du_site_de_comptage", "Date_d_installation_du_site_de_comptage", "Coordonnées_géographiques", "mois_annee_comptage","Date_et_heure_de_comptage", "Date", "Saison", "pluie_intensity"], axis=1)
  df1_LR.replace({False:0, True:1}, inplace=True) #On remplace False par 0 et True par 1
  df1_LR["Identifiant_du_compteur"] = df1_LR["Identifiant_du_compteur"].str.replace("-", "")
  df1_2024 = df1_LR.loc[(df1_LR["Année"] == 2024)]# servira pour comparer les valeurs prédites par le modèle sur le mois de janvier 2024
  df1_2024.reset_index(drop=True, inplace=True)
  df1_LR = df1_LR.loc[(df1_LR["Année"] == 2023)]
  
  #Train test split
  feats=df1_LR.drop("Comptage_horaire", axis=1)
  target=df1_LR["Comptage_horaire"]
  X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size = 0.2, random_state=42)
  
  #Standardisation des données 
  scaler=StandardScaler()
  
  X_train= scaler.fit_transform(X_train)
  X_test= scaler.transform(X_test)

  #Entrainement
  lr=LinearRegression()
  lr.fit(X_train, y_train)
  y_pred=lr.predict(X_test)


  X_train2=feats
  y_train2=target
  X_test2=df1_2024.drop("Comptage_horaire", axis=1)

  scaler=StandardScaler()

  X_train2= scaler.fit_transform(X_train2)
  X_test2= scaler.transform(X_test2)

  lr=LinearRegression()
  lr.fit(X_train2, y_train2)
  y_pred2=lr.predict(X_test2).round(0)

  # Concaténer y_pred2 avec df_2024. Transformons y_pred2 en DataFrame d'abord
  y_pred2 = pd.DataFrame(y_pred2, columns=["Comptage_horaire_prédit"])
  df1_2024_predLR = pd.concat([df1_2024, y_pred2], axis=1)


  df_hourly_count = df1_2024_predLR.groupby('Heure')['Comptage_horaire'].mean()
  df_hourly_count_predicted=df1_2024_predLR.groupby('Heure')['Comptage_horaire_prédit'].mean()

  # Tracer la répartition du comptage horaire par heure
  fig = px.line()
  fig.add_scatter(x=df_hourly_count.index, y=df_hourly_count.values, mode='lines', name='Comptage horaire réel')
  fig.add_scatter(x=df_hourly_count_predicted.index, y=df_hourly_count_predicted.values, mode='lines', name='Comptage horaire prédit')
  fig.update_layout(
      xaxis_title='Heure',
      yaxis_title='Comptage horaire moyen',
      title='Comptage horaire moyen pour janvier 2024',
      title_x=0.25  # Centrer horizontalement
)
  st.plotly_chart(fig)


  st.write("#### Série Temporelle - Modèle SARIMA")
  st.write("<span style='font-weight: bold; color: blue;'>Rappels :</span>", unsafe_allow_html=True)
  
  st.write("Un processus ARMA d'ordres p et q est la combinaison \n"
           "- d'un processus autorégressif AR d'ordre p \n"
           "- et d'un processus de moyenne mobile MA d'ordre q \n")
  st.write("Un modèle SARIMA s'écrit sous la forme SARIMA(𝑝,𝑑,𝑞)(𝑃,𝐷,𝑄)𝑘 et combine \n"
           "- un processus ARMA(p,q) pour la partie non saisonnière \n"
           "- un processus ARMA(P,Q) pour la partie saisonnière\n"
           "- un paramètre I(d,D) pour les ordres de différenciation menant à une série stationnaire\n"
           "- un paramètre S(k) pour l'ordre de la saisonnalité")
  
           
           
  st.write("<span style='font-weight: bold; color: blue;'>Variables :</span> Date et Comptage_journalier.", unsafe_allow_html=True)
  st.write("<span style='font-weight: bold; color: blue;'>Focus :</span> Point de comptage le plus fréquenté : **Boulevard Sébastopol**.", unsafe_allow_html=True)
  st.write("<span style='font-weight: bold; color: blue;'>Entraînement </span> sur les relevés de l'année 2023 et prédictions pour janvier 2024.", unsafe_allow_html=True)
  st.write("Afin d'appliquer un modèle SARIMA, la série doit être stationnaire :\n"
           "- Une moyenne ou espérance constante  𝔼(𝑋𝑡)=𝜇\n"
           "- Une variance constante et finie  𝑉𝑎𝑟(𝑋𝑡)=𝜎2<∞\n"
           "- L'autocorrélation entre la variable  𝑋𝑡 et la variable  𝑋𝑡−𝑘 dépend uniquement\n"
           "du décalage k, et est égale quel que soit t.")
  
  
  #On retient un seul point de comptage pour la modélisation, le plus fréquenté : Boulevard Sébastopol
  df_sebastopol=df1.loc[df["Identifiant_du_compteur"]== "100057445-103057445"]

  #On ne garde que 2 variables : heure de comptage et comptage horaire
  df_sebastopol=df_sebastopol[["Date_et_heure_de_comptage","Comptage_horaire"]]
  df_sebastopol = df_sebastopol.sort_values(by="Date_et_heure_de_comptage")

  # Grouper les données par jour et agréger les valeurs
  df_sebastopol_daily = df_sebastopol.groupby(pd.Grouper(freq='D', key='Date_et_heure_de_comptage')).sum()
  df_sebastopol_daily = df_sebastopol_daily.rename(columns={ "Comptage_horaire": "Comptage_journalier"})
  df_sebastopol_daily = df_sebastopol_daily.rename_axis("Date_de_comptage") # Contient l'intégralité des données pour le point de comprage sebastopol agrégées par jour

  df_sebastopol_daily_2024=df_sebastopol_daily.loc[(df_sebastopol_daily.index >= "2024-01-01")] #servira pour comparer aux valauers prédites du mois de janvier 2024


  #Comme précédemment, on n'utilise que les données de l'année 2023
  df_sebastopol_daily_2023 = df_sebastopol_daily.loc[(df_sebastopol_daily.index >= "2023-01-01") & (df_sebastopol_daily.index < "2024-01-01")]

  # Convertir les valeurs en un tableau 1D
  values_array = df_sebastopol_daily_2023.values.flatten()

  # Tracer les données avec Plotly Express en utilisant l'index comme axe x
  fig = px.line(x=df_sebastopol_daily_2023.index, y=values_array)
  fig.update_layout(
      xaxis_title='Jour',
      yaxis_title='Comptage journalier',
      title='Valeurs de comptage journalier en 2023',
      title_x=0.3  # Centrer horizontalement
)
  st.plotly_chart(fig)

  st.write("**Décomposition de la série**")

  tsa = seasonal_decompose(df_sebastopol_daily_2023)
  fig=tsa.plot()
  plt.xticks(rotation=45)
  st.pyplot(fig)

  st.text("Résidu ≠ Bruit blanc faible.")
  st.write("**Modèle multiplicatif**")

  tsa = seasonal_decompose(df_sebastopol_daily_2023, model="multiplicative")
  fig=tsa.plot()
  plt.xticks(rotation=45)
  st.pyplot(fig)

  df_sebastopol_daily_2023=np.log(df_sebastopol_daily_2023)

  st.text("Résidu ~ Bruit blanc faible.")
  st.write("**Décomposition en transformation logarithmique**")

  tsa = seasonal_decompose(df_sebastopol_daily_2023)
  fig=tsa.plot()
  plt.xticks(rotation=45)
  st.pyplot(fig)

  st.write("**Vérification de la stationnarité**")
  st.text("Test Dickey-Fuller = 6,9%")

  # Convertir les valeurs en un tableau 1D
  values_array = df_sebastopol_daily_2023.values.flatten()

  st.write("Pour stationnariser la série, on procède par différenciation")
  fig = px.line(df_sebastopol_daily_2023.diff(1), x=df_sebastopol_daily_2023.diff(1).index, y=df_sebastopol_daily_2023.diff(1).columns)
  fig.update_layout(
     title='Série différenciée',
     title_x=0.45,  # Centrer horizontalement,
     xaxis_title="",
     yaxis_title="")
  for trace in fig.data:
    trace.update(showlegend=False)
  st.plotly_chart(fig)

  from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

  plot_acf(df_sebastopol_daily_2023.diff(1).dropna(), lags = 30, ax=ax1)
  plot_pacf(df_sebastopol_daily_2023.diff(1).dropna(), lags = 30, ax=ax2)
  st.pyplot(fig)

  st.write("Différenciation d'ordre 7 pour réduire les pics saisonniers")
  fig = px.line(df_sebastopol_daily_2023.diff(1).diff(7), x=df_sebastopol_daily_2023.diff(1).diff(7).index, y=df_sebastopol_daily_2023.diff(1).diff(7).columns)
  fig.update_layout(
     title='Série différenciée',
     title_x=0.45,  # Centrer horizontalement,
     xaxis_title="",
     yaxis_title="")
  for trace in fig.data:
    trace.update(showlegend=False)
  st.plotly_chart(fig)

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,7))

  plot_acf(df_sebastopol_daily_2023.diff(1).diff(7).dropna(), lags = 30, ax=ax1)
  plot_pacf(df_sebastopol_daily_2023.diff(1).diff(7).dropna(), lags = 30, ax=ax2)
  st.pyplot(fig)
  
  st.text("Test Dickey-Fuller = 0%")

  st.write("**Paramètres du modèle**")

  data = [
    ["",'AR(p)', 'MA(q)', 'ARMA(p,q)'],
    ["ACF", "Tend vers 0", "S'annule après l'ordre q", "Tend vers 0"],
    ["PACF", "S'annule après l'ordre q", "Tend vers 0", "Tend vers 0"]]
  
  
  ACF=pd.DataFrame(data)
  styled_ACF = ACF.style.set_table_styles([{'selector': 'th', 'props': [('display', 'none')]}])


  # Afficher le DataFrame dans un tableau sans les numéros de lignes et de colonnes
  st.table(styled_ACF)

  st.write("**SARIMA(0,1,1)(0,1,1)7**")
  st.write("<span style='font-weight: bold; color: blue;'>Performance :</span> **AIC** = -259, **BIC** = -248, **HQIC** = -255, **p-values** = 0.", unsafe_allow_html=True)


  @st.cache_resource
  def fit_sarima1(data):
    model1 = sm.tsa.SARIMAX(data, order=(0, 1, 1), seasonal_order=(0, 1, 1, 7))
    sarima1 = model1.fit()
    return sarima1

  sarima1 = fit_sarima1(df_sebastopol_daily_2023)

  prediction1 = np.exp(sarima1.forecast(29))

  df_plot = pd.DataFrame({
     'Date': df_sebastopol_daily_2024.index,
     'Comptage journalier réél': df_sebastopol_daily_2024.values.flatten(),
     'Comptage journalier prédit': prediction1
     })

  # Tracer le graphique avec Plotly Express
  fig = px.line(df_plot, x='Date', y=['Comptage journalier réél', 'Comptage journalier prédit'],
                labels={'value': 'Comptage journalier moyen', 'Date': 'Jour'},
                title='Comptage journalier réél VS prédit',
                color_discrete_map={'Comptage journalier réél': '#0000ff', 'Comptage journalier prédit': '#ffa500'})
  fig.update_layout(title_x=0.3)
  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)


  st.write("Le modèle traduit bien les variations journalières mais avec un décalage significatif pour les valeurs de comptage quotidien prédites.")
  st.write("##### Modèle apprenant sur les données jusqu'à janvier 2024")

  df_sebastopol_daily=np.log(df_sebastopol_daily)

  @st.cache_resource
  def fit_sarima2(data):
    model2 = sm.tsa.SARIMAX(data, order=(1, 0, 1), seasonal_order=(0, 1, 1, 7))
    sarima2 = model2.fit()
    return sarima2

  sarima2 = fit_sarima2(df_sebastopol_daily)

  st.write("**Paramètres du modèle**")
  st.write("**SARIMA(1,0,1)(0,1,1)7**")
  st.write("<span style='font-weight: bold; color: blue;'>Performance :</span> **AIC** = -293, **BIC** = -297, **HQIC** = -287, **p-values** = 0.", unsafe_allow_html=True)


  df_fevrier = pd.read_csv(filepath_or_buffer = 'C:\\Users\\DELL\\Desktop\\Data analyst\\Projet\\comptage-velo-donnees-compteurs-fevrier.csv', sep =';')

  df_fevrier = df_fevrier.rename(columns={'Date et heure de comptage': 'Date_et_heure_de_comptage'})
  df_fevrier = df_fevrier.rename(columns={'Identifiant du compteur': 'Identifiant_du_compteur'})
  df_fevrier = df_fevrier.rename(columns={'Comptage horaire': 'Comptage_horaire'})
  df_fevrier=df_fevrier[["Date_et_heure_de_comptage","Comptage_horaire",'Identifiant_du_compteur']]
  df_fevrier["Date_et_heure_de_comptage"]=pd.to_datetime(df_fevrier["Date_et_heure_de_comptage"],utc=True)

  df_fevrier_sebastopol=df_fevrier.loc[df_fevrier["Identifiant_du_compteur"]== "100057445-103057445"]
  df_fevrier_sebastopol.drop("Identifiant_du_compteur", axis=1, inplace=True)

  df_fevrier_sebastopol_daily = df_fevrier_sebastopol.groupby(pd.Grouper(freq='D', key='Date_et_heure_de_comptage')).sum()
  df_fevrier_sebastopol_daily = df_fevrier_sebastopol_daily.drop(df_fevrier_sebastopol_daily.index[0])

  prediction2 = np.exp(sarima2.forecast(22))

  df_plot = pd.DataFrame({
     'Date': prediction2.index,
     'Comptage journalier réél': df_fevrier_sebastopol_daily.values.flatten(),
     'Comptage journalier prédit': prediction2
     })

  # Tracer le graphique avec Plotly Express
  fig = px.line(df_plot, x='Date', y=['Comptage journalier réél', 'Comptage journalier prédit'],
                labels={'value': 'Comptage journalier moyen', 'Date': 'Jour'},
                title='Comptage journalier réél VS prédit',
                color_discrete_map={'Comptage journalier réél': '#0000ff', 'Comptage journalier prédit': '#ffa500'}   )
  fig.update_layout(title_x=0.3)

  # Afficher le graphique dans Streamlit
  st.plotly_chart(fig)
  st.write("<span style='font-weight: bold; color: blue;'>Difficultés</span> liées à la détermination des paramètres du modèle.", unsafe_allow_html=True)

if page == pages[4] : 
  st.markdown("---")
  st.write("### Conclusions & Perspectives")
  st.write("Cette étude a permis de mettre en évidence plusieurs tendances, l'intensité du trafic dépend entre autres des facteurs suivants :\n"
           "- L'<span style='font-weight: bold; color: blue;'>heure</span>\n"
           "- Le <span style='font-weight: bold; color: blue;'>jour </span>de la semaine\n"
           "- Les <span style='font-weight: bold; color: blue;'>congés </span> (vacances et jours fériés)\n"
           "- La <span style='font-weight: bold; color: blue;'>zone géographique </span>\n"
           "- Les <span style='font-weight: bold; color: blue;'>conditions météorologiques </span>\n",
           unsafe_allow_html=True
           )
  st.markdown("")
  st.write("Pour aller plus loin :\n"
           "- Intégrations de nouvelles variables telles que l' <span style='font-weight: bold; color: blue;'>accidentologie </span>, les <span style='font-weight: bold; color: blue;'>données soci-économiques </span> par quartiers ...\n"
           "- Analyser les données sur une échelle de temps plus grande\n"
           "- Amélioration du modèle => <span style='font-weight: bold; color: blue;'>SARIMAX </span>\n"
           "- Outil d'aide à la planification pour la conception d'infrastructures pour le développement de la pratique cycliste.\n",
           unsafe_allow_html=True
           )