import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import nltk
import re
import numpy as np

# --- 1. CONFIGURATION INITIALE ---

# Téléchargement des stopwords (indispensable pour Streamlit Cloud)
nltk.download('stopwords')
FRENCH_STOPWORDS = stopwords.words('french')

# Configuration de la page
st.set_page_config(page_title="Slow Travel IA", page_icon="🌿", layout="centered")

# Récupération des secrets
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    GOOGLE_SHEET_URL = st.secrets["GOOGLE_SHEET_URL"]
except Exception:
    st.error("❌ Erreur de configuration. Vérifiez que GEMINI_API_KEY et GOOGLE_SHEET_URL sont bien dans les Secrets Streamlit.")
    st.stop()

# Connexion à Gemini
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-2.5-flash'

# --- 2. LE CERVEAU (PROMPT SYSTÈME) ---
SYSTEM_PROMPT = """
Rôle : Tu es un Curateur de Voyage expert en "Slow Living" et Gastronomie en France. Ton but est d'aider des utilisateurs exigeants à trouver des hébergements "Rustique Chic" pour déconnecter.

Ta Source de Vérité : Tu ne dois utiliser EXCLUSIVEMENT que les informations fournies dans le CONTEXTE (la base de données).
* Si l'utilisateur demande un lieu qui n'est pas dans le contexte, dis poliment que tu ne l'as pas encore.
* INTERDICTION D'INVENTER.

Ton Style : Sensoriel, élégant, faisant appel aux 5 sens.
Données disponibles : Tu as accès à l'adresse exacte et au temps de trajet depuis Paris. Utilise-les si pertinent.

Structure de Réponse :
---
**Recommandation Curatée : [NOM_LIEU]**
**Localisation :** [REGION_FR] ([DISTANCE_DE_PARIS]h de Paris)
**L'Accroche Émotionnelle :** Une phrase qui capture l'essence.
**Pourquoi ce lieu :** Lien avec la demande utilisateur.
**L'Expérience Food :** Détails gastronomiques.
**Le Petit + :** Équipement ou détail unique.
**Budget :** À partir de [PRIX_MIN_NUIT] €/nuit.
**Lien :** [URL_RESA]
---
"""

# --- 3. CHARGEMENT ET NETTOYAGE DES DONNÉES ---

def clean_col_name(name):
    """Nettoie les noms de colonnes (majuscules, sans symboles)"""
    name = str(name).upper().strip()
    name = re.sub(r'[^A-Z0-9_]', '', name)
    return name

@st.cache_data
def load_data():
    try:
        # Lecture du CSV (Google Sheets exporte en virgule par défaut via le lien pub)
        df = pd.read_csv(GOOGLE_SHEET_URL, sep=',')
        
        # 1. Nettoyage des noms de colonnes
        df.columns = [clean_col_name(col) for col in df.columns]

        # 2. Mapping intelligent pour retrouver nos colonnes clés
        # On cherche des mots clés dans les colonnes nettoyées
        rename_dict = {}
        for col in df.columns:
            if 'NOM' in col and 'LIEU' in col: rename_dict[col] = 'NOM_LIEU'
            elif 'DESC' in col: rename_dict[col] = 'DESCRIPTION_RAG'
            elif 'PRIX' in col: rename_dict[col] = 'PRIX_MIN_NUIT'
            elif 'URL' in col: rename_dict[col] = 'URL_RESA'
            elif 'DIST' in col: rename_dict[col] = 'DISTANCE_DE_PARIS'
            elif 'ADRESSE' in col: rename_dict[col] = 'ADRESSE'
            elif 'REGION' in col: rename_dict[col] = 'REGION_FR'
        
        df.rename(columns=rename_dict, inplace=True)

        # 3. Vérification des colonnes vitales
        required = ['NOM_LIEU', 'DESCRIPTION_RAG', 'PRIX_MIN_NUIT', 'URL_RESA']
        for col in required:
            if col not in df.columns:
                st.error(f"⚠️ Colonne manquante : '{col}'. Vérifiez votre Google Sheet.")
                return pd.DataFrame()

        # 4. Nettoyage du Prix (conversion en nombre)
        df['PRIX_MIN_NUIT'] = df['PRIX_MIN_NUIT'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['PRIX_MIN_NUIT'] = pd.to_numeric(df['PRIX_MIN_NUIT'], errors='coerce').fillna(0).astype(int)

        return df

    except Exception as e:
        st.error(f"Erreur de lecture des données : {e}")
        return pd.DataFrame()

df = load_data()


# --- 4. MOTEUR DE RECHERCHE (AMÉLIORÉ : TOUTES COLONNES) ---

def trouver_lieux_pertinents(requete, dataframe, top_k=3):
    if dataframe.empty: return pd.DataFrame()

    # 1. Préparation : On remplit les vides
    df_search = dataframe.fillna('')

    # 2. CRÉATION DE LA SUPER-COLONNE
    # On fusionne TOUTES les colonnes de chaque ligne en un seul texte géant
    # Cela permet de trouver "Mercantour" (Région), "Piscine" (Équipement) ou "180" (Prix)
    dataframe['SEARCH_CONTENT'] = df_search.apply(
        lambda row: ' '.join(row.values.astype(str)), axis=1
    )

    # 3. Moteur TF-IDF
    documents = dataframe['SEARCH_CONTENT'].tolist()
    documents_et_requete = [requete] + documents
    
    vectorizer = TfidfVectorizer(stop_words=FRENCH_STOPWORDS)
    tfidf_matrix = vectorizer.fit_transform(documents_et_requete)
    
    # 4. Calcul de similarité
    cosine_sim = cosine_similarity(tfidf_matrix[0].reshape(1,-1), tfidf_matrix[1:])
    sim_scores = sorted(list(enumerate(cosine_sim[0])), key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in sim_scores[:top_k]]
    return dataframe.iloc[top_indices]


# --- 5. GÉNÉRATION IA (GEMINI) ---

def generer_recommandation(requete, lieux_contextuels):
    if lieux_contextuels.empty:
        return "Je n'ai rien trouvé qui corresponde parfaitement. Essayez d'élargir votre recherche."

    # Construction du contexte enrichi pour l'IA
    context_text = ""
    for _, row in lieux_contextuels.iterrows():
        # On gère le cas où certaines colonnes n'existent pas encore pour éviter les bugs
        dist = row.get('DISTANCE_DE_PARIS', 'Non spécifié')
        addr = row.get('ADRESSE', 'Non spécifié')
        region = row.get('REGION_FR', 'France')
        
        context_text += f"- LIEU: {row['NOM_LIEU']} ({region})\n"
        context_text += f"  Adresse: {addr} (Env. {dist}h de Paris)\n"
        context_text += f"  Description: {row['DESCRIPTION_RAG']}\n"
        context_text += f"  Prix: {row['PRIX_MIN_NUIT']}€\n"
        context_text += f"  Lien: {row['URL_RESA']}\n\n"
    
    user_prompt = f"REQUÊTE UTILISATEUR: {requete}\n\nCONTEXTE DES LIEUX TROUVÉS:\n{context_text}\n\nAgis comme le curateur et fais ta recommandation."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        return response.text
    except Exception as e:
        return f"Désolé, l'IA est capricieuse : {e}"


# --- 6. INTERFACE UTILISATEUR ---

st.title("🌿 L'Assistant Slow Travel")
st.markdown("""
**Bienvenue.** Je connais 19 pépites cachées en France. Dites-moi vos envies (Région, ambiance, temps de trajet...) et je trouve le lieu parfait.
""")

# Barre de recherche
with st.form("search_form"):
    user_query = st.text_input("Votre envie :", placeholder="Ex: Gîte dans le Mercantour, ou à moins de 2h de Paris...")
    submitted = st.form_submit_button("Dénicher la perle ✨")

if submitted and user_query:
    if df.empty:
        st.warning("La base de données semble vide ou inaccessible.")
    else:
        with st.spinner("Je fouille ma collection..."):
            # Recherche
            lieux_trouves = trouver_lieux_pertinents(user_query, df)
            
            # Génération
            reponse = generer_recommandation(user_query, lieux_trouves)
            
            # Affichage
            st.markdown("---")
            st.markdown(reponse)
            
            # Petit debug discret pour voir ce qu'il a trouvé (utile pour vérifier la distance)
            with st.expander("Voir les lieux analysés (Transparence)"):
                cols_to_show = ['NOM_LIEU', 'PRIX_MIN_NUIT', 'URL_RESA']
                # On ajoute les nouvelles colonnes si elles existent
                if 'DISTANCE_DE_PARIS' in lieux_trouves.columns: cols_to_show.append('DISTANCE_DE_PARIS')
                if 'REGION_FR' in lieux_trouves.columns: cols_to_show.append('REGION_FR')
                
                st.dataframe(lieux_trouves[cols_to_show])
