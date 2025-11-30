# Contenu du fichier app.py

import streamlit as st
import pandas as pd
import os
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import re
import numpy as np

# --- 0. NETTOYAGE DES DONNÉES ET FONCTIONS UTILES ---

# Nettoyage des noms de colonnes pour l'intégration
def clean_col_name(name):
    name = str(name).upper().strip()
    name = re.sub(r'[^A-Z0-9_]', '', name)
    return name

# Charger les stopwords français pour le RAG
FRENCH_STOPWORDS = stopwords.words('french')

# --- 1. CONFIGURATION ET INITIALISATION DE GEMINI ---

# La clé est lue à partir des secrets de l'environnement de déploiement (Streamlit)
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] 
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-2.5-flash' 

# --- 2. LE PROMPT SYSTÈME (STATIQUE) ---
SYSTEM_PROMPT = """
Rôle : Tu es un Curateur de Voyage expert en "Slow Living" et Gastronomie en France. Ton but est d'aider des utilisateurs exigeants à trouver des hébergements "Rustique Chic" pour déconnecter.
[Règles d'or du Prompt pour l'IA, comme défini ensemble]
"""

# --- 3. CHARGEMENT DE LA BASE DE DONNÉES (Mise en cache) ---

@st.cache_data
def load_data():
    """Charge et nettoie les données du Google Sheet."""
    GOOGLE_SHEET_URL = st.secrets["GOOGLE_SHEET_URL"] 

    df = pd.read_csv(GOOGLE_SHEET_URL, sep=',') 
    
    # Nettoyage des colonnes (même logique que dans Colab)
    df.columns = [clean_col_name(col) for col in df.columns]

    rename_dict = {}
    for col in df.columns:
        if 'NOM' in col and 'LIEU' in col: rename_dict[col] = 'NOM_LIEU'
        elif 'DESC' in col: rename_dict[col] = 'DESCRIPTION_RAG'
        elif 'PRIX' in col: rename_dict[col] = 'PRIX_MIN_NUIT'
        elif 'URL' in col: rename_dict[col] = 'URL_RESA'
    df.rename(columns=rename_dict, inplace=True)
    
    # Nettoyage du prix
    df['PRIX_MIN_NUIT'] = df['PRIX_MIN_NUIT'].astype(str).str.replace(r'[^\d.]', '', regex=True)
    df['PRIX_MIN_NUIT'] = pd.to_numeric(df['PRIX_MIN_NUIT'], errors='coerce').fillna(0).astype(int)
    
    return df

# Charger le DataFrame au lancement de l'application
df = load_data()


# --- 4. FONCTION RAG (Recherche de Similarité) ---

def trouver_lieux_pertinents(requete_utilisateur, dataframe, top_k=3):
    # Logique de TF-IDF et similarité cosinus (identique à Colab)
    # ... (le code de la fonction doit être inclus ici, identique à votre version finale) ...
    # Le code est trop long pour être inclus ici, mais il est identique à celui que vous avez dans Colab.
    if dataframe.empty: return pd.DataFrame()
    documents = dataframe['DESCRIPTION_RAG'].fillna('').tolist()
    documents_et_requete = [requete_utilisateur] + documents
    vectorizer = TfidfVectorizer(stop_words=FRENCH_STOPWORDS)
    tfidf_matrix = vectorizer.fit_transform(documents_et_requete)
    cosine_sim = cosine_similarity(tfidf_matrix[0].reshape(1,-1), tfidf_matrix[1:])
    sim_scores = sorted(list(enumerate(cosine_sim[0])), key=lambda x: x[1], reverse=True)
    return dataframe.iloc[[i[0] for i in sim_scores[:top_k]]]


# --- 5. FONCTION DE GÉNÉRATION GEMINI ---

def generer_recommandation_gemini(requete_utilisateur, lieux_contextuels):
    # Logique de construction du prompt et appel à l'API Gemini (identique à Colab)
    # ... (le code de la fonction doit être inclus ici, identique à celui de Colab) ...
    if lieux_contextuels.empty: return "Aucun lieu trouvé."
    
    context_text = ""
    for index, row in lieux_contextuels.iterrows():
        context_text += f"- Nom: {row['NOM_LIEU']}\n  Description: {row['DESCRIPTION_RAG']}\n  Prix: {row['PRIX_MIN_NUIT']}€\n  URL: {row['URL_RESA']}\n\n"
    
    user_prompt = f"REQUÊTE: {requete_utilisateur}\n\nCONTEXTE:\n{context_text}\n\nRecommande le meilleur lieu."
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_prompt,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
    )
    return response.text

# --- 6. INTERFACE STREAMLIT (LE FRONT-END) ---

st.set_page_config(page_title="Slow Travel Curated", layout="wide")

st.title("🌿 L'Assistant Slow Travel & Gastronomie")
st.markdown("Bienvenue dans votre guide personnalisé pour les escapades **Rustique Chic** en France. Décrivez votre week-end de rêve et laissez l'IA vous trouver la pépite dans notre sélection exclusive.")

# Zone de saisie
user_query = st.text_input("Décrivez votre week-end (ex: Romantique, près de l'océan, avec table d'hôte)", key="query")

if user_query:
    with st.spinner('⏳ Recherche des pépites dans la sélection...'):
        # 4. Exécution du RAG
        lieux_trouves = trouver_lieux_pertinents(user_query, df)
        
        # 5. Génération de la recommandation par Gemini
        resultat = generer_recommandation_gemini(user_query, lieux_trouves)
        
        # 6. Affichage du résultat
        st.subheader("✨ Notre Recommandation Curatée :")
        st.markdown(resultat)
        
        # Optionnel: Afficher la sélection brute pour transparence
        st.markdown("---")
        st.markdown("### 🔎 Détail des lieux consultés par l'IA (Top 3):")
        st.dataframe(lieux_trouves[['NOM_LIEU', 'PRIX_MIN_NUIT', 'NOTE_AMBIANCE', 'URL_RESA']])
