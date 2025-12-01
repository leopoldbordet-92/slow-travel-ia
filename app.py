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

# --- CORRECTION DE L'ERREUR NLTK (IMPORTANT) ---
# Télécharge les dictionnaires nécessaires au démarrage du serveur
nltk.download('stopwords')
# -----------------------------------------------

# --- 0. NETTOYAGE ET UTILITAIRES ---

def clean_col_name(name):
    """Nettoie les noms de colonnes (majuscules, sans accents/symboles)"""
    name = str(name).upper().strip()
    name = re.sub(r'[^A-Z0-9_]', '', name)
    return name

# Charger les mots vides français une fois pour toutes
FRENCH_STOPWORDS = stopwords.words('french')


# --- 1. CONFIGURATION GEMINI ---

# Récupération de la clé API depuis les secrets Streamlit
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    st.error("❌ Clé API manquante. Ajoutez GEMINI_API_KEY dans les Secrets de Streamlit.")
    st.stop()

client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = 'gemini-2.5-flash'


# --- 2. LE PROMPT SYSTÈME ---
SYSTEM_PROMPT = """
Rôle : Tu es un Curateur de Voyage expert en "Slow Living" et Gastronomie en France. Ton but est d'aider des utilisateurs exigeants à trouver des hébergements "Rustique Chic" pour déconnecter.

Ta Source de Vérité : Tu ne dois utiliser EXCLUSIVEMENT que les informations fournies dans le CONTEXTE (la base de données).
* Si l'utilisateur demande un lieu qui n'est pas dans le contexte, tu dois répondre poliment que tu n'as pas encore cette adresse dans ta sélection curatée, et proposer le lieu le plus proche disponible dans ta base.
* INTERDICTION FORMELLE D'INVENTER des lieux ou des caractéristiques.

Ton Style et Ton (Option B) :
* Tu n'es pas un robot, tu es un esthète. Utilise un langage sensoriel, émotionnel et élégant.
* Parle d'ambiance, de lumière, de silence, d'odeurs et de goût.
* Tu dois faire rêver, mais rester précis sur la logistique.

Structure de ta Réponse :
Pour chaque recommandation, tu dois suivre ce format strict :
---
**Recommandation Curatée : [NOM_LIEU]**
**L'Accroche Émotionnelle :** Une phrase qui capture l'essence du lieu.
**Pourquoi ce lieu pour vous :** Explique pourquoi cela correspond à la demande (en utilisant la description fournie).
**L'Expérience Food (Crucial) :** Détaille précisément l'aspect gastronomique.
**Le Petit + Luxe :** Mentionne l'équipement ou le détail qui fait la différence.
**Budget Minimum :** À partir de [PRIX_MIN_NUIT] €/nuit.
**Lien de Réservation :** [URL_RESA]
---
"""


# --- 3. CHARGEMENT DES DONNÉES (CACHE) ---

@st.cache_data
def load_data():
    """Charge les données depuis Google Sheets et nettoie les colonnes."""
    try:
        url = st.secrets["GOOGLE_SHEET_URL"]
        # Lecture avec virgule (standard Google Sheets web)
        df = pd.read_csv(url, sep=',')
        
        # 1. Nettoyage des noms de colonnes
        df.columns = [clean_col_name(col) for col in df.columns]

        # 2. Mapping intelligent pour retrouver nos colonnes
        rename_dict = {}
        for col in df.columns:
            if 'NOM' in col and 'LIEU' in col: rename_dict[col] = 'NOM_LIEU'
            elif 'DESC' in col: rename_dict[col] = 'DESCRIPTION_RAG'
            elif 'PRIX' in col: rename_dict[col] = 'PRIX_MIN_NUIT'
            elif 'URL' in col: rename_dict[col] = 'URL_RESA'
        
        df.rename(columns=rename_dict, inplace=True)

        # 3. Vérification des colonnes essentielles
        required = ['NOM_LIEU', 'DESCRIPTION_RAG', 'PRIX_MIN_NUIT', 'URL_RESA']
        for col in required:
            if col not in df.columns:
                st.error(f"Colonne manquante : {col}. Colonnes trouvées : {df.columns.tolist()}")
                return pd.DataFrame()

        # 4. Nettoyage du prix (enlever '€' et convertir en nombre)
        df['PRIX_MIN_NUIT'] = df['PRIX_MIN_NUIT'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['PRIX_MIN_NUIT'] = pd.to_numeric(df['PRIX_MIN_NUIT'], errors='coerce').fillna(0).astype(int)

        return df

    except Exception as e:
        st.error(f"Erreur lors du chargement des données : {e}")
        return pd.DataFrame()

# Chargement au démarrage
df = load_data()


# --- 4. MOTEUR RAG (RECHERCHE) ---

def trouver_lieux_pertinents(requete, dataframe, top_k=3):
    if dataframe.empty: return pd.DataFrame()

    # Remplir les vides par du texte vide pour éviter les bugs
    documents = dataframe['DESCRIPTION_RAG'].fillna('').tolist()
    
    # Ajout de la requête utilisateur au début pour la comparaison
    documents_et_requete = [requete] + documents
    
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer(stop_words=FRENCH_STOPWORDS)
    tfidf_matrix = vectorizer.fit_transform(documents_et_requete)
    
    # Calcul de similarité (Requête vs Tous les lieux)
    cosine_sim = cosine_similarity(tfidf_matrix[0].reshape(1,-1), tfidf_matrix[1:])
    
    # Tri des résultats
    sim_scores = sorted(list(enumerate(cosine_sim[0])), key=lambda x: x[1], reverse=True)
    
    # Récupération des meilleurs index
    top_indices = [i[0] for i in sim_scores[:top_k]]
    return dataframe.iloc[top_indices]


# --- 5. GÉNÉRATION GEMINI ---

def generer_recommandation(requete, lieux_contextuels):
    if lieux_contextuels.empty:
        return "Désolé, je n'ai trouvé aucun lieu correspondant dans la sélection."

    # Construction du contexte texte pour l'IA
    context_text = ""
    for _, row in lieux_contextuels.iterrows():
        context_text += f"- Nom: {row['NOM_LIEU']}\n"
        context_text += f"  Description: {row['DESCRIPTION_RAG']}\n"
        context_text += f"  Prix: {row['PRIX_MIN_NUIT']}€\n"
        context_text += f"  URL: {row['URL_RESA']}\n\n"
    
    user_prompt = f"REQUÊTE UTILISATEUR: {requete}\n\nCONTEXTE DISPONIBLE:\n{context_text}\n\nFais ta recommandation."

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_prompt,
            config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT)
        )
        return response.text
    except Exception as e:
        return f"Erreur de l'IA : {e}"


# --- 6. INTERFACE STREAMLIT ---

st.set_page_config(page_title="Slow Travel IA", page_icon="🌿", layout="centered")

st.title("🌿 L'Assistant Slow Travel")
st.markdown("""
Bienvenue. Je suis votre curateur personnel pour des escapades **Rustique Chic** en France.
Dites-moi ce que vous cherchez (ambiance, région, envie culinaire...), et je fouille ma sélection exclusive pour vous.
""")

# Formulaire de recherche
with st.form("search_form"):
    user_query = st.text_input("Votre recherche :", placeholder="Ex: Week-end romantique avec table d'hôte bio...")
    submitted = st.form_submit_button("Trouver ma pépite ✨")

if submitted and user_query:
    if df.empty:
        st.error("La base de données n'est pas chargée correctement.")
    else:
        with st.spinner("Analyse de la sélection en cours..."):
            # 1. Recherche RAG
            lieux_trouves = trouver_lieux_pertinents(user_query, df)
            
            # 2. Génération IA
            reponse_ia = generer_recommandation(user_query, lieux_trouves)
            
            # 3. Affichage
            st.markdown("---")
            st.markdown(reponse_ia)
            
            # (Optionnel) Debug : Voir quels lieux ont été envoyés à l'IA
            with st.expander("Voir les lieux analysés (Debug)"):
                st.dataframe(lieux_trouves[['NOM_LIEU', 'PRIX_MIN_NUIT', 'URL_RESA']])
