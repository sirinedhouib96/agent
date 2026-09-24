import os
import io
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

def check_data_quality(df: pd.DataFrame) -> dict:
    """Effectue un contrôle qualité automatisé sur le DataFrame."""
    metrics = {
        "shape": {"rows": df.shape[0], "columns": df.shape[1]},
        "missing_values": df.isnull().sum().to_dict(),
        "total_missing": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "column_types": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "numeric_summary": {},
    }

    # Résumé statistique pour les colonnes numériques
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        metrics["numeric_summary"] = numeric_df.describe().to_dict()

    return metrics


def analyze_dataframe(df: pd.DataFrame) -> str:
    """Analyse le DataFrame et génère un rapport via le LLM."""
    # 1. Extraction des métriques de qualité
    metrics = check_data_quality(df)

    # Aperçu des 5 premières lignes
    head_sample = df.head(5).to_markdown()

    # 2. Configuration du LLM (OpenAI GPT-4o)
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Tu es un expert en analyse de données et contrôle qualité. "
            "Ton rôle est d'analyser les métriques transmises, d'identifier les erreurs/anomalies "
            "et de rédiger un rapport clair, structuré et professionnel en français.",
        ),
        (
            "user",
            """Voici les informations extraites d'un fichier de données :

### 1. Aperçu des données (5 premières lignes) :
{head_sample}

### 2. Métriques globales & Détection d'erreurs :
- **Nombre de lignes** : {rows}
- **Nombre de colonnes** : {cols}
- **Doublons stricts** : {duplicates}
- **Valeurs manquantes par colonne** : {missing_values}
- **Types de données** : {column_types}

### Consignes pour ton rapport :
1. **Résumé exécutif** : Présente succinctement ce que contient le fichier (volume, structure).
2. **Diagnostic qualité (Erreurs & Anomalies)** : Liste clairement les problèmes détectés (doublons, colonnes fortement incomplètes, types incohérents).
3. **Analyse statistique rapide** : Commente la distribution ou la cohérence des chiffres.
4. **Recommandations** : Indique les actions de nettoyage à réaliser prioritairement.

Rédige un rapport directement exploitable en Markdown.""",
        ),
    ])

    chain = prompt | llm

    # 3. Exécution de la chaîne
    response = chain.invoke({
        "head_sample": head_sample,
        "rows": metrics["shape"]["rows"],
        "cols": metrics["shape"]["columns"],
        "duplicates": metrics["duplicate_rows"],
        "missing_values": metrics["missing_values"],
        "column_types": metrics["column_types"],
    })

    return response.content