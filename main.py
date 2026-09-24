from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import io
from agent import analyze_dataframe

app = FastAPI(
    title="Agent Analyseur de Données",
    description="API permettant d'analyser des fichiers CSV/Excel, de détecter les erreurs et de générer une synthèse.",
    version="1.0.0",
)


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Agent d'analyse opérationnel."}


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    # Vérification de l'extension du fichier
    filename = file.filename.lower()
    if not (filename.endswith(".csv") or filename.endswith(".xlsx") or filename.endswith(".xls")):
        raise HTTPException(
            status_code=400,
            detail="Format non supporté. Veuillez envoyer un fichier CSV ou Excel.",
        )

    try:
        contents = await file.read()

        # Chargement en mémoire selon le format
        if filename.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))

        if df.empty:
            raise HTTPException(status_code=400, detail="Le fichier transmis est vide.")

        # Appel à l'agent
        report = analyze_dataframe(df)

        return {
            "status": "success",
            "filename": file.filename,
            "total_rows": df.shape[0],
            "total_columns": df.shape[1],
            "report": report,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du traitement du fichier : {str(e)}")