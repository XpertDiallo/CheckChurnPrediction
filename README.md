# Churn Prediction Studio

Application Streamlit moderne pour explorer les donnees telecom, entrainer un modele de prediction de churn et scorer des clients.

## Fonctionnalites

- Dataset demo integre pour lancer l'application sans fichier externe
- Upload CSV avec selection de la colonne cible
- Entrainement Logistic Regression ou Random Forest
- Dashboard churn, distributions et segments a risque
- Matrice de confusion, metriques et variables influentes
- Prediction client individuelle
- Scoring batch avec export CSV

## Lancement local

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

La colonne cible doit contenir exactement deux classes. Pour un dataset Telco classique, utilisez `Churn`.
