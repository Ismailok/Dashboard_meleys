import streamlit as st
import pandas as pd
import sqlite3

try:
    import plotly.express as px
except ModuleNotFoundError:
    import os
    os.system("pip install plotly")
    import plotly.express as px


# 🔌 Connexion à la base MLflow
DB_FILE = "mlflow.db"

@st.cache_data
def load_data():
    conn = sqlite3.connect(DB_FILE)
    df_metrics = pd.read_sql_query("SELECT * FROM metrics", conn)
    df_params = pd.read_sql_query("SELECT * FROM params", conn)
    df_runs = pd.read_sql_query("SELECT * FROM runs", conn)
    conn.close()
    
    # Convert timestamps to datetime format
    df_metrics['timestamp'] = pd.to_datetime(df_metrics['timestamp'], unit='ms')
    df_runs['start_time'] = pd.to_datetime(df_runs['start_time'], unit='ms')
    df_runs['end_time'] = pd.to_datetime(df_runs['end_time'], unit='ms')
    
    return df_metrics, df_params, df_runs

df_metrics, df_params, df_runs = load_data()

# Charger les prédictions de test
test_predictions_path = "Vermax_test_predictions.csv"
df_test_predictions = pd.read_csv(test_predictions_path)

# 🎨 Interface Streamlit
st.title("📊 Dashboard MLflow - Expérimentations")

# Filtrer les expériences 'Arrax' et 'Vermax'
df_runs_filtered = df_runs[df_runs["name"].isin(["Arrax", "Vermax"])]

# 📌 Sélection du modèle
st.sidebar.header("🔍 Filtrer les modèles")
selected_model = st.sidebar.selectbox("Sélectionner un modèle", df_runs_filtered["name"].unique())

# 🎯 Filtrer les données selon le modèle sélectionné
df_metrics_filtered = df_metrics[df_metrics["run_uuid"].isin(df_runs_filtered[df_runs_filtered["name"] == selected_model]["run_uuid"])]
df_params_filtered = df_params[df_params["run_uuid"].isin(df_runs_filtered[df_runs_filtered["name"] == selected_model]["run_uuid"])]
df_runs_filtered = df_runs_filtered[df_runs_filtered["name"] == selected_model]

# 📊 Graphique des métriques
st.subheader("📈 Distribution des métriques")
fig = px.histogram(df_metrics_filtered, x="key", y="value", color="key", barmode="group")
st.plotly_chart(fig)

# 📌 Afficher les métriques
st.subheader("📋 Metrics enregistrées")
st.dataframe(df_metrics_filtered)

# 📌 Afficher les paramètres
st.subheader("⚙️ Paramètres des modèles")
st.dataframe(df_params_filtered)

# 📌 Afficher les informations des runs
st.subheader("🕒 Informations des runs")
st.dataframe(df_runs_filtered[['run_uuid', 'start_time', 'end_time']])

# 📌 Meilleurs résultats
best_runs = df_metrics_filtered.sort_values(by="value", ascending=True).head(10)
st.subheader("🏆 Top 10 Meilleurs Modèles (selon la métrique)")
st.dataframe(best_runs)

# 📌 Afficher les prédictions de test
st.subheader("🔍 Prédictions de test")
st.dataframe(df_test_predictions)

# 📌 Téléchargement des résultats
st.download_button("📥 Télécharger les métriques (CSV)", df_metrics_filtered.to_csv(index=False), "mlflow_metrics.csv")
st.download_button("📥 Télécharger les paramètres (CSV)", df_params_filtered.to_csv(index=False), "mlflow_params.csv")
st.download_button("📥 Télécharger les prédictions de test (CSV)", df_test_predictions.to_csv(index=False), "test_predictions.csv")
