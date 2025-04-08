import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

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

# 🎨 Interface Streamlit
st.title("📊 Dashboard MLflow - Expérimentations")

# Filtrer les expériences 'Arrax', 'Vermax', et 'Meleys'
df_runs_filtered = df_runs[df_runs["name"].isin(["Arrax", "Vermax", "Meleys"])]

# 📌 Sélection du modèle
st.sidebar.header("🔍 Filtrer les modèles")
selected_model = st.sidebar.selectbox("Sélectionner un modèle", df_runs_filtered["name"].unique())

# Charger les prédictions de test en fonction du modèle sélectionné
test_predictions_path = f"{selected_model}_test_predictions.csv"
df_test_predictions = pd.read_csv(test_predictions_path)

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

# 📌 Composante d'inférence
st.sidebar.header("🧠 Inférence")
selected_inference_model = st.sidebar.selectbox("Choisir un modèle pour l'inférence", ["Vermax", "Meleys"])
uploaded_file = st.sidebar.file_uploader("Charger une image pour l'inférence", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    from PIL import Image
    import torch
    import mlflow.pytorch

    # Charger l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée pour l'inférence", use_column_width=True)

    # Charger le modèle
    model_path = f"./mlruns/1/{selected_inference_model}_model/artifacts/model"
    model = mlflow.pytorch.load_model(model_path)
    model.eval()

    # Prétraiter l'image
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    input_tensor = preprocess(image).unsqueeze(0)

    # Faire l'inférence
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    st.write(f"Prédiction : {'Healthy' if prediction == 0 else 'Parkinson'}")

# 📌 Composante d'analyse
st.header("📊 Analyse des résultats")
if st.button("Afficher l'analyse des résultats"):
    y_true = df_test_predictions["true_label"]
    y_pred = df_test_predictions["prediction"]

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred)
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Healthy", "Parkinson"], yticklabels=["Healthy", "Parkinson"])
    st.pyplot(fig)

    # Rapport de classification
    st.subheader("Rapport de classification")
    report = classification_report(y_true, y_pred, target_names=["Healthy", "Parkinson"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())
