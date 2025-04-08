import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, matthews_corrcoef
import seaborn as sns
import matplotlib.pyplot as plt

import os

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
# st.subheader("📈 Distribution des métriques")
# fig = px.histogram(df_metrics_filtered, x="key", y="value", color="key", barmode="group")
# st.plotly_chart(fig)

# 📊 Graphique des métriques
st.subheader("📈 Courbes des métriques")

# Filtrer les métriques pertinentes
metrics_to_plot = ["train_loss", "val_auc", "val_accuracy"]
df_metrics_filtered_plot = df_metrics_filtered[df_metrics_filtered["key"].isin(metrics_to_plot)]

# Tracer les courbes
fig = px.line(
    df_metrics_filtered_plot,
    x="step",  # Utiliser le timestamp pour l'axe X
    y="value",      # Valeur des métriques pour l'axe Y
    color="key",    # Différencier les courbes par la clé (métrique)
    title="Évolution des métriques au cours du temps",
    labels={"timestamp": "Temps", "value": "Valeur", "key": "Métrique"}
)

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

# Map des IDs des modèles
model_ids = {
    "Vermax": "13a7703d7db64531aa1130cdb05855a8",
    "Meleys": "53ca430a698741c09ba4aa538de81628"
}

if uploaded_file is not None:
    from PIL import Image
    import numpy as np
    import torch
    import mlflow.pytorch
    from monai.transforms import (
        Compose,
        LoadImage,
        EnsureChannelFirst,
        ScaleIntensity,
        Resize,
        EnsureType,
        Lambda,
    )

    # Charger l'image
    image = Image.open(uploaded_file).convert("RGB")  # Convertir en RGB pour s'assurer qu'il y a 3 canaux
    st.image(image, caption="Image chargée pour l'inférence", use_column_width=True)

    # Sauvegarder temporairement l'image pour utiliser LoadImage
    temp_image_path = "temp_image.png"
    image.save(temp_image_path)

    # Charger le modèle
    model_id = model_ids[selected_inference_model]
    model_path = f"./mlruns/1/{model_id}/artifacts/model"
    try:
        model = mlflow.pytorch.load_model(model_path)
        model.eval()

        # Prétraiter l'image avec les mêmes transformations que dans le test
        preprocess = Compose([
            LoadImage(image_only=True),  # Charger l'image depuis le chemin
            EnsureChannelFirst(),       # Convertir (H, W, C) en (C, H, W)
            ScaleIntensity(),           # Normaliser les intensités des pixels
            Resize((256, 256)),         # Redimensionner l'image à 256x256
            EnsureType(),               # S'assurer que l'image est un tenseur PyTorch
            Lambda(lambda x: x[:3, :, :] if x.shape[0] == 4 else x),  # S'assurer que l'image a 3 canaux
        ])
        
        # Appliquer les transformations
        input_tensor = preprocess(temp_image_path)
        input_tensor = torch.unsqueeze(input_tensor, 0)  # Ajouter une dimension batch

        # Faire l'inférence
        device = torch.device("cpu")
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output = model(input_tensor)
            prediction = torch.argmax(output, dim=1).item()

        st.write(f"Prédiction : {'Healthy' if prediction == 0 else 'Parkinson'}")
    except OSError as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
    except RuntimeError as e:
        st.error(f"Erreur lors du prétraitement ou de l'inférence : {e}")
    finally:
        # Supprimer le fichier temporaire
        if os.path.exists(temp_image_path):
            os.remove(temp_image_path)

# 📌 Composante d'analyse
st.header("📊 Analyse des résultats")
if st.button("Afficher l'analyse des résultats"):
    # Ajouter dynamiquement la colonne 'true_label' si elle n'existe pas
    if "true_label" not in df_test_predictions.columns:
        # Calculer dynamiquement le nombre d'exemples pour chaque classe
        num_samples = len(df_test_predictions)
        num_healthy = num_samples // 2
        num_parkinson = num_samples - num_healthy
        df_test_predictions["true_label"] = ["healthy"] * num_healthy + ["parkinson"] * num_parkinson

    # Extraire les vraies valeurs et les prédictions
    y_true = df_test_predictions["true_label"]
    y_pred = df_test_predictions["prediction"]

    # Matrice de confusion
    cm = confusion_matrix(y_true, y_pred, labels=["healthy", "parkinson"])
    st.subheader("Matrice de confusion")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["healthy", "parkinson"], yticklabels=["healthy", "parkinson"])
    st.pyplot(fig)

    # Rapport de classification
    st.subheader("Rapport de classification")
    report = classification_report(y_true, y_pred, target_names=["healthy", "parkinson"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Afficher des métriques supplémentaires
    accuracy = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    st.write(f"**Exactitude (Accuracy)** : {round(accuracy * 100, 2)}%")
    st.write(f"**MCC (Matthews Correlation Coefficient)** : {round(mcc, 2)}")
