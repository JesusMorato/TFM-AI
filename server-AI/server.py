from flask import Flask, request, jsonify
import os
import pandas as pd
import pickle
import numpy as np
from joblib import load
import datetime
from pyflowmeter.sniffer import create_sniffer

# Rutas de archivos
UPLOAD_FOLDER = "uploads"
CSV_FOLDER = "converted_csv"
MODEL_PATH = "xgboost_model.pkl"
SCALER_PATH = "preprocessing_artifacts/scaler.bin"
IP_ENCODER_PATH = "preprocessing_artifacts/ip_encoder.bin"
LABEL_ENCODER_PATH = "label_encoder.pkl"
PREDICTIONS_LOG = "predictions_log.csv"

# Se crean las carpetas si no existen
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CSV_FOLDER, exist_ok=True)

# Se carga el modelo y los artefactos de preprocesamiento
with open(MODEL_PATH, "rb") as model_file:
    model = pickle.load(model_file)

scaler = load(SCALER_PATH)
ip_encoder = load(IP_ENCODER_PATH)
label_encoder = load(LABEL_ENCODER_PATH)

# Columnas necesarias en orden para el modelo (obtenidad del feature engineering)
FEATURE_COLUMNS = [
    "protocol",
    "totlen_fwd_pkts", "fwd_pkt_len_min", "bwd_pkt_len_min",
    "flow_byts_s", "flow_pkts_s", "flow_iat_min", "bwd_iat_tot",
    "bwd_iat_mean", "fwd_psh_flags", "bwd_header_len", "fwd_pkts_s",
    "bwd_pkts_s", "pkt_len_min", "pkt_len_mean", "fin_flag_cnt",
    "syn_flag_cnt", "rst_flag_cnt", "psh_flag_cnt", "ack_flag_cnt",
    "urg_flag_cnt", "ece_flag_cnt", "down_up_ratio", "fwd_seg_size_avg",
    "bwd_seg_size_avg", "init_fwd_win_byts", "init_bwd_win_byts",
    "fwd_act_data_pkts", "fwd_seg_size_min", "active_std", "active_min",
    "idle_std", "idle_min"
]

# Creamos aplicacion Flask
app = Flask(__name__)

def convert_pcap_to_csv(pcap_path):
    # Se convierte el archivo pcap a csv usando pyflowmeter
    csv_output = os.path.join(CSV_FOLDER, os.path.basename(pcap_path) + ".csv")
    
    sniffer = create_sniffer(
        input_file=pcap_path,
        to_csv=True,
        output_file=csv_output
    )

    sniffer.start()
    try:
        sniffer.join()
        print(f"PCAP convertido a CSV: {csv_output}")
        return csv_output
    except KeyboardInterrupt:
        sniffer.stop()
        print("Error: conversion interrumpida")
        return None
    finally:
        sniffer.join()

def preprocess_csv_old(csv_path):
    # Preprocesa el csv para el modelo: ordena columnas y aplica escalado
    df = pd.read_csv(csv_path)

    missing_cols = set(FEATURE_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Columnas faltantes en el CSV: {missing_cols}")

    print(df.columns)

    df = df[FEATURE_COLUMNS]
    scaled_data = scaler.transform(df)

    return scaled_data

def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)

    # Paso 1: timestamp a datetime + ordenar
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df.sort_values(by='timestamp', inplace=True)
        df.set_index('timestamp', inplace=True)

    # Paso 2: seleccionar solo las features que entrenaron el modelo
    missing = set(FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    X = df[FEATURE_COLUMNS]
    # Paso 3: escalar
    print("Escalando datos...")
    X_scaled = scaler.transform(X)
    print("Datos escalados:", X_scaled.shape)

    return X_scaled

@app.route("/send_pcap", methods=["POST"])
def receive_and_save_predictions():
    if "file" not in request.files:
        return jsonify({"error": "Archivo no enviado"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Nombre de archivo vacío"}), 400

    filename = os.path.basename(file.filename)
    pcap_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(pcap_path)

    csv_path = convert_pcap_to_csv(pcap_path)
    if not csv_path:
        return jsonify({"error": "Error al convertir pcap a csv"}), 500

    try:
        data = preprocess_csv(csv_path)
        predictions = model.predict(data)
        predicted_labels = label_encoder.inverse_transform(predictions)

        timestamp = datetime.datetime.now().isoformat()
        result_df = pd.DataFrame({
            "timestamp": [timestamp] * len(predicted_labels),
            "file": [filename] * len(predicted_labels),
            "prediction": predicted_labels
        })

        if os.path.exists(PREDICTIONS_LOG):
            result_df.to_csv(PREDICTIONS_LOG, mode='a', header=False, index=False)
        else:
            result_df.to_csv(PREDICTIONS_LOG, index=False)

        return jsonify({
            "predictions": predicted_labels.tolist(),
            "message": f"{len(predicted_labels)} predicciones guardadas en {PREDICTIONS_LOG}"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)



