import random
import time
import subprocess
import pandas as pd
import requests

# Archivos PCAP organizados por tipo de tráfico
pcap_files = {
    "UDP Flood": "test_files/CIC-DDoS-2019-UDPLag_test.pcap",
    "DNS": "test_files/CIC-DDoS-2019-DNS_test.pcap",
    "Syn Flood": "test_files/CIC-DDoS-2019-SynFlood_test.pcap",
    "Benign": "test_files/CIC-DDoS-2019-Benign_test.pcap",
    "Webb": "test_files/CIC-DDoS-2019-WebDDoS_test.pcap"

}

# Probabilidades de cada tipo de tráfico
probabilities = {
    "UDP Flood": 0.15,   # UDP DDoS
    "DNS": 0.15,   # DNS DDoS
    "Syn Flood": 0.15,   # Syn DDoS
    "Benign": 0.40,   # Benigno
    "Webb": 0.15   # Webb DDoS
}


server_url = "http://127.0.0.1:5000/send_pcap"

def send_pcap_to_server(pcap_path):
    """Envía un archivo .pcap directamente al servidor Flask."""
    with open(pcap_path, "rb") as pcap_file:
        files = {"file": (pcap_path, pcap_file, "application/octet-stream")}
        response = requests.post(server_url, files=files)

    if response.status_code == 200:
        print("Tráfico enviado correctamente")
        print("Predicciones:", response.json())
    else:
        print(f"Error al enviar tráfico: {response.status_code}")
        print(response.text)

def send_random_traffic():
    """Selecciona y envía tráfico aleatorio desde archivos PCAP."""
    try:
        while True:
            attack_type = random.choices(
                list(pcap_files.keys()), weights=probabilities.values(), k=1
            )[0]
            pcap_path = pcap_files[attack_type]

            print(f"Enviando tráfico: {attack_type}")
            send_pcap_to_server(pcap_path)

            time.sleep(random.uniform(5, 10))  # Espera aleatoria entre envíos

    except KeyboardInterrupt:
        print("\nSimulación de tráfico detenida por el usuario.")

if __name__ == "__main__":
    send_random_traffic()