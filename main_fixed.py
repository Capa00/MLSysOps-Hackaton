import requests
import csv
from datetime import datetime, timedelta, timezone

# CONFIG
GRAFANA_URL = "http://91.98.64.132:3000/grafana"
USERNAME = "admin"
PASSWORD = "csl123"
DATASOURCE_UID = "mimir"
OUTPUT_FILE = "metrics.csv"

# Intervallo (ultime 6h come richiesto)
now = datetime.now(timezone.utc)
end_time = now.timestamp()
start_time = (now - timedelta(hours=6)).timestamp()

session = requests.Session()
session.auth = (USERNAME, PASSWORD)

# 1️⃣ Recupera lista metriche
print("🔍 Recupero lista metriche...")
resp = session.get(
    f"{GRAFANA_URL}/api/datasources/uid/{DATASOURCE_UID}/resources/api/v1/label/__name__/values"
)
if resp.status_code != 200:
    print(f"❌ Errore nel recupero metriche: {resp.status_code}")
    exit(1)

metrics = resp.json().get("data", [])
print(f"✅ Trovate {len(metrics)} metriche")

# 2️⃣ Scrivi CSV
with open(OUTPUT_FILE, mode="w", newline="") as csvfile:
    fieldnames = ["metric", "timestamp", "value", "labels"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    # 3️⃣ Query per ogni metrica usando API Prometheus diretta
    total_rows = 0
    successful_metrics = 0
    
    for i, metric in enumerate(metrics, 1):
        print(f"🔄 Processando metrica {i}/{len(metrics)}: {metric}")
        
        try:
            # Usa l'API query_range di Prometheus direttamente
            resp = session.get(
                f"{GRAFANA_URL}/api/datasources/uid/{DATASOURCE_UID}/resources/api/v1/query_range",
                params={
                    "query": metric,
                    "start": start_time,
                    "end": end_time,
                    "step": "60s"  # 1 minuto di intervallo
                }
            )
            
            if resp.status_code == 200:
                data = resp.json()
                
                if data.get("status") == "success":
                    result = data.get("data", {}).get("result", [])
                    print(f"  📊 Trovate {len(result)} serie temporali")
                    
                    metric_rows = 0
                    for series in result:
                        labels = series.get("metric", {})
                        values = series.get("values", [])
                        
                        # Crea stringa labels per identificazione
                        labels_str = ", ".join([f"{k}={v}" for k, v in labels.items() if k != "__name__"])
                        
                        print(f"    📈 Serie con {len(values)} punti dati: {labels_str[:100]}")
                        
                        for timestamp, value in values:
                            try:
                                ts_iso = datetime.fromtimestamp(float(timestamp), tz=timezone.utc).isoformat()
                                writer.writerow({
                                    "metric": metric,
                                    "timestamp": ts_iso,
                                    "value": value,
                                    "labels": labels_str
                                })
                                metric_rows += 1
                                total_rows += 1
                            except Exception as e:
                                print(f"    ❌ Errore parsing punto dati: {e}")
                    
                    if metric_rows > 0:
                        successful_metrics += 1
                        print(f"  ✅ Scritte {metric_rows} righe per {metric}")
                    else:
                        print(f"  ⚠️ Nessun dato trovato per {metric}")
                        
                else:
                    print(f"  ❌ Errore nella risposta per {metric}: {data}")
                    
            else:
                print(f"  ❌ Errore HTTP per {metric}: {resp.status_code}")
                if resp.status_code == 400:
                    try:
                        error_detail = resp.json()
                        print(f"    📝 Dettagli: {error_detail}")
                    except:
                        print(f"    📝 Testo errore: {resp.text}")
                        
        except Exception as e:
            print(f"  💥 Eccezione per {metric}: {str(e)}")
    
    print(f"\n📊 RIASSUNTO:")
    print(f"  • Metriche totali: {len(metrics)}")
    print(f"  • Metriche con dati: {successful_metrics}")
    print(f"  • Righe totali scritte: {total_rows}")

print(f"📂 Dati salvati in {OUTPUT_FILE}")