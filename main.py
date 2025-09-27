import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Configurazione generale del modello
DEVICE = _default_device()
SEQ_LENGTH = 20
PRED_HORIZON = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 500
BATCH_SIZE = 32

def create_sequences_multi_step(data, seq_length, pred_horizon=10):
    """
    data: array (num_samples, num_features)
    seq_length: lunghezza della finestra di input
    pred_horizon: numero di step futuri da predire
    """
    X, y = [], []
    for i in range(len(data) - seq_length - pred_horizon + 1):
        X.append(data[i:i+seq_length, :])
        y.append(data[i+seq_length:i+seq_length+pred_horizon, -1])  # FPS ultima colonna
    return np.array(X), np.array(y)

class LSTMModelMultiStep(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, pred_horizon=10):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_horizon)  # output multi-step
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # prendi l'ultimo timestep per predire tutti i futuri
        return out

def load_real_data(csv_path):
    """
    Carica e preprocessa i dati reali dal CSV
    """
    print(f"📂 Caricamento dati da: {csv_path}")
    
    # Leggi CSV
    df = pd.read_csv(csv_path)
    print(f"Shape originale: {df.shape}")
    print(f"Colonne disponibili: {len(df.columns)}")
    
    # Converti timestamp
    df['Time'] = pd.to_datetime(df['Time'])
    df = df.sort_values('Time').reset_index(drop=True)
    
    # Seleziona colonne principali per i due nodi
    node1_cols = {
        'cpu_avg': [f'group1-node1_cpu_{i}' for i in range(8)],
        'memory': 'group1-node1_available_memory'
    }
    
    node2_cols = {
        'cpu_avg': [f'group1-node2_cpu_{i}' for i in range(8)],  
        'memory': 'group1-node2_available_memory'
    }
    
    # === ESTRAZIONE FEATURES AVANZATE ===
    
    # 1. CPU medio per node1 e node2
    node1_cpu_mean = df[node1_cols['cpu_avg']].mean(axis=1, skipna=True)
    node2_cpu_mean = df[node2_cols['cpu_avg']].mean(axis=1, skipna=True)
    
    # 2. Memory disponibile per entrambi i nodi
    node1_memory = df[node1_cols['memory']]
    node2_memory = df[node2_cols['memory']]
    
    # 3. People box count (indicatore di carico di lavoro)
    box_count_cols = [col for col in df.columns if 'people_box_count' in col]
    if box_count_cols:
        box_count_mean = df[box_count_cols].mean(axis=1, skipna=True)
    else:
        box_count_mean = pd.Series([0] * len(df))
    
    # 4. Frame classify latency (latenza di classificazione)
    frame_latency_cols = [col for col in df.columns if 'frame_classify_latency' in col]
    print(f"Trovate colonne frame_classify_latency: {frame_latency_cols}")
    if frame_latency_cols:
        frame_classify_latency_mean = df[frame_latency_cols].mean(axis=1, skipna=True)
    else:
        frame_classify_latency_mean = pd.Series([0.01] * len(df))
    
    # 5. Detector frame latency (per confronto)
    detector_latency_cols = [col for col in df.columns if 'detector_frame_latency' in col]
    if detector_latency_cols:
        detector_latency_mean = df[detector_latency_cols].mean(axis=1, skipna=True)
    else:
        detector_latency_mean = pd.Series([0.01] * len(df))
    
    # 6. CPU seconds total (se disponibile)
    cpu_seconds_total_cols = [col for col in df.columns if 'cpu_seconds_total' in col]
    if cpu_seconds_total_cols:
        cpu_seconds_total = df[cpu_seconds_total_cols].sum(axis=1, skipna=True)
    else:
        # Simula CPU seconds total come derivato dal CPU usage
        cpu_seconds_total = (node1_cpu_mean + node2_cpu_mean) * 20  # stima basata su CPU usage
    
    # === CALCOLO FPS ===
    # Converti latency in FPS approssimato (1/latency)
    fps_from_frame_latency = 1000.0 / (frame_classify_latency_mean + 1e-6)
    fps_from_detector_latency = 1000.0 / (detector_latency_mean + 1e-6)
    
    # Clip FPS a valori realistici
    fps_from_frame_latency = np.clip(fps_from_frame_latency, 1, 120)
    fps_from_detector_latency = np.clip(fps_from_detector_latency, 1, 120)
    
    # Usa FPS da frame classify latency come target principale
    fps_target = fps_from_frame_latency
    
    # === COMBINA TUTTE LE FEATURES ===
    data = np.column_stack([
        node1_cpu_mean.fillna(0),                    # Feature 0: Node1 CPU
        node2_cpu_mean.fillna(0),                    # Feature 1: Node2 CPU  
        node1_memory.fillna(0),                      # Feature 2: Node1 Memory
        node2_memory.fillna(0),                      # Feature 3: Node2 Memory
        box_count_mean.fillna(0),                    # Feature 4: People box count
        frame_classify_latency_mean.fillna(0.01),    # Feature 5: Frame classify latency
        detector_latency_mean.fillna(0.01),          # Feature 6: Detector latency
        cpu_seconds_total.fillna(0),                 # Feature 7: CPU seconds total
        fps_target.fillna(30)                        # Feature 8: FPS (target)
    ])
    
    # Rimuovi righe con troppi valori mancanti
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]
    
    print(f"Dati processati: {data.shape}")
    print("Features: ['Node1_CPU', 'Node2_CPU', 'Node1_Memory', 'Node2_Memory',")
    print("          'Box_Count', 'Frame_Classify_Latency', 'Detector_Latency', 'CPU_Seconds_Total', 'FPS']")
    print(f"📊 STATISTICHE FEATURES:")
    print(f"   FPS (target): {data[:, -1].min():.2f} - {data[:, -1].max():.2f}")
    print(f"   CPU Node1: {data[:, 0].min():.4f} - {data[:, 0].max():.4f}")
    print(f"   CPU Node2: {data[:, 1].min():.4f} - {data[:, 1].max():.4f}")
    print(f"   Memory Node1: {data[:, 2].min():.0f} - {data[:, 2].max():.0f}")
    print(f"   Memory Node2: {data[:, 3].min():.0f} - {data[:, 3].max():.0f}")
    print(f"   Box Count: {data[:, 4].min():.0f} - {data[:, 4].max():.0f}")
    print(f"   Frame Latency: {data[:, 5].min():.4f} - {data[:, 5].max():.4f}")
    print(f"   CPU Seconds Total: {data[:, 7].min():.2f} - {data[:, 7].max():.2f}")
    
    return data

def prepare_data(data, seq_length=SEQ_LENGTH, pred_horizon=PRED_HORIZON, train_split=0.8):
    """
    Prepara i dati per il training
    """
    # Normalizza i dati
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Crea sequenze
    X, y = create_sequences_multi_step(data_scaled, seq_length, pred_horizon)
    
    # Split train/validation
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val, scaler

def train_model(X_train, y_train, X_val, y_val, input_size):
    """Allena il modello LSTM multi-step e restituisce il modello addestrato."""
    model = LSTMModelMultiStep(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_HORIZON).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)

    print(f"Training su {DEVICE} (epochs={EPOCHS})")

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 25 == 0 or epoch == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(X_val_tensor)
                val_loss = criterion(val_output, y_val_tensor)
            print(f"Epoch {epoch + 1:03d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")

    return model


def predict_future_fps(model, scaler, recent_window):
    """Esegue una predizione multi-step a partire da una finestra di osservazioni grezze."""
    model.eval()

    if recent_window.shape[0] != SEQ_LENGTH:
        raise ValueError(f"recent_window must have {SEQ_LENGTH} rows (got {recent_window.shape[0]})")

    # Normalizza con lo stesso scaler usato in training
    scaled_window = scaler.transform(recent_window)
    input_tensor = torch.tensor(scaled_window.reshape(1, SEQ_LENGTH, -1), dtype=torch.float32).to(DEVICE)

    with torch.no_grad():
        preds_normalized = model(input_tensor).cpu().numpy().flatten()

    # Denormalizza le predizioni FPS (FPS è l'ultima feature, indice -1)
    fps_mean = scaler.mean_[-1]  # Media degli FPS durante il training
    fps_scale = scaler.scale_[-1]  # Deviazione standard degli FPS
    preds_denormalized = preds_normalized * fps_scale + fps_mean

    return preds_denormalized

def save_model_pkcl(model, scaler, filename: str = "trained_model.pkcl") -> str:
    """Salva lo state_dict del modello e lo scaler in un file .pkcl usando pickle.

    Restituisce il percorso assoluto del file salvato.
    """
    import os
    import pickle

    # Prepara il payload da salvare
    payload = {
        "model_state_dict": model.state_dict(),
        "scaler": scaler
    }

    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, filename)

    # Scrivi su disco
    with open(save_path, "wb") as f:
        pickle.dump(payload, f)

    return save_path


def load_model_pkcl(model_class, input_size, filename: str = "trained_model.pkcl"):
    """Carica model_state_dict e scaler da un file .pkcl e restituisce (model, scaler).

    model_class: callable da invocare come model_class(input_size, ...)
    input_size: numero di input features per ricreare il modello
    """
    import os
    import pickle
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(save_dir, filename)

    if not os.path.exists(save_path):
        raise FileNotFoundError(save_path)

    with open(save_path, "rb") as f:
        payload = pickle.load(f)

    state = payload.get("model_state_dict")
    scaler = payload.get("scaler")

    # Ricostruisci il modello con la stessa architettura
    model = model_class(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_HORIZON).to(DEVICE)
    if state is not None:
        model.load_state_dict(state)

    model.eval()
    return model, scaler

def main():
    """Esempio d'uso: preprocessing, training e predizione sulle ultime osservazioni."""
    csv_path = "/Users/mariobarbieri/Developer/hack/Group1/Export Data-data-as-joinbyfield-2025-09-26 21_04_20.csv"

    try:
        data = load_real_data(csv_path)
    except Exception as exc:  # noqa: BLE001
        print(f"⚠️  Errore nel caricamento del dataset reale ({exc}). Uso dati sintetici.")
        exit(1)
        
    

    # Prova a caricare un modello salvato; se non esiste, esegui prepare/train/save
    save_filename = "trained_model.pkcl"
    try:
        # Per ricreare il modello abbiamo bisogno di input_size; proviamo a derivarlo dai dati
        X_train, X_val, y_train, y_val, scaler = prepare_data(data)
        input_size = X_train.shape[2]

        try:
            model, scaler = load_model_pkcl(LSTMModelMultiStep, input_size, filename=save_filename)
            print(f"Caricato modello esistente da: {save_filename}")
        except FileNotFoundError:
            # Se non trovato, esegui allenamento
            model = train_model(X_train, y_train, X_val, y_val, input_size)
            recent_window = data[-SEQ_LENGTH:]
            future_fps = predict_future_fps(model, scaler, recent_window)
            print("Predizione FPS prossimi step:", np.round(future_fps, 2))

            # === Salva modello e scaler in un file .pkcl ===
            try:
                save_path = save_model_pkcl(model, scaler, filename=save_filename)
                print(f"Modello e scaler salvati in: {save_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"Errore nel salvataggio del modello: {exc}")
    except Exception as exc:  # noqa: BLE001
        print(f"Errore nella preparazione/caricamento del modello: {exc}")


if __name__ == "__main__":
    main()


