import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import random

# Configurazione
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEQ_LENGTH = 20
PRED_HORIZON = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 100
BATCH_SIZE = 32
THRESHOLD_FPS = 30  # FPS minimo accettabile
NODE_SWITCH_COOLDOWN = 5  # secondi prima di poter cambiare nodo

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
    
    # Calcola metriche aggregate
    features = []
    
    # CPU medio per node1 e node2
    node1_cpu_mean = df[node1_cols['cpu_avg']].mean(axis=1, skipna=True)
    node2_cpu_mean = df[node2_cols['cpu_avg']].mean(axis=1, skipna=True)
    
    # Memory per node1 e node2
    node1_memory = df[node1_cols['memory']]
    node2_memory = df[node2_cols['memory']]
    
    # Latenze (per calcolare FPS)
    frame_latency_cols = [col for col in df.columns if 'frame_classify_latency' in col]
    detector_latency_cols = [col for col in df.columns if 'detector_frame_latency' in col]
    
    # Calcola latenza media 
    frame_latency_mean = df[frame_latency_cols].mean(axis=1, skipna=True)
    detector_latency_mean = df[detector_latency_cols].mean(axis=1, skipna=True)
    
    # Converti latency in FPS approssimato (1/latency)
    # Usa un valore di fallback se latency è 0 o NaN
    fps_from_frame_latency = 1.0 / (frame_latency_mean + 1e-6)  # evita divisione per zero
    fps_from_detector_latency = 1.0 / (detector_latency_mean + 1e-6)
    
    # Clip FPS a valori realistici
    fps_from_frame_latency = np.clip(fps_from_frame_latency, 1, 120)
    fps_from_detector_latency = np.clip(fps_from_detector_latency, 1, 120)
    
    # Combina features: [node1_cpu, node2_cpu, node1_mem, node2_mem, fps_metric]
    data = np.column_stack([
        node1_cpu_mean.fillna(0),
        node2_cpu_mean.fillna(0), 
        node1_memory.fillna(0),
        node2_memory.fillna(0),
        fps_from_frame_latency.fillna(30)  # FPS come target
    ])
    
    # Rimuovi righe con troppi valori mancanti
    valid_rows = ~np.isnan(data).any(axis=1)
    data = data[valid_rows]
    
    print(f"Dati processati: {data.shape}")
    print(f"Features: ['Node1_CPU', 'Node2_CPU', 'Node1_Memory', 'Node2_Memory', 'FPS']")
    print(f"Range FPS: {data[:, -1].min():.2f} - {data[:, -1].max():.2f}")
    print(f"Range CPU Node1: {data[:, 0].min():.4f} - {data[:, 0].max():.4f}")
    print(f"Range CPU Node2: {data[:, 1].min():.4f} - {data[:, 1].max():.4f}")
    
    return data

def generate_sample_data(num_samples=1000, num_features=5):
    """
    Genera dati di esempio per il training (fallback se non ci sono dati reali)
    Features: CPU_usage, Memory_usage, Network_load, GPU_temp, FPS
    """
    np.random.seed(42)
    
    # Simula andamento temporale con trend e seasonalità
    t = np.arange(num_samples)
    
    # CPU usage (30-90%)
    cpu = 60 + 20 * np.sin(t * 0.01) + np.random.normal(0, 5, num_samples)
    cpu = np.clip(cpu, 20, 95)
    
    # Memory usage (40-85%)
    memory = 60 + 15 * np.sin(t * 0.008 + 1) + np.random.normal(0, 3, num_samples)
    memory = np.clip(memory, 35, 90)
    
    # Network load (0-100 Mbps)
    network = 50 + 30 * np.sin(t * 0.005 + 2) + np.random.normal(0, 10, num_samples)
    network = np.clip(network, 0, 100)
    
    # GPU temperature (40-80°C)
    gpu_temp = 55 + 15 * np.sin(t * 0.007 + 0.5) + np.random.normal(0, 2, num_samples)
    gpu_temp = np.clip(gpu_temp, 35, 85)
    
    # FPS - correlato inversamente con CPU, Memory e GPU temp
    fps_base = 60 - 0.3 * cpu - 0.2 * memory - 0.4 * gpu_temp + 0.1 * network
    fps = fps_base + np.random.normal(0, 2, num_samples)
    fps = np.clip(fps, 10, 60)
    
    data = np.column_stack([cpu, memory, network, gpu_temp, fps])
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
    """
    Training del modello LSTM
    """
    model = LSTMModelMultiStep(input_size, HIDDEN_SIZE, NUM_LAYERS, PRED_HORIZON).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Converti in tensori
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(DEVICE)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(DEVICE)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(DEVICE)
    
    train_losses = []
    val_losses = []
    
    print(f"Training su {DEVICE}")
    print(f"Dati training: {X_train.shape}, Validation: {X_val.shape}")
    
    for epoch in range(EPOCHS):
        model.train()
        
        # Training
        optimizer.zero_grad()
        output = model(X_train_tensor)
        loss = criterion(output, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_loss = criterion(val_output, y_val_tensor)
        
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        
        if epoch % 10 == 0:
            print(f'Epoca {epoch}/{EPOCHS}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
    
    return model, train_losses, val_losses

def decide_node(fps_prediction, threshold_fps, current_node, time_on_node, available_nodes=None):
    """
    Decide se cambiare nodo basandosi sulla predizione FPS
    
    Args:
        fps_prediction: FPS predetto (può essere min, mean, o array)
        threshold_fps: soglia FPS minima accettabile
        current_node: nodo attualmente in uso
        time_on_node: tempo trascorso sul nodo corrente
        available_nodes: lista dei nodi disponibili
        
    Returns:
        action: 'keep', 'switch', 'optimize'
        new_node: nuovo nodo (se switch)
        updated_time: tempo aggiornato
    """
    if available_nodes is None:
        available_nodes = ['group1-node1', 'group1-node2']
    
    current_time = time.time()
    
    # Se FPS predetto è sotto soglia e sono passati abbastanza secondi dal ultimo switch
    if fps_prediction < threshold_fps and time_on_node >= NODE_SWITCH_COOLDOWN:
        # Scegli un nodo diverso (round-robin semplice)
        available_alternatives = [n for n in available_nodes if n != current_node]
        if available_alternatives:
            new_node = random.choice(available_alternatives)
            print(f"⚠️  FPS predetto ({fps_prediction:.1f}) sotto soglia ({threshold_fps}). Switch da {current_node} a {new_node}")
            return 'switch', new_node, 0  # reset time_on_node
    
    # Se FPS è buono ma potrebbe migliorare
    elif fps_prediction > threshold_fps * 1.2:  # 20% sopra soglia
        print(f"✅ FPS predetto ({fps_prediction:.1f}) buono su {current_node}")
        return 'keep', current_node, time_on_node + 1
    
    # FPS accettabile ma non ottimale
    else:
        print(f"⚡ FPS predetto ({fps_prediction:.1f}) accettabile su {current_node}, possibile ottimizzazione")
        return 'optimize', current_node, time_on_node + 1
    
    return 'keep', current_node, time_on_node + 1

def simulate_real_time_prediction(model, scaler, initial_data):
    """
    Simula predizioni in tempo reale e decisioni di switching
    """
    current_node = 'group1-node1'
    time_on_node = 0
    
    # Buffer per mantenere le ultime SEQ_LENGTH osservazioni
    data_buffer = initial_data[-SEQ_LENGTH:].copy()
    
    print("🚀 Avvio simulazione predizioni FPS in tempo reale...")
    print(f"Nodo iniziale: {current_node}")
    print("-" * 60)
    
    for step in range(50):  # Simula 50 step temporali
        # Prepara input per predizione - applica scaler riga per riga poi reshape
        scaled_buffer = scaler.transform(data_buffer)  # shape: (SEQ_LENGTH, n_features)
        input_sequence = scaled_buffer.reshape(1, SEQ_LENGTH, -1)  # shape: (1, SEQ_LENGTH, n_features)
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).to(DEVICE)
        
        # Predizione multi-step
        model.eval()
        with torch.no_grad():
            fps_pred_multi = model(input_tensor).cpu().numpy().flatten()
        
        # Usa FPS minimo per decisione conservativa
        fps_min = fps_pred_multi.min()
        fps_mean = fps_pred_multi.mean()
        
        # Decisione node switching
        action, current_node, time_on_node = decide_node(
            fps_min, THRESHOLD_FPS, current_node, time_on_node
        )
        
        # Simula nuova osservazione (in realtà dovresti leggere dati reali)
        new_observation = generate_next_observation(data_buffer[-1])
        
        # Aggiorna buffer (sliding window)
        data_buffer = np.vstack([data_buffer[1:], new_observation])
        
        # Log stato
        if step % 5 == 0 or action == 'switch':
            print(f"Step {step:2d}: FPS pred min/mean: {fps_min:.1f}/{fps_mean:.1f} | "
                  f"Nodo: {current_node} | Azione: {action} | Tempo su nodo: {time_on_node}")
        
        time.sleep(0.1)  # Simula delay temporale
    
    print("-" * 60)
    print("✅ Simulazione completata")

def generate_next_observation(last_obs):
    """
    Genera la prossima osservazione basandosi sull'ultima
    (In un caso reale, questo verrebbe da sensori/monitoring)
    """
    noise_scale = 0.05  # 5% di rumore
    next_obs = last_obs + np.random.normal(0, noise_scale, last_obs.shape)
    
    # Applica bounds realistici
    next_obs[0] = np.clip(next_obs[0], 20, 95)   # CPU %
    next_obs[1] = np.clip(next_obs[1], 35, 90)   # Memory %
    next_obs[2] = np.clip(next_obs[2], 0, 100)   # Network Mbps
    next_obs[3] = np.clip(next_obs[3], 35, 85)   # GPU temp °C
    next_obs[4] = np.clip(next_obs[4], 10, 60)   # FPS
    
    return next_obs

def main():
    """
    Funzione principale
    """
    print("🎮 Sistema di Predizione FPS e Node Switching")
    print("=" * 60)
    
    # 1. Carica dati reali
    csv_path = "/Users/mariobarbieri/Developer/hack/Group1/Export Data-data-as-joinbyfield-2025-09-26 21_04_20.csv"
    try:
        print("📊 Caricamento dati reali...")
        data = load_real_data(csv_path)
        print(f"Dati caricati: {data.shape}")
    except Exception as e:
        print(f"⚠️  Errore caricamento dati reali: {e}")
        print("📊 Uso dati sintetici come fallback...")
        data = generate_sample_data(2000, 5)
        print(f"Dati generati: {data.shape}")
    
    # 2. Prepara dati per training
    print("🔄 Preparazione dati...")
    X_train, X_val, y_train, y_val, scaler = prepare_data(data)
    input_size = X_train.shape[2]  # numero di features
    
    # 3. Training del modello
    print("🧠 Training modello LSTM...")
    model, train_losses, val_losses = train_model(X_train, y_train, X_val, y_val, input_size)
    
    print(f"Training completato! Loss finale: {train_losses[-1]:.4f}")
    
    # 4. Simulazione tempo reale
    print("⏱️  Avvio simulazione tempo reale...")
    simulate_real_time_prediction(model, scaler, data)

if __name__ == "__main__":
    main()