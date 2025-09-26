import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import time
import random

# Configurazione
DEVICE = torch.device('mps')
SEQ_LENGTH = 20
PRED_HORIZON = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
LEARNING_RATE = 0.001
EPOCHS = 1000
BATCH_SIZE = 32

# === CONFIGURAZIONE GESTIONE NODI ===
THRESHOLD_FPS_CRITICAL = 5   # Soglia critica - sotto questa usiamo nodo 2 immediatamente
THRESHOLD_FPS_LOW = 10          # Soglia bassa - consideriamo switch a nodo 2
THRESHOLD_FPS_GOOD = 15       # Soglia buona - possiamo tornare a nodo 1

# === CONFIGURAZIONE COSTI NODO 2 ===
NODE2_MAX_TIME_PER_HOUR = 300    # Massimo 5 minuti/ora su nodo 2 (costoso)
NODE2_MAX_CONSECUTIVE_TIME = 60   # Massimo 1 minuto consecutivo su nodo 2
NODE2_COST_PER_SECOND = 0.10     # Costo in € per secondo
NODE2_DAILY_BUDGET = 50.0        # Budget giornaliero in €

NODE_SWITCH_COOLDOWN = 10     # secondi prima di poter cambiare nodo (aumentato)
PERFORMANCE_CHECK_INTERVAL = 5 # ogni quanti secondi verificare performance

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

class CostManager:
    """Gestisce i costi e l'utilizzo del nodo 2"""
    
    def __init__(self):
        self.reset_daily_stats()
        self.hour_usage = {}  # utilizzo per ora
        self.consecutive_time_on_node2 = 0
        
    def reset_daily_stats(self):
        """Reset statistiche giornaliere"""
        self.daily_cost = 0.0
        self.daily_node2_time = 0
        self.current_hour = None
        
    def can_use_node2(self, fps_prediction, current_time_on_node2=0):
        """Verifica se possiamo usare il nodo 2 considerando i vincoli di costo"""
        current_hour = time.strftime("%H")
        
        # Inizializza utilizzo orario se necessario
        if current_hour not in self.hour_usage:
            self.hour_usage[current_hour] = 0
            
        # Check budget giornaliero
        if self.daily_cost >= NODE2_DAILY_BUDGET:
            return False, "💰 Budget giornaliero esaurito"
            
        # Check utilizzo orario
        if self.hour_usage[current_hour] >= NODE2_MAX_TIME_PER_HOUR:
            return False, f"⏱️ Utilizzo orario massimo raggiunto ({NODE2_MAX_TIME_PER_HOUR}s/h)"
            
        # Check tempo consecutivo
        if self.consecutive_time_on_node2 >= NODE2_MAX_CONSECUTIVE_TIME:
            return False, f"⏳ Tempo consecutivo massimo raggiunto ({NODE2_MAX_CONSECUTIVE_TIME}s)"
            
        # Check soglia critica - in questo caso forziamo l'uso
        if fps_prediction < THRESHOLD_FPS_CRITICAL:
            return True, "🚨 FPS critico - switch forzato a nodo 2"
            
        # Check soglia bassa - switch solo se abbiamo budget
        if fps_prediction < THRESHOLD_FPS_LOW:
            remaining_budget = NODE2_DAILY_BUDGET - self.daily_cost
            estimated_cost = NODE2_COST_PER_SECOND * 30  # stima 30 secondi
            if estimated_cost <= remaining_budget:
                return True, f"⚠️ FPS basso - switch economico (costo stimato: €{estimated_cost:.2f})"
            else:
                return False, f"💸 FPS basso ma budget insufficiente (rimangono €{remaining_budget:.2f})"
                
        return False, "✅ FPS accettabile - resta su nodo 1"
        
    def update_usage(self, node_name, seconds=1):
        """Aggiorna statistiche di utilizzo"""
        if node_name == 'group1-node2':
            current_hour = time.strftime("%H")
            if current_hour not in self.hour_usage:
                self.hour_usage[current_hour] = 0
                
            self.hour_usage[current_hour] += seconds
            self.daily_node2_time += seconds
            self.daily_cost += NODE2_COST_PER_SECOND * seconds
            self.consecutive_time_on_node2 += seconds
        else:
            self.consecutive_time_on_node2 = 0  # Reset se non siamo su nodo 2
            
    def get_cost_report(self):
        """Genera report sui costi"""
        current_hour = time.strftime("%H")
        hour_usage = self.hour_usage.get(current_hour, 0)
        
        return {
            'daily_cost': self.daily_cost,
            'daily_budget_remaining': NODE2_DAILY_BUDGET - self.daily_cost,
            'hour_usage': hour_usage,
            'hour_budget_remaining': NODE2_MAX_TIME_PER_HOUR - hour_usage,
            'consecutive_time': self.consecutive_time_on_node2,
            'consecutive_remaining': NODE2_MAX_CONSECUTIVE_TIME - self.consecutive_time_on_node2
        }

def decide_node_with_cost_control(fps_prediction, current_node, time_on_node, cost_manager):
    """
    Decide se cambiare nodo considerando FPS e vincoli di costo per nodo 2
    
    Args:
        fps_prediction: FPS predetto
        current_node: nodo attualmente in uso  
        time_on_node: tempo trascorso sul nodo corrente
        cost_manager: gestore dei costi
        
    Returns:
        action: 'keep', 'switch_to_node2', 'switch_to_node1', 'force_node1'
        new_node: nuovo nodo
        updated_time: tempo aggiornato
        reason: motivo della decisione
    """
    
    # === LOGICA DI DECISIONE ===
    
    # 1. Se siamo già su nodo 2, verifichiamo se possiamo continuare
    if current_node == 'group1-node2':
        
        # Se FPS è tornato buono, torna a nodo 1 per risparmiare
        if fps_prediction >= THRESHOLD_FPS_GOOD and time_on_node >= 3:  # Switch più veloce
            print(f"💚 FPS migliorato ({fps_prediction:.1f}≥{THRESHOLD_FPS_GOOD}) - torno a nodo 1 per risparmiare")
            return 'switch_to_node1', 'group1-node1', 0, "risparmio_costo"
        
        # Strategia aggressiva: dopo 20s su nodo 2, torna a nodo 1 se FPS non è critico
        if time_on_node >= 20 and fps_prediction >= THRESHOLD_FPS_CRITICAL:
            print(f"💸 Limito costo: 20s su nodo 2, FPS non critico ({fps_prediction:.1f}) - torno a nodo 1")
            return 'switch_to_node1', 'group1-node1', 0, "limite_costo_preventivo"
            
        # Se siamo al limite del tempo consecutivo, forza switch a nodo 1
        if cost_manager.consecutive_time_on_node2 >= NODE2_MAX_CONSECUTIVE_TIME:
            print(f"⏱️ Limite tempo consecutivo raggiunto ({NODE2_MAX_CONSECUTIVE_TIME}s) - forzo switch a nodo 1")
            return 'force_node1', 'group1-node1', 0, "limite_tempo_consecutivo"
            
        # Altrimenti continua su nodo 2
        print(f"🔄 Continuo su nodo 2 - FPS: {fps_prediction:.1f} (t={time_on_node}s)")
        return 'keep', current_node, time_on_node + 1, "performance_non_ottimale"
    
    # 2. Se siamo su nodo 1, verifichiamo se serve switch a nodo 2
    else:  # current_node == 'group1-node1'
        
        # Se FPS è buono, resta su nodo 1
        if fps_prediction >= THRESHOLD_FPS_GOOD:
            print(f"✅ FPS buono ({fps_prediction:.1f}) su nodo 1 - continuo")
            return 'keep', current_node, time_on_node + 1, "performance_buona"
            
        # Se FPS è problematico, controlla se possiamo usare nodo 2
        elif fps_prediction < THRESHOLD_FPS_LOW and time_on_node >= NODE_SWITCH_COOLDOWN:
            
            can_use, reason = cost_manager.can_use_node2(fps_prediction, 0)
            
            if can_use:
                print(f"⚠️ {reason}")
                return 'switch_to_node2', 'group1-node2', 0, "fps_basso_switch_permesso"
            else:
                print(f"⚠️ FPS basso ({fps_prediction:.1f}) ma {reason}")
                return 'keep', current_node, time_on_node + 1, "fps_basso_ma_vincoli_costo"
        
        # FPS non ottimale ma non abbastanza basso per switch
        else:
            print(f"⚡ FPS accettabile ({fps_prediction:.1f}) su nodo 1")
            return 'keep', current_node, time_on_node + 1, "performance_accettabile"

def simulate_real_time_prediction_with_cost_control(model, scaler, initial_data):
    """
    Simula predizioni in tempo reale con gestione costi del nodo 2
    """
    current_node = 'group1-node1'  # Inizia sempre con nodo economico
    time_on_node = 0
    cost_manager = CostManager()
    
    # Buffer per mantenere le ultime SEQ_LENGTH osservazioni
    data_buffer = initial_data[-SEQ_LENGTH:].copy()
    
    print("🚀 Simulazione FPS con Gestione Costi Nodo 2")
    print("=" * 70)
    print(f"💰 Budget giornaliero: €{NODE2_DAILY_BUDGET}")
    print(f"⏱️  Limite orario nodo 2: {NODE2_MAX_TIME_PER_HOUR}s/h")
    print(f"⏳ Limite consecutivo nodo 2: {NODE2_MAX_CONSECUTIVE_TIME}s")
    print(f"🎯 Soglie FPS: Critica={THRESHOLD_FPS_CRITICAL}, Bassa={THRESHOLD_FPS_LOW}, Buona={THRESHOLD_FPS_GOOD}")
    print(f"🏁 Nodo iniziale: {current_node}")
    print("-" * 70)
    
    for step in range(50):  # Simula 50 step temporali
        # Prepara input per predizione
        scaled_buffer = scaler.transform(data_buffer)
        input_sequence = scaled_buffer.reshape(1, SEQ_LENGTH, -1)
        input_tensor = torch.tensor(input_sequence, dtype=torch.float32).to(DEVICE)
        
        # Predizione multi-step
        model.eval()
        with torch.no_grad():
            fps_pred_multi = model(input_tensor).cpu().numpy().flatten()
        
        # Usa FPS minimo per decisione conservativa
        fps_min = fps_pred_multi.min()
        fps_mean = fps_pred_multi.mean()
        
        # Simula FPS variabili per testare meglio la logica
        if step < 15:
            fps_simulated = 12 + random.uniform(-2, 2)  # FPS critici inizialmente
        elif step < 25:
            fps_simulated = 22 + random.uniform(-3, 3)  # FPS bassi 
        elif step < 35:
            fps_simulated = 38 + random.uniform(-5, 5)  # FPS buoni
        else:
            fps_simulated = 28 + random.uniform(-8, 8)  # FPS variabili
        
        # Decisione node switching con controllo costi
        action, new_node, new_time_on_node, reason = decide_node_with_cost_control(
            fps_simulated, current_node, time_on_node, cost_manager
        )
        
        # Aggiorna statistiche costi
        cost_manager.update_usage(current_node, 1)
        
        # Applica la decisione
        if new_node != current_node:
            print(f"🔄 SWITCH: {current_node} → {new_node} (Motivo: {reason})")
            
        current_node = new_node
        time_on_node = new_time_on_node
        
        # Simula nuova osservazione
        new_observation = generate_next_observation(data_buffer[-1])
        data_buffer = np.vstack([data_buffer[1:], new_observation])
        
        # Log dettagliato ogni 5 step o quando c'è un'azione importante
        if step % 5 == 0 or action.startswith('switch') or action == 'force_node1':
            cost_report = cost_manager.get_cost_report()
            
            print(f"Step {step:2d}: FPS {fps_simulated:.1f} | Nodo: {current_node} | "
                  f"Azione: {action}")
            print(f"        💰 Costo: €{cost_report['daily_cost']:.2f}/{NODE2_DAILY_BUDGET} | "
                  f"⏱️ Uso orario: {cost_report['hour_usage']}/{NODE2_MAX_TIME_PER_HOUR}s | "
                  f"⏳ Consecutivo: {cost_report['consecutive_time']}s")
        
        time.sleep(0.1)  # Simula delay temporale
    
    # Report finale
    final_report = cost_manager.get_cost_report()
    print("-" * 70)
    print("📊 REPORT FINALE COSTI")
    print(f"💰 Costo totale giornaliero: €{final_report['daily_cost']:.2f}")
    print(f"💳 Budget rimanente: €{final_report['daily_budget_remaining']:.2f}")
    print(f"⏱️  Tempo totale su nodo 2: {cost_manager.daily_node2_time}s")
    print(f"📈 Efficienza costo: {(final_report['daily_cost']/NODE2_DAILY_BUDGET*100):.1f}% del budget usato")
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
    
    # 4. Simulazione tempo reale con controllo costi
    print("⏱️  Avvio simulazione con gestione costi...")
    simulate_real_time_prediction_with_cost_control(model, scaler, data)

if __name__ == "__main__":
    main()