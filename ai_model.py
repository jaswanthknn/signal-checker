import os
import argparse
import numpy as np
import joblib
import subprocess
import re
import time
import platform
from scipy.stats import skew, kurtosis
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_absolute_error

# --- Configuration ---
MODEL_DIR = 'models'
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'network_classifier.joblib')
REGRESSOR_PATH = os.path.join(MODEL_DIR, 'network_regressor.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'network_scaler.joblib')
FFT_SIZE = 1024

# ------------------------------------
# 1. SYNTHETIC DATA GENERATION
# ------------------------------------

def generate_stable_signal(length, base_latency=30, noise=5):
    """Generates a stable signal with low jitter."""
    return np.random.normal(loc=base_latency, scale=noise, size=length)

def add_jitter(signal, start_factor=0.5, jitter_scale=30):
    """Adds high jitter (instability) to the latter part of a signal."""
    start_index = int(len(signal) * start_factor)
    jitter_part = np.random.normal(
        loc=np.mean(signal[:start_index]),
        scale=jitter_scale,
        size=len(signal) - start_index
    )
    signal[start_index:] = jitter_part
    return signal, start_index

def add_spikes(signal, num_spikes=3, spike_height=200):
    """Adds random latency spikes."""
    for _ in range(num_spikes):
        idx = np.random.randint(0, len(signal))
        signal[idx] += np.random.uniform(spike_height * 0.8, spike_height * 1.2)
    return signal, 0

def add_dropout_trend(signal, start_factor=0.6, final_latency=500):
    """Adds a trend of increasing latency simulating a drop-out."""
    start_index = int(len(signal) * start_factor)
    end_index = len(signal)
    dropout_len = end_index - start_index
    trend = np.linspace(0, final_latency - np.mean(signal), dropout_len)
    signal[start_index:] += trend
    return signal, start_index

def generate_network_dataset(n_samples=2000, length=100):
    """Generates a dataset of synthetic network latency signals."""
    signals, is_degraded_labels, degradation_time_labels = [], [], []
    print(f"Generating {n_samples} synthetic network signals...")
    for _ in range(n_samples):
        stable_signal = generate_stable_signal(length)
        degradation_type = np.random.choice(['stable', 'jitter', 'spikes', 'dropout'])
        if degradation_type == 'stable':
            signal, is_degraded, start_time = stable_signal, 0, -1
        else:
            is_degraded = 1
            if degradation_type == 'jitter':
                signal, start_time = add_jitter(stable_signal.copy())
            elif degradation_type == 'spikes':
                signal, start_time = add_spikes(stable_signal.copy())
            else:
                signal, start_time = add_dropout_trend(stable_signal.copy())
        signals.append(signal)
        is_degraded_labels.append(is_degraded)
        degradation_time_labels.append(start_time)
    return np.array(signals), np.array(is_degraded_labels), np.array(degradation_time_labels)

# ------------------------------------
# 2. FEATURE EXTRACTION
# ------------------------------------

def extract_features(signal, sr=10):
    """Extracts a feature vector from a single signal."""
    mean_val, std_dev, skewness, kurt_val = np.mean(signal), np.std(signal), skew(signal), kurtosis(signal)
    max_val, min_val = np.max(signal), np.min(signal)
    fft_spectrum = np.abs(np.fft.rfft(signal, FFT_SIZE))
    fft_freqs = np.fft.rfftfreq(FFT_SIZE, d=1./sr)
    spectral_centroid = np.sum(fft_freqs * fft_spectrum) / np.sum(fft_spectrum) if np.sum(fft_spectrum) > 1e-6 else 0
    return np.array([mean_val, std_dev, skewness, kurt_val, max_val, min_val, spectral_centroid])

# ------------------------------------
# 3. MODEL TRAINING
# ------------------------------------

def train_models(signals, y_class, y_reg):
    """Trains classification and regression models."""
    print("Extracting features from signals...")
    X = np.array([extract_features(s) for s in signals])
    
    print("--- Training Degradation Classifier ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.25, random_state=42, stratify=y_class)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    classifier.fit(X_train_scaled, y_train)
    print("\nClassifier Report:\n", classification_report(y_test, classifier.predict(X_test_scaled)))
    
    print("--- Training Degradation Time Regressor ---")
    degraded_indices = np.where(y_class == 1)[0]
    if len(degraded_indices) > 0:
        X_degraded, y_degraded_time = X[degraded_indices], y_reg[degraded_indices]
        X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_degraded, y_degraded_time, test_size=0.25, random_state=42)
        X_train_reg_scaled, X_test_reg_scaled = scaler.transform(X_train_reg), scaler.transform(X_test_reg)
        regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        regressor.fit(X_train_reg_scaled, y_train_reg)
        mae = mean_absolute_error(y_test_reg, regressor.predict(X_test_reg_scaled))
        print(f"Regressor Mean Absolute Error (MAE): {mae:.2f} samples")
    else:
        regressor = None
        print("No degraded samples, skipping regressor training.")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(regressor, REGRESSOR_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Models saved to '{MODEL_DIR}' directory.")

# ------------------------------------
# 4. LIVE PREDICTION & NETWORK SCANNING
# ------------------------------------

def get_network_latency(host="8.8.8.8", count=1):
    """Pings a host to get network latency. Returns latency in ms or None."""
    try:
        param = '-n' if platform.system().lower() == 'windows' else '-c'
        command = ['ping', param, str(count), host]
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, timeout=2).decode(errors='ignore')
        avg_match = re.search(r"(?:Average|Moyenne|Mittelwert)\s*=\s*([\d\.]+)\s*ms", result)
        if avg_match: return float(avg_match.group(1))
        times = re.findall(r"(?:time|temps|zeit)[=<]([\d\.]+)\s*ms", result)
        if times: return np.mean([float(t) for t in times])
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return 1000.0
    return None

def scan_wifi_networks():
    """Scans for WiFi networks and returns a dictionary of SSIDs and signal strengths."""
    os_name = platform.system().lower()
    networks = {}
    print(f"Scanning for networks on {os_name.capitalize()}...")
    try:
        if os_name == "windows":
            output = subprocess.check_output(["netsh", "wlan", "show", "network", "mode=Bssid"], encoding='cp850', errors='ignore')
            ssids = re.findall(r"SSID \d+ : (.+)", output)
            signals = re.findall(r"Signal\s+: (\d+)%", output)
            for ssid, signal in zip(ssids, signals):
                if ssid.strip() not in networks or networks.get(ssid.strip(), 0) < int(signal):
                    networks[ssid.strip()] = int(signal)
       
        
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not scan for networks: {e}")
    return networks

def find_best_network():
    """Scans networks, measures latency, and recommends the best one (Windows only)."""
    available_networks = scan_wifi_networks()
    if not available_networks:
        print("No Wi-Fi networks found.")
        return None

    results = [{'ssid': ssid, 'strength': strength, 'latency': float('inf')} 
               for ssid, strength in available_networks.items()]
    
    print("Testing latency of current connection...")
    current_latency = get_network_latency(count=3)
    if current_latency is not None:
        print(f"Current connection latency: {current_latency:.2f} ms")
        if results:
            best_signal_net = max(results, key=lambda x: x['strength'])
            best_signal_net['latency'] = current_latency
    else:
        print("Could not measure latency for the current connection.")

    # scoring
    for net in results:
        norm_latency = min(net['latency'], 1000) / 1000.0
        norm_strength = net['strength'] / 100.0
        net['score'] = (0.7 * norm_strength) - (0.3 * norm_latency)
    
    sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)

    # print ranking
    print("\n--- Network Ranking ---")
    print(f"{'Rank':<5} {'SSID':<30} {'Signal (%)':<15} {'Latency (ms)':<15}")
    print("-" * 70)
    for i, net in enumerate(sorted_results):
        latency_str = f"{net['latency']:.2f}" if net['latency'] != float('inf') else "N/A"
        print(f"{i+1:<5} {net['ssid']:<30} {net['strength']:<15} {latency_str:<15}")
    
    best_net = sorted_results[0]
    print("\n" + "="*30 + f"\n🏆 Recommended Network: '{best_net['ssid']}'\n" + "="*30 + "\n")
    return best_net


def connect_to_network(ssid):
    """Attempts to connect to a specified Wi-Fi SSID using native OS commands."""
    os_name = platform.system().lower()
    try:
        print(f"Attempting to connect to '{ssid}' on {os_name.capitalize()}...")
        if os_name == "windows":
            command = ["netsh", "wlan", "connect", f"name={ssid}"]
            output = subprocess.check_output(command, stderr=subprocess.STDOUT, encoding='cp850', errors='ignore')
            if "Connection request was completed successfully" in output:
                return True, f"Successfully sent connection request to {ssid}."
            else:
                return False, f"Could not connect. Windows output: {output}"
        
       
        else:
            return False, f"Unsupported operating system: {os_name}"
    except subprocess.CalledProcessError as e:
        error_message = e.output
        if "you must run this command as an administrator" in error_message.lower() or "run as root" in error_message.lower():
            return False, "Connection failed. Please run with administrator/sudo privileges."
        return False, f"Failed to execute connection command. Error: {error_message}"
    except FileNotFoundError:
        return False, "Network command-line tools not found (e.g., netsh, nmcli)."
    except Exception as e:
        return False, f"An unexpected error occurred: {e}"

class NetworkPredictor:
    """Loads trained models to make predictions on live network data."""
    def __init__(self):
        try:
            self.classifier = joblib.load(CLASSIFIER_PATH)
            self.regressor = joblib.load(REGRESSOR_PATH)
            self.scaler = joblib.load(SCALER_PATH)
            print("Models for degradation analysis loaded successfully.")
        except FileNotFoundError:
            raise FileNotFoundError("Model files not found! Please run training first with '--step all'.")

    def predict(self, signal):
        """Predicts degradation on a captured signal."""
        features = extract_features(np.array(signal)).reshape(1, -1)
        features_scaled = self.scaler.transform(features)
        class_prediction = self.classifier.predict(features_scaled)[0]
        if class_prediction == 0:
            return "Prediction: Network signal is STABLE."
        else:
            time_prediction = self.regressor.predict(features_scaled)[0]
            return f"Prediction: Network is DEGRADING. Instability likely started around sample index: {int(time_prediction)}"

# ------------------------------------
# 5. COMMAND-LINE INTERFACE
# ------------------------------------

def main():
    parser = argparse.ArgumentParser(description="AI for Network Signal Optimization and Prediction.")
    parser.add_argument('--step', type=str, choices=['all', 'live_predict'], required=True,
                        help="'all' to train models. 'live_predict' to optimize and analyze network.")
    args = parser.parse_args()

    if args.step == 'all':
        signals, y_class, y_reg = generate_network_dataset()
        train_models(signals, y_class, y_reg)
        print("\nTraining complete! You can now run 'live_predict'.")

    elif args.step == 'live_predict':
        best_net = find_best_network()
        
        if best_net and best_net.get('ssid'):
            print(f"\nAttempting to auto-connect to best network: '{best_net['ssid']}'...")
            # IMPORTANT: This requires admin/sudo privileges to run successfully.
            success, message = connect_to_network(best_net['ssid'])
            print(f"Connection attempt status: {message}")
            if success:
                print("Waiting 5 seconds for connection to establish...")
                time.sleep(5)
        else:
            print("\nCould not determine a best network to connect to. Analyzing current connection.")
        
        try:
            predictor = NetworkPredictor()
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
            
        duration, sampling_rate = 10, 2
        n_samples = duration * sampling_rate
        
        print(f"\n📶 Capturing current network latency for {duration}s...")
        latency_signal = []
        for i in range(n_samples):
            latency = get_network_latency()
            if latency is not None:
                print(f"   Sample {i+1}/{n_samples}: {latency:.2f} ms")
                latency_signal.append(latency)
            else:
                print(f"   Sample {i+1}/{n_samples}: Ping failed")
                latency_signal.append(1000.0)
            time.sleep(1 / sampling_rate)

        if len(latency_signal) < 10:
            print("\nCould not capture enough data for a reliable prediction.")
            return

        print("\nAnalyzing captured data for instability...")
        result = predictor.predict(latency_signal)
        print("\n" + "="*40 + "\nDegradation Analysis Result\n" + result + "\n" + "="*40)

if __name__ == '__main__':
    main()