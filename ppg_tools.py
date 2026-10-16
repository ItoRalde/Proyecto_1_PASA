import hashlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt

def build_U(acc, M):
    """Fila n = [ejex(n), ejex(n-1), ..., ejex(n-M+1), ejey(n), ..., ejez(n-M+1)] (ceros antes de n=0)."""
    n_ref, N = acc.shape
    U = np.zeros((N, n_ref * M))
    for k in range(n_ref):
        padded = np.concatenate([np.zeros(M - 1), acc[k]])
        for i in range(M):
            U[:, k * M + i] = padded[M - 1 - i : M - 1 - i + N]
    return U

# construiendo una función de LMS para poder aplicar después

def LMS(ppg1, ppg2, accx, accy, accz, M):
    # combinando las dos señales de PPG y normalizandolas
    d = (ppg1 + ppg2) / 2
    d = (d - d.mean()) / d.std()

    #combinando las señales del acelerómetro y normalizandolas
    acc = np.vstack([accx, accy, accz])
    acc = (acc - acc.mean(axis=1, keepdims=True)) / acc.std(axis=1, keepdims=True)
    
    # algoritmo LMS

    U = build_U(acc, M)
    w = np.zeros(U.shape[1])
    e = np.zeros(len(d))
    y = np.zeros(len(d))

    # calculando el máximo mu possible
    trR = np.mean(np.sum(U**2, axis=1))
    mu_max = 2 / (3 * trR) # usando un factor de 2/3 para asegurar estabilidad, pero para afinar el numero se tiene q hacer montecarlos
    mu = mu_max 
    
    for n in range(len(d)):
        # calculando la salida del filtro
        y[n] = w @ U[n]
        
        # calculando el error
        e[n] = d[n] - y[n]
        
        # actualizando los pesos
        w = w + mu * e[n] * U[n]
    
    return e, y, w

# construiendo una función de newton LMS para poder aplicar después

def Newton_LMS(ppg1, ppg2, accx, accy, accz, M):
    # combinando las dos señales de PPG y normalizandolas
    d = (ppg1 + ppg2) / 2
    d = (d - d.mean()) / d.std()

    #combinando las señales del acelerómetro y normalizandolas
    acc = np.vstack([accx, accy, accz])
    acc = (acc - acc.mean(axis=1, keepdims=True)) / acc.std(axis=1, keepdims=True)
    
    # algoritmo LMS

    U = build_U(acc, M)
    w = np.zeros(U.shape[1])
    e = np.zeros(len(d))
    y = np.zeros(len(d))

    # calculando el máximo mu possible
    trR = np.mean(np.sum(U**2, axis=1))
    mu_max = 2 / (3 * trR) # usando un factor de 2/3 para asegurar estabilidad, pero para afinar el numero se tiene q hacer montecarlos
    mu = mu_max 

    # calculando la matriz de correlación R y su inversa
    R = np.cov(U, rowvar=False)
    R_inv = np.linalg.inv(R)
    
    for n in range(len(d)):
        # calculando la salida del filtro
        y[n] = w @ U[n]
        
        # calculando el error
        e[n] = d[n] - y[n]
        
        # actualizando los pesos (utilizando la inversa de la matriz de correlación para el método de Newton)
        w = w + mu * R_inv @ (e[n] * U[n])
    
    return e, y, w

def bpm_from_window(x, fs=125, nfft=8192, lo=0.5, hi=4.0):
    x = (x - x.mean()) * np.hanning(len(x))   # ventana de Hann: reduce el leakage
    X = np.abs(np.fft.rfft(x, nfft))           # espectro, con zero-padding
    f = np.fft.rfftfreq(nfft, 1/fs)            # eje de frecuencias en Hz
    m = (f >= lo) & (f <= hi)                  # solo la banda cardíaca
    return 60 * f[m][np.argmax(X[m])]          # frecuencia del máximo -> BPM

def bpm_series(sig):
    """BPM estimado en cada ventana de 8 s (paso 2 s). Devuelve un vector (n_ventanas,)."""
    return np.array([bpm_from_window(w) for w in split_windows(sig)])

FS, WIN, STEP = 125, 1000, 250
OVERLAP = WIN - STEP
CH_NAMES = ["ppg1", "ppg2", "accx", "accy", "accz"]


def load_ts(path):
    """Lee un .ts multivariado. Devuelve X (ventanas, canales, muestras) e y (BPM)."""
    X, y, in_data = [], [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                in_data = line.lower().startswith("@data")
                continue
            *dims, target = line.split(":")
            X.append([[np.nan if v == "?" else float(v) for v in d.split(",")] for d in dims])
            y.append(float(target))
    return np.asarray(X), np.asarray(y)


def _key(a):
    return hashlib.blake2b(np.round(a, 6).tobytes(), digest_size=16).digest()


def reconstruct_recordings(X, y):
    """Une ventanas consecutivas (comparten 750 muestras) en grabaciones continuas."""
    n = len(X)
    starts = {}
    for j in range(n):
        starts.setdefault(_key(X[j, :, :OVERLAP]), []).append(j)
    nxt, has_prev = {}, set()
    for i in range(n):
        cand = [j for j in starts.get(_key(X[i, :, STEP:]), []) if j != i]
        if len(cand) == 1 and cand[0] not in has_prev:
            nxt[i] = cand[0]
            has_prev.add(cand[0])
    recs = []
    for i0 in range(n):
        if i0 in has_prev:
            continue
        chain = [i0]
        while chain[-1] in nxt:
            chain.append(nxt[chain[-1]])
        x = np.concatenate([X[chain[0]]] + [X[j, :, OVERLAP:] for j in chain[1:]], axis=1)
        recs.append({"x": x, "bpm": y[chain], "windows": np.asarray(chain)})
    return recs


def bandpass(x, lo=0.5, hi=4.0, fs=FS, order=4, causal=True):
    """Pasabanda sobre el último eje. causal=True para simular tiempo real."""
    sos = butter(order, [lo, hi], btype="bandpass", fs=fs, output="sos")
    return (sosfilt if causal else sosfiltfilt)(sos, x, axis=-1)


def leave_one_recording_out(n_rec):
    for k in range(n_rec):
        yield [i for i in range(n_rec) if i != k], k


def save_recordings(recs, path):
    data = {}
    for k, r in enumerate(recs):
        data[f"rec{k}_x"], data[f"rec{k}_bpm"], data[f"rec{k}_windows"] = r["x"], r["bpm"], r["windows"]
    np.savez_compressed(path, n_rec=len(recs), **data)


def load_recordings(path):
    z = np.load(path)
    return [{"x": z[f"rec{k}_x"], "bpm": z[f"rec{k}_bpm"], "windows": z[f"rec{k}_windows"]}
            for k in range(int(z["n_rec"]))]

def prepare_recording(rec, primary="mean", lo=0.5, hi=4.0):
    """
    Devuelve:
      d   : señal primaria, (N,)   — PPG filtrada y normalizada
      acc : referencias, (3, N)    — acelerómetro filtrado y normalizado
      bpm : BPM de referencia por ventana, (n_ventanas,)
    """
    x = rec["x"]
    ppg = bandpass(x[:2], lo, hi, causal=True)
    acc = bandpass(x[2:], lo, hi, causal=True)
    d = ppg.mean(axis=0) if primary == "mean" else ppg[int(primary)]
    d = (d - d.mean()) / d.std()
    acc = (acc - acc.mean(axis=1, keepdims=True)) / acc.std(axis=1, keepdims=True)
    return d, acc, rec["bpm"]


def split_windows(sig, win=WIN, step=STEP):
    """Corta una señal (N,) en ventanas de 8 s cada 2 s. Devuelve (n_ventanas, 1000)."""
    n_win = (len(sig) - win) // step + 1
    return np.stack([sig[k * step : k * step + win] for k in range(n_win)])