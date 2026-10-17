import hashlib
from pprint import pp
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, sosfiltfilt
import time, os


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

# Funcion para simular una señal de ppg - es para utilizar después en la simulación de montecarlo

def ppg_simulada(seed, t, fs):
    # simulando la señal PPG
    rng = np.random.default_rng(seed) 

    fh = rng.uniform(1, 3)  # el rango lo escogimos porq 60 bpm y 177 bpm es el rango de la base de datos IEEEPPG 
    a = [1, 0.4, 0.2] # amplitudes de los armónicos
    ppg_simulada = 0 
    k = 1 # para los harmonicos

    for i in a:  
        phase = rng.uniform(0, 2 * np.pi)  # fase aleatoria
        ppg_simulada += i * np.cos(2 * np.pi * k * fh/fs * t + phase)
        k += 1

    # normalizando la señal PPG simulada para q tenga potencia 1
    ppg_simulada = ppg_simulada / np.sqrt(np.mean(ppg_simulada**2))

    return ppg_simulada, fh

# Funcion para simular una señal de acc - es para utilizar después en la simulación de montecarlo

def acc_simulada(seed, t, fs):
    #definiendo variables
    rng = np.random.default_rng(seed)
    f_c = rng.uniform(1.5, 3)  # cadencia: una sola para los tres ejes, es el mismo movimiento
    k = np.arange(1, 6)  # armónicos de la cadencia: f_c, 2*f_c, ..., 5*f_c
    acc = []

    #simulando la acceleracion para cada eje
    for eje in range(3):
        a = rng.standard_normal(size=5) / k  # amplitudes que decaen con el armónico
        i = 0 

        for count,j in enumerate(a):
            phase = rng.uniform(0, 2 * np.pi)  # fase aleatoria
            i += j * np.cos(2 * np.pi * k[count] * f_c/fs * t + phase)

        white_noise = 0.2 * rng.standard_normal(size=len(t))  # ruido blanco chico
        i += white_noise

        i = (i - i.mean()) / i.std()  # media 0 y desvío 1, como en los datos reales
        acc.append(i)

    return np.vstack(acc), f_c

# Es para inttegrar la señal de acc en la señal de ppg

def camino_artefacto(seed, M_true, tau=None):
    # camino por el que el movimiento entra en la PPG: un FIR desconocido por eje
    rng = np.random.default_rng(seed)
    if tau is None:
        tau = M_true / 3  # constante de decaimiento
    h = rng.standard_normal(size=(3, M_true)) * np.exp(-np.arange(M_true) / tau)

    return h  # forma (3, M_true)

def lms_core(d, U, mu, R_inv=None):
    Nn, L = U.shape 
    w = np.zeros(L); e = np.zeros(Nn)
    for n in range(Nn):
        e[n] = d[n] - w @ U[n]
        w = w + mu * e[n] * U[n]
        if not np.isfinite(e[n]) or abs(e[n]) > DIV_THR:   # divergió: cortamos
            e[n:] = np.nan
            return e, True
    return e, False

def newton_core(d, U, mu, R_inv):
    Nn, L = U.shape
    w = np.zeros(L); e = np.zeros(Nn)
    for n in range(Nn):
        e[n] = d[n] - w @ U[n]
        w = w + mu * (R_inv @ (e[n] * U[n]))
        if not np.isfinite(e[n]) or abs(e[n]) > DIV_THR:
            e[n:] = np.nan
            return e, True
    return e, False

ALGOS = {"LMS": lms_core, "Newton-LMS": newton_core}


DIV_THR = 1e3     # criterio de divergencia: |e| > 1e3 o no finito

os.makedirs("figuras_mc", exist_ok=True)
def guardar(nombre):
    plt.savefig(f"figuras_mc/{nombre}.pdf", bbox_inches="tight")
    plt.savefig(f"figuras_mc/{nombre}.png", dpi=200, bbox_inches="tight")

def escenario(seed, N, M_true, rho, sigma_v):
    rng = np.random.default_rng(seed)      # un solo generador para todo el escenario
    t = np.arange(N)
    s, fh = ppg_simulada(rng, t, FS)       # latido (potencia 1)
    acc, f_c = acc_simulada(rng, t, FS)    # acelerómetro (3, N)
    h = camino_artefacto(rng, M_true)      # camino desconocido (3, M_true)
    m = build_U(acc, M_true) @ h.reshape(-1)
    esc = np.sqrt(rho * np.mean(s**2) / np.mean(m**2))   # escala: potencia de m = rho * potencia de s
    h, m = h * esc, m * esc
    v = sigma_v * rng.standard_normal(N)
    d = s + m + v
    return dict(d=d, acc=acc, s=s, m=m, h=h, fh=fh, f_c=f_c)

def preparar(esc, M):
    U = build_U(esc["acc"], M)
    L = U.shape[1]
    trR = np.mean(np.sum(U**2, axis=1))                  # estimación de tr(R)
    R = U.T @ U / len(U)
    R_inv = np.linalg.inv(R + 1e-3 * np.trace(R) / L * np.eye(L))   # con carga diagonal
    return dict(U=U, trR=trR, R_inv=R_inv)
