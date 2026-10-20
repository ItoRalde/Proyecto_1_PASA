import numpy as np

# toma todas las señales de accelerometro y las combina, añadiendo zeros para q sea pueda utilizar después

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
    
    return e, y, 

#gemini implemetation without harcoded hyperparameters:
'''
import numpy as np


def build_U(acc: np.ndarray, M: int) -> np.ndarray:
    """Construye la matriz de regresores espacio-temporales U mediante retardos."""
    n_ref, N = acc.shape
    U = np.zeros((N, n_ref * M))
    for k in range(n_ref):
        padded = np.concatenate([np.zeros(M - 1), acc[k]])
        for i in range(M):
            U[:, k * M + i] = padded[M - 1 - i : M - 1 - i + N]
    return U


def LMS(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    M: int,
    mu_scale: float = 0.5,
    mu_fixed: float | None = None,
    ppg_mode: str = "mean",
    include_magnitude: bool = True,
    center_only: bool = False,
    normalized: bool = False,
    leakage: float = 0.0,
    eps_nlms: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Algoritmo LMS completamente parametrizado para optimización de hiperparámetros.

    Parámetros
    ----------
    ppg1, ppg2 : np.ndarray
        Señales fotopletismográficas primarias.
    accx, accy, accz : np.ndarray
        Señales triaxiales del acelerómetro.
    M : int
        Número de coeficientes (taps) por canal de ruido.
    mu_scale : float, default=0.5
        Factor multiplicativo sobre mu_max = 2 / (3 * tr(R)).
    mu_fixed : float, opcional
        Valor escalar explícito para mu. Si se define, anula a mu_scale.
    ppg_mode : {'mean', 'ppg1', 'ppg2', 'pca'}, default='mean'
        Método de fusión de las dos señales PPG.
    include_magnitude : bool, default=True
        Si es True, concatena la norma euclídea del acelerómetro como canal adicional.
    center_only : bool, default=False
        Si es True, sólo centra las señales a media cero sin dividir por su desviación estándar.
    normalized : bool, default=False
        Si es True, activa el paso normalizado instantáneo (NLMS clásico).
    leakage : float, default=0.0
        Factor de fuga gamma para Leaky-LMS (0.0 equivale al LMS clásico).
    eps_nlms : float, default=1e-6
        Término de regularización para evitar división por cero en NLMS.

    Retorna
    -------
    e : np.ndarray
        Señal de error (PPG filtrada y estimada sin artefactos de movimiento).
    y : np.ndarray
        Señal estimada del artefacto de movimiento.
    w : np.ndarray
        Vector final de pesos del filtro FIR adaptativo.
    """
    # 1. Selección o fusión de los canales PPG
    if ppg_mode == "mean":
        d = 0.5 * (ppg1 + ppg2)
    elif ppg_mode == "ppg1":
        d = ppg1.copy()
    elif ppg_mode == "ppg2":
        d = ppg2.copy()
    elif ppg_mode == "pca":
        P = np.vstack([ppg1, ppg2])
        P_centered = P - np.mean(P, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(P_centered, full_matrices=False)
        d = Vt[0]
    else:
        raise ValueError(f"ppg_mode no reconocido: {ppg_mode}")

    # Normalización del PPG
    d_mean = np.mean(d)
    d_scale = 1.0 if center_only else (np.std(d) + 1e-8)
    d = (d - d_mean) / d_scale

    # 2. Ensamblado y acondicionamiento de las señales de aceleración
    acc_channels = [accx, accy, accz]
    if include_magnitude:
        acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
        acc_channels.append(acc_mag)

    acc = np.vstack(acc_channels)
    acc_mean = np.mean(acc, axis=1, keepdims=True)
    acc_scale = 1.0 if center_only else (np.std(acc, axis=1, keepdims=True) + 1e-8)
    acc = (acc - acc_mean) / acc_scale

    # 3. Construcción de la matriz de regresores
    U = build_U(acc, M)
    N, P_dim = U.shape

    # 4. Determinación del paso de aprendizaje mu
    if mu_fixed is not None:
        mu = mu_fixed
    else:
        trR = np.mean(np.sum(U**2, axis=1))
        mu_max = 2.0 / (3.0 * trR) if trR > 0 else 1e-3
        mu = mu_scale * mu_maxs

    w = np.zeros(P_dim)
    e = np.zeros(N)
    y = np.zeros(N)
    leak_factor = 1.0 - leakage

    # 5. Iteración adaptativa
    for n in range(N):
        u_n = U[n]
        y[n] = w @ u_n
        e[n] = d[n] - y[n]

        step = mu / (u_n @ u_n + eps_nlms) if normalized else mu
        w = leak_factor * w + step * e[n] * u_n

    return e, y, w




def Newton_LMS(
    ppg1: np.ndarray,
    ppg2: np.ndarray,
    accx: np.ndarray,
    accy: np.ndarray,
    accz: np.ndarray,
    M: int,
    mu_scale: float = 0.5,
    reg_factor: float = 1e-4,
    ppg_mode: str = "mean",
    include_magnitude: bool = True,
    center_only: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Newton-LMS adaptive filter with parameterized degrees of freedom for hyperparameter tuning.

    Parameters
    ----------
    ppg1, ppg2 : np.ndarray
        Raw photoplethysmogram channel signals.
    accx, accy, accz : np.ndarray
        Tri-axial acceleration reference signals.
    M : int
        Filter order (taps per reference channel).
    mu_scale : float, default=0.5
        Step-size scaling factor in range (0, 1].
    reg_factor : float, default=1e-4
        Diagonal loading coefficient epsilon relative to tr(R) for matrix inversion.
    ppg_mode : {'mean', 'ppg1', 'ppg2', 'pca'}, default='mean'
        Fusion strategy for the primary PPG channels.
    include_magnitude : bool, default=True
        Whether to append the acceleration norm as a fourth reference signal.
    center_only : bool, default=False
        If True, signals are zero-centered without unit-variance scaling.
    """
    # 1. Primary PPG Signal Selection / Fusion
    if ppg_mode == "mean":
        d = 0.5 * (ppg1 + ppg2)
    elif ppg_mode == "ppg1":
        d = ppg1.copy()
    elif ppg_mode == "ppg2":
        d = ppg2.copy()
    elif ppg_mode == "pca":
        P = np.vstack([ppg1, ppg2])
        P_centered = P - np.mean(P, axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(P_centered, full_matrices=False)
        d = Vt[0]
    else:
        raise ValueError(f"Unknown ppg_mode: {ppg_mode}")

    # PPG Normalization
    d_mean = np.mean(d)
    d_scale = 1.0 if center_only else (np.std(d) + 1e-8)
    d = (d - d_mean) / d_scale

    # 2. Reference Signal Assembly
    acc_channels = [accx, accy, accz]
    if include_magnitude:
        acc_mag = np.sqrt(accx**2 + accy**2 + accz**2)
        acc_channels.append(acc_mag)

    acc = np.vstack(acc_channels)
    acc_mean = np.mean(acc, axis=1, keepdims=True)
    acc_scale = 1.0 if center_only else (np.std(acc, axis=1, keepdims=True) + 1e-8)
    acc = (acc - acc_mean) / acc_scale

    # 3. Regressor Matrix Construction
    U = build_U(acc, M)
    N, P_dim = U.shape

    # 4. Correlation Matrix and Regularized Inversion
    R = (U.T @ U) / N
    trR = np.trace(R)
    epsilon = reg_factor * (trR / P_dim)
    R_inv = np.linalg.inv(R + epsilon * np.eye(P_dim))

    # 5. Step-Size Calculation
    mu = mu_scale * (2.0 / (3.0 * trR))

    # 6. Adaptive Filtering Loop
    w = np.zeros(P_dim)
    e = np.zeros(N)
    y = np.zeros(N)

    for n in range(N):
        u_n = U[n]
        y[n] = w @ u_n
        e[n] = d[n] - y[n]
        w += mu * (R_inv @ (e[n] * u_n))

    return e, y, w

'''