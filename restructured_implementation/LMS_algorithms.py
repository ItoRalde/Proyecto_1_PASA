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
    
    return e, y, w