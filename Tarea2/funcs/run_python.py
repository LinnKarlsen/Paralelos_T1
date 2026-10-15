import subprocess
import pandas as pd

def run_python(binary, N, n, p):
    """
    Ejecuta el algoritmo en python y devuelve un diccionario con los resultados.
    """
    # ejecutar 
    resultado_proc = subprocess.run(
        ["python3", binary, str(N), str(n), str(p)], 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # sacar el resultado
    output = resultado_proc.stdout.strip().splitlines()
    tiempo_medido = float(output[-1])  # sacamos sólo el último 
    
    # devolver diccionario
    return {
        "algoritmo": binary,
        "N": str(N),
        "threads": str(p),
        "tiempo": tiempo_medido
    }