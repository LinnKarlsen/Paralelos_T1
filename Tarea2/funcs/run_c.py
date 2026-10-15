import subprocess

def run_c(binary, N, n, p):

    ### correr codigo de c (PREVIAMENTE COMPILADO)
    resultado_proc = subprocess.run(
        [f"./{binary}", str(N), str(n), str(p)], 
        capture_output=True, 
        text=True, 
        check=True
    )

    ### capturar el codigo de c
    tiempo_medido = float(resultado_proc.stdout.strip())

    ### devolver en formato dict para pasar directo al data frame
    return {
        "algoritmo": binary,
        "N": N,
        "n": n,
        "threads": p,
        "tiempo": tiempo_medido
        }