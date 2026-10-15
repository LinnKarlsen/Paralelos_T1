import subprocess

def run_serial(binary, N, n):

    ### correr codigo de c (PREVIAMENTE COMPILADO)
    resultado_proc = subprocess.run(
        [f"./{binary}", str(N), str(n)], 
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
        "tiempo": tiempo_medido
        }