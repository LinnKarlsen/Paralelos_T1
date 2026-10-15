import subprocess
import pandas as pd

### algoritmo serial
n_list = [2048, 1024, 512, 256, 128, 64, 32]
resultados = []

for n in n_list:    

    # ejecutar 
    resultado_proc = subprocess.run(
        ["./serial", "2048", str(n)], 
        capture_output=True, 
        text=True, 
        check=True
    )
    
    # sacar el resultado
    tiempo_medido = float(resultado_proc.stdout.strip())
    
    # guardar
    resultados.append({
        "algoritmo": "algoritmo1",
        "n": str(n),
        "tiempo": tiempo_medido
    })

# data frame + csv de los resultados guardados
df = pd.DataFrame(resultados)
df.to_csv("resultados.csv", index=False)

print("csv listo!!")