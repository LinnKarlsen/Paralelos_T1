import pandas as pd
import sys
from funcs.plot_funcs import plot_time, plot_speedup, plot_eficiencia, plot_vs_n, plot_eficiencia_debil_simple

name = sys.argv[1] # vicho o linn

### csv de resultados

path = f"resultados_{name}/resultados.csv"
df = pd.read_csv(path)

Ta1 = list(df[df["algoritmo"] == "algoritmo1"]["tiempo"])
Ta2 = list(df[df["algoritmo"] == "algoritmo2"]["tiempo"])
Ta3 = list(df[df["algoritmo"] == "algoritmo3.py"]["tiempo"])

tm_fig = plot_time(Ta1, Ta2, Ta3)
tm_fig.savefig(f'graficos_{name}/tm_fig.png', dpi=300)

sp_fig = plot_speedup(Ta1, Ta2, Ta3)
sp_fig.savefig(f'graficos_{name}/sp_fig.png', dpi=300)

ef_fig = plot_eficiencia(Ta1, Ta2, Ta3)
ef_fig.savefig(f'graficos_{name}/ef_fig.png', dpi=300)

### csv de serial time

path = f"resultados_{name}/serial_time_vs_n.csv"
df = pd.read_csv(path)
T = list(df["tiempo"])
serial_fig = plot_vs_n(T)
serial_fig.savefig(f'graficos_{name}/serial_fig.png', dpi=300)

### csv de conquer time

path = f"resultados_{name}/conquer_time_vs_n.csv"
df = pd.read_csv(path)
T = list(df["tiempo"])
serial_fig = plot_vs_n(T)
serial_fig.savefig(f'graficos_{name}/conquer_fig.png', dpi=300)

path = f"resultados_{name}/resultados_escalabilidad_debil.csv"
df = pd.read_csv(path)

# Lista con los nombres de los algoritmos evaluados
algoritmos = ["algoritmo1", "algoritmo2", "algoritmo3.py"]

# Iterar sobre cada algoritmo para generar su respectivo gráfico de eficiencia débil
for algo in algoritmos:
    # Filtrar el DataFrame para el algoritmo actual
    df_algo = df[df["algoritmo"] == algo]
    
    # Extraer las listas de tiempos y workers correspondientes a los pares proporcionales (N, p)
    T_weak = list(df_algo["tiempo"])
    workers = list(df_algo["threads"]) # Asegúrate de que la columna se llame 'threads' o 'p' según tu CSV
    
    # Generar la figura usando la función de eficiencia débil
    ef_debil_fig = plot_eficiencia_debil_simple(T_weak, workers)
    
    # Guardar el gráfico en alta resolución
    nombre_archivo = algo.replace(".py", "") # Reemplaza el punto en 'algoritmo3.py' para evitar problemas con extensiones
    ef_debil_fig.savefig(f'graficos_{name}/ef_debil_{nombre_archivo}.png', dpi=300)