import pandas as pd
import sys
from funcs.plot_funcs import plot_time, plot_speedup, plot_eficiencia, plot_vs_n

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