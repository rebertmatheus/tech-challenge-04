import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import argparse


def load_dataframe_from_file(default_path=None):
	"""Tenta carregar DataFrame a partir de um CSV. Se falhar, retorna None."""
	if default_path and os.path.exists(default_path):
		try:
			return pd.read_csv(default_path)
		except Exception:
			return None
	return None


parser = argparse.ArgumentParser(description='Plot price series from CSV file.')
parser.add_argument('--file', '-f', help='Caminho para o arquivo CSV com colunas timestamp,close')
parser.add_argument('--output', '-o', help='Caminho de saída do PNG (ex: graficos/output/petr4/price_series.png)')
args = parser.parse_args()

# Caminho padrão relativo ao workspace (graficos/output/price_series_bbas3.csv)
script_dir = os.path.dirname(__file__)
default_file = os.path.normpath(os.path.join(script_dir, '..', 'graficos', 'output', 'price_series_bbas3.csv'))

# Tenta carregar do arquivo indicado pelo usuário, depois do padrão; caso contrário usa a string embutida
df = None
if args.file:
	df = load_dataframe_from_file(args.file)
if df is None:
	df = load_dataframe_from_file(default_file)

# Se não foi possível carregar do arquivo, usa dados embutidos
if df is None:
	# Carregando os dados fornecidos (fallback)
	data = """timestamp,close
2025-12-28 17:56:28.286402-03:00,23.469999313354492
2025-12-28 19:04:39.599211-03:00,21.440000534057617
2025-12-28 19:17:29.793173-03:00,22.860000610351562
2025-12-29 19:00:40.460785-03:00,21.40999984741211
2026-01-02 23:59:57.282716-03:00,22.559999465942383
2026-01-04 20:02:58.590617-03:00,23.469999313354492
2026-01-04 21:24:16.974287-03:00,20.3700008392334
2026-01-05 19:00:16.204620-03:00,20.3700008392334
2026-01-06 19:00:15.322302-03:00,21.8799991607666"""

	# Criar DataFrame a partir da string
	df = pd.read_csv(io.StringIO(data))

# Converter timestamp para objeto datetime e ordenar
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# Configuração do Gráfico
plt.figure(figsize=(12, 6))
plt.plot(df['timestamp'], df['close'], marker='o', linestyle='-', color='#2c3e50', linewidth=2)

# Estilização
plt.title('Variação de Preço (Dez/2025 - Jan/2026)', fontsize=14)
plt.xlabel('Data e Hora', fontsize=12)
plt.ylabel('Preço de Fechamento', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()

# Exibir gráfico
# Determinar caminho de saída
output_file = None
if args.output:
	output_file = args.output
else:
	output_file = os.path.normpath(os.path.join(script_dir, '..', 'graficos', 'output', 'price_series.png'))

# Garantir diretório existe
output_dir = os.path.dirname(output_file)
if output_dir:
	os.makedirs(output_dir, exist_ok=True)

# Salvar PNG
plt.savefig(output_file, dpi=150)
print(f"Saved plot to: {output_file}")

# Também mostra a figura na tela
plt.show()