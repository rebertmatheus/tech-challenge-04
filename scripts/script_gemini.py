import pandas as pd
import matplotlib.pyplot as plt
import io
import os
import argparse
import glob


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
parser.add_argument('--ticker', '-t', help='Nome do ticker (ex: bbas3, petr4, vale3)')
parser.add_argument('--all', action='store_true', help='Processar todos os arquivos price_series_*.csv no diretório graficos/output')
parser.add_argument('--output', '-o', help='Caminho de saída do PNG (ex: graficos/output/price_series.png)')
parser.add_argument('--no-show', action='store_true', help='Não chamar plt.show() (útil para execução em CI)')
args = parser.parse_args()

# Caminho padrão relativo ao workspace (graficos/output)
script_dir = os.path.dirname(__file__)
output_dir_default = os.path.normpath(os.path.join(script_dir, '..', 'graficos', 'output'))


def process_file(input_file, output_file=None, show=True):
	"""Carrega CSV, plota e salva PNG. Se input_file não existir, tenta fallback embutido."""
	df = load_dataframe_from_file(input_file)
	if df is None:
		# fallback para os mesmos dados embutidos
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
		df = pd.read_csv(io.StringIO(data))

	df['timestamp'] = pd.to_datetime(df['timestamp'])
	df = df.sort_values('timestamp')

	plt.figure(figsize=(12, 6))
	plt.plot(df['timestamp'], df['close'], marker='o', linestyle='-', color='#2c3e50', linewidth=2)
	plt.title('Variação de Preço (Dez/2025 - Jan/2026)', fontsize=14)
	plt.xlabel('Data e Hora', fontsize=12)
	plt.ylabel('Preço de Fechamento', fontsize=12)
	plt.grid(True, linestyle='--', alpha=0.7)
	plt.xticks(rotation=45)
	plt.tight_layout()

	if not output_file:
		# derive default name from input_file
		base = os.path.splitext(os.path.basename(input_file))[0]
		output_file = os.path.join(output_dir_default, base.replace('price_series', 'price_series') + '.png')

	os.makedirs(os.path.dirname(output_file), exist_ok=True)
	plt.savefig(output_file, dpi=150)
	print(f"Saved plot to: {output_file}")
	if show:
		plt.show()
	plt.close()


def find_default_file_for_ticker(ticker):
	if not ticker:
		return None
	fname = f'price_series_{ticker.lower()}.csv'
	return os.path.join(output_dir_default, fname)


# Main logic: support --all, --ticker or explicit --file
if args.all:
	pattern = os.path.join(output_dir_default, 'price_series_*.csv')
	files = glob.glob(pattern)
	if not files:
		print('No files found for pattern:', pattern)
	for f in files:
		base = os.path.splitext(os.path.basename(f))[0]
		out = os.path.join(output_dir_default, base + '.png')
		process_file(f, out, show=not args.no_show)
else:
	input_file = None
	if args.file:
		input_file = args.file
	elif args.ticker:
		input_file = find_default_file_for_ticker(args.ticker)
	else:
		# default to bbas3 if nothing provided
		input_file = os.path.join(output_dir_default, 'price_series_bbas3.csv')

	output_file = args.output
	process_file(input_file, output_file, show=not args.no_show)