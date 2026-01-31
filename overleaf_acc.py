import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

# --- CONFIGURAÇÕES DE ESTILO (Acadêmico) ---
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('seaborn-whitegrid')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 13,
    'figure.figsize': (10, 6),
    'lines.linewidth': 2.5
})

# --- PALETA DE CORES (Padronizada) ---
COLORS = {
    "FedAvg": "#7f7f7f",   # Cinza (Baseline)
    "FedProx": "#1f77b4",  # Azul (Estrutural)
    "MOON": "#2ca02c",     # Verde (Semântico - Seu novo resultado!)
    "ASTRA": "#9467bd"     # Roxo (Ours - Destaque)
}

# --- PARÂMETROS DO ORIENTADOR ---
WINDOW_SIZE = 5  # Janela de suavização (Média Móvel)

def load_data():
    all_data = []
    
    # LISTA DE ARQUIVOS
    # Verifique se os nomes na sua pasta batem exatamente com estes
    files = [
        # --- ASTRA (Scenario 4 - Hybrid) ---
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed10_new.csv"),
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed42_new.csv"),      # Seed 42 (Geralmente sem sufixo)
        ("ASTRA", "results_metrics/metrics_Scenario4_Hybrid_Seed999_new.csv"),

        # --- MOON (Arquivos que você acabou de gerar) ---
        ("MOON", "results_metrics/metrics_moon_seed10.csv"),
        ("MOON", "results_metrics/metrics_moon_seed42.csv"),
        ("MOON", "results_metrics/metrics_moon_seed999.csv"),

        # --- FedProx ---
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx_Seed10.csv"),
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx.csv"),   # Seed 42
        ("FedProx", "results_metrics/metrics_Scenario2_FedProx_Seed999.csv"),

        # --- FedAvg ---
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg_Seed10.csv"),
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg.csv"),     # Seed 42
        ("FedAvg", "results_metrics/metrics_Scenario1_FedAvg_Seed999.csv"),
    ]

    print("--- Carregando Datasets ---")
    for method, filepath in files:
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Normaliza nomes de colunas (caso mude entre csvs)
                df.columns = [c.lower() for c in df.columns] 
                
                if 'round' in df.columns and 'accuracy' in df.columns:
                    # Seleciona apenas o necessário
                    temp_df = df[['round', 'accuracy']].copy()
                    temp_df['method'] = method
                    all_data.append(temp_df)
                    print(f"✅ Carregado: {filepath} ({len(df)} rounds)")
                else:
                    print(f"⚠️  Colunas erradas em: {filepath}")
            except Exception as e:
                print(f"❌ Erro ao ler {filepath}: {e}")
        else:
            print(f"❌ Arquivo ausente: {filepath}") # Não pare o script, apenas avise

    if not all_data:
        print("⛔ Nenhum dado encontrado. Verifique os caminhos na lista 'files'.")
        return pd.DataFrame() # Retorna vazio

    return pd.concat(all_data, ignore_index=True)

def plot_comparison(df):
    if df.empty:
        return

    plt.figure()
    
    # Plota na ordem de COLORS para consistência
    for method in COLORS.keys():
        subset = df[df['method'] == method]
        if subset.empty:
            continue

        # 1. AGRUPA AS SEEDS (Calcula a média "bruta" e desvio entre as 3 execuções)
        # Isso funde as seeds 10, 42 e 999 em uma linha só por método
        grouped = subset.groupby('round')['accuracy'].agg(['mean', 'std']).reset_index()

        # 2. APLICA A SUAVIZAÇÃO (Solicitação do Orientador)
        # Rolling Mean: Média dos últimos 5 rounds
        grouped['smooth_mean'] = grouped['mean'].rolling(window=WINDOW_SIZE, min_periods=1).mean()
        grouped['smooth_std'] = grouped['std'].rolling(window=WINDOW_SIZE, min_periods=1).mean()

        # 3. PLOTAGEM
        # Linha Principal (Média Suavizada)
        plt.plot(grouped['round'], grouped['smooth_mean'], 
                 label=method, color=COLORS[method], alpha=1.0)
        
        # Sombra de Erro (Desvio Padrão Suavizado)
        # Preenche a área entre (Média - Std) e (Média + Std)
        plt.fill_between(grouped['round'], 
                         grouped['smooth_mean'] - grouped['smooth_std'], 
                         grouped['smooth_mean'] + grouped['smooth_std'], 
                         color=COLORS[method], alpha=0.15)

    # Decorações do Gráfico
    plt.xlabel('Communication Rounds')
    plt.ylabel('Test Accuracy (%)')
    plt.title(f'Non-IID Convergence (Smoothed Window={WINDOW_SIZE})')
    plt.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 50) # Garante que mostre até o round 50
    plt.tight_layout()
    
    # Salvar
    filename = 'fig_accuracy_comparison.pdf'
    plt.savefig(filename, format='pdf', dpi=300)
    plt.savefig('fig_accuracy_comparison.png', format='png', dpi=300) # Backup PNG
    print(f"\n✨ Sucesso! Gráfico salvo como: {filename}")

if __name__ == "__main__":
    df_combined = load_data()
    plot_comparison(df_combined)