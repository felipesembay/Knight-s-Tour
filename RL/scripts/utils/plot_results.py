import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_training_results(log_file='./logs/training_log.csv', output_dir='./logs/'):
    """
    Lê o arquivo de log do treinamento e gera gráficos de desempenho.
    """
    if not os.path.exists(log_file):
        print(f"Arquivo de log não encontrado: {log_file}")
        return

    # Carrega os dados
    df = pd.read_csv(log_file)
    
    if df.empty:
        print("Arquivo de log está vazio. Nenhum gráfico gerado.")
        return

    # Configura o estilo dos gráficos
    sns.set_theme(style="darkgrid")
    
    # Cria a figura para os gráficos
    fig, axes = plt.subplots(3, 1, figsize=(12, 18))
    fig.suptitle('Resultados do Treinamento do Agente DQN', fontsize=16)

    # 1. Gráfico de Score por Episódio com Média Móvel
    axes[0].plot(df['Episode'], df['Score'], alpha=0.3, label='Score por Episódio')
    # Calcula a média móvel para suavizar a curva
    score_moving_avg = df['Score'].rolling(window=100, min_periods=1).mean()
    axes[0].plot(df['Episode'], score_moving_avg, color='red', linewidth=2, label='Média Móvel (100 ep.)')
    axes[0].set_xlabel('Episódio')
    axes[0].set_ylabel('Score (Movimentos)')
    axes[0].set_title('Score Alcançado por Episódio')
    axes[0].legend()

    # 2. Gráfico da Taxa de Vitórias
    df['WinRate'] = df['Win'].rolling(window=100, min_periods=1).mean() * 100
    axes[1].plot(df['Episode'], df['WinRate'], color='green', linewidth=2, label='Taxa de Vitórias (%)')
    axes[1].set_xlabel('Episódio')
    axes[1].set_ylabel('Taxa de Vitórias (%)')
    axes[1].set_title('Taxa de Vitórias (Média Móvel de 100 Episódios)')
    axes[1].legend()

    # 3. Gráfico da Decaimento do Epsilon
    axes[2].plot(df['Episode'], df['Epsilon'], color='purple', linewidth=2, label='Epsilon')
    axes[2].set_xlabel('Episódio')
    axes[2].set_ylabel('Valor do Epsilon')
    axes[2].set_title('Decaimento do Epsilon (Taxa de Exploração)')
    axes[2].legend()

    # Salva o gráfico em um arquivo
    output_path = os.path.join(output_dir, 'training_dashboard.png')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path)
    print(f"Gráfico do dashboard salvo em: {output_path}")
    plt.close()

if __name__ == "__main__":
    plot_training_results() 