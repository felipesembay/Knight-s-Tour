import pandas as pd
import os

def update_csv_with_win_column():
    """
    Adiciona a coluna 'Win' ao arquivo CSV existente baseado no Max_Visited
    """
    csv_path = '../logs/training_log.csv'
    
    if not os.path.exists(csv_path):
        print(f"Arquivo não encontrado: {csv_path}")
        return
    
    # Lê o arquivo CSV sem cabeçalho
    df = pd.read_csv(csv_path, header=None)
    
    # Define os nomes das colunas baseado na estrutura atual
    df.columns = ['Episode', 'Score', 'Max_Visited', 'Avg_Reward', 'Invalid_Moves', 'Epsilon']
    
    # Adiciona a coluna Win (1 se Max_Visited == 25 para tabuleiro 5x5, 0 caso contrário)
    df['Win'] = (df['Max_Visited'] == 25).astype(int)
    
    # Salva o arquivo atualizado
    df.to_csv(csv_path, index=False)
    print(f"Arquivo atualizado com sucesso: {csv_path}")
    print(f"Total de vitórias encontradas: {df['Win'].sum()}")
    print(f"Taxa de vitórias: {(df['Win'].sum() / len(df) * 100):.2f}%")

if __name__ == "__main__":
    update_csv_with_win_column() 