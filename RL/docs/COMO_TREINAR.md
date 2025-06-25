# Como Executar o Treinamento - Knight's Tour DQN

Este guia mostra como executar o treinamento do modelo DQN de forma automática, mantendo a configuração antiga de plotagem no terminal.

## 🚀 Opções de Execução

### Opção 1: Script Bash (Mais Simples)
```bash
cd RL
./run_training.sh
```

### Opção 2: Script Python (Mais Flexível)
```bash
cd RL
python3 auto_train.py
```

### Opção 3: Execução Direta
```bash
cd RL
python3 train.py
```

## 📊 O que você verá no terminal

Durante o treinamento, você verá:

```
Training Progress:   6%|███▏                                                  | 599/10000 [03:53<1:13:55,  2.12it/s]
[Resumo dos Episódios 500 a 600]
Melhor Episódio: 594 | Score: 28 | Max Visited: 29/36 | Win: 0 | Epsilon: 0.0514
Vitórias no grupo: 0/100 | Média de casas visitadas: 19.7
Modelo salvo em: models/6x6/knight_tour_dqn_b6_e600.h5
```

## ⚙️ Configurações

### Configurações Padrão (6x6)
- **Tabuleiro**: 6x6
- **Episódios**: 10.000
- **Batch Size**: 64
- **Target Update**: A cada 10 episódios
- **Max Steps**: 1.000 por episódio

### Personalizar Configurações

#### Via Script Python:
```bash
# Treinar por 5000 episódios
python3 auto_train.py --episodes 5000

# Pular verificações de dependências (mais rápido)
python3 auto_train.py --skip-checks

# Ver todas as opções
python3 auto_train.py --help
```

#### Via Edição do Código:
Edite o arquivo `train.py` e modifique as variáveis no início:
```python
BOARD_SIZE = 6          # Tamanho do tabuleiro
EPISODES = 10000        # Número de episódios
BATCH_SIZE = 64         # Tamanho do batch
TARGET_UPDATE_FREQ = 10 # Frequência de atualização da target network
MAX_STEPS_PER_EPISODE = 1000 # Máximo de passos por episódio
```

## 📁 Arquivos Gerados

### Durante o Treinamento:
- `logs/training_log.csv` - Log detalhado de cada episódio
- `models/6x6/knight_tour_dqn_b6_e100.h5` - Modelos salvos a cada 100 episódios
- `models/6x6/knight_tour_dqn_b6_e200.h5`
- `models/6x6/knight_tour_dqn_b6_e300.h5`
- ...

### Ao Final:
- `models/6x6/knight_tour_dqn_b6_final.h5` - Modelo final

## 🛠️ Solução de Problemas

### Erro de Dependências:
```bash
pip3 install -r requirements.txt
```

### Erro de GPU:
O treinamento funciona na CPU, mas será mais lento. Para usar GPU:
```bash
pip3 install tensorflow-gpu
```

### Interromper Treinamento:
Pressione `Ctrl+C` a qualquer momento para parar o treinamento.

### Verificar Progresso:
```bash
# Monitorar em tempo real
python3 monitor_training.py monitor 6

# Gerar gráfico
python3 monitor_training.py plot 6
```

## 📈 Interpretando os Resultados

### Métricas Importantes:
- **Score**: Número de passos no episódio
- **Max Visited**: Máximo de casas visitadas (objetivo: 36 para 6x6)
- **Win**: 1 se completou o tour, 0 caso contrário
- **Epsilon**: Taxa de exploração (diminui com o tempo)
- **Vitórias no grupo**: Quantas vitórias nos últimos 100 episódios

### Critérios de Sucesso:
- **Bom**: Máximo visitado > 30 casas
- **Muito Bom**: Máximo visitado > 33 casas  
- **Excelente**: Vitórias completas (36 casas)

## ⏱️ Tempo Estimado

- **CPU**: ~2-4 horas para 10.000 episódios
- **GPU**: ~30-60 minutos para 10.000 episódios

## 🔄 Retomar Treinamento

Para continuar de onde parou, edite o `train.py` e adicione:
```python
# Carregar modelo existente
agent.load("models/6x6/knight_tour_dqn_b6_e600.h5")
```

## 📞 Suporte

Se encontrar problemas:
1. Verifique se está no diretório `RL/`
2. Confirme que todas as dependências estão instaladas
3. Verifique se há espaço suficiente em disco
4. Consulte os logs em `logs/training_log.csv` 