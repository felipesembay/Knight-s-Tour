# 🐎 Cavalo Solitário (Knight's Tour)

Um jogo clássico de quebra-cabeça de xadrez onde o cavalo deve visitar todas as casas do tabuleiro exatamente uma vez, agora com inteligência artificial!

## 🎮 Como Jogar

### Início Rápido

**Opção 1 - Script Python:**
```bash
python start_game.py
```

**Opção 2 - Script Bash:**
```bash
./start_game.sh
```

**Opção 3 - Manual:**
```bash
# Terminal 1 - Servidor Flask (AI)
cd RL
python app.py

# Terminal 2 - Servidor Web
python -m http.server 8000
```

Depois acesse: **http://localhost:8000**

## 🚀 Funcionalidades

### Jogo Principal
- **Tabuleiros**: 4x4 até 9x9
- **Modos**: Tour aberto e fechado
- **Contador de movimentos** e **timer**
- **Sistema de dicas** e **desfazer**
- **Leaderboard** com armazenamento local

### Inteligência Artificial
- **Deep Q-Learning (DQN)** para tabuleiro 5x5
- **Dica de IA** em tempo real
- **Treinamento progressivo** com métricas
- **Dashboard de treinamento** com gráficos

## 🤖 Treinamento da IA

### Pré-requisitos
```bash
# Ative o ambiente virtual (se usar)
conda activate tf_env

# Instale as dependências
cd RL
pip install -r requirements.txt
```

### Executar Treinamento
```bash
cd RL
python train.py
```

### Visualizar Resultados
```bash
cd RL
python plot_results.py
```

## 📁 Estrutura do Projeto

```
Cavalo_solitario/
├── index.html          # Interface do jogo
├── script.js           # Lógica do jogo
├── styles.css          # Estilos
├── start_game.py       # Script de inicialização (Python)
├── start_game.sh       # Script de inicialização (Bash)
├── models/             # Modelos treinados (.h5)
├── logs/               # Logs de treinamento (.csv)
└── RL/                 # Código de Reinforcement Learning
    ├── train.py        # Script de treinamento
    ├── app.py          # Servidor Flask para IA
    ├── knight_env.py   # Ambiente do jogo
    ├── dqn_agent.py    # Agente DQN
    ├── plot_results.py # Visualização de resultados
    └── requirements.txt # Dependências
```

## 🎯 Como Usar

1. **Inicie o jogo** usando um dos scripts acima
2. **Escolha o tamanho** do tabuleiro (4x4 a 9x9)
3. **Selecione o modo**: aberto ou fechado
4. **Clique nas casas** para mover o cavalo
5. **Use as dicas** se precisar de ajuda
6. **Tente completar** o tour visitando todas as casas

### Recursos Especiais
- **AI Hint**: Para tabuleiro 5x5, use o botão "AI Hint" para obter sugestões da IA
- **Undo**: Desfaz o último movimento
- **Reset**: Reinicia o jogo
- **Timer**: Cronômetro para acompanhar seu tempo

## 📊 Métricas de Treinamento

O sistema registra automaticamente:
- **Score**: Número de movimentos por episódio
- **Taxa de Vitórias**: Porcentagem de tours completos
- **Epsilon**: Taxa de exploração vs exploração
- **Recompensa Média**: Performance do agente

## 🔧 Configurações Avançadas

### Parâmetros de Treinamento (RL/train.py)
```python
BOARD_SIZE = 5              # Tamanho do tabuleiro
EPISODES = 10000           # Número de episódios
BATCH_SIZE = 64            # Tamanho do batch
MAX_STEPS_PER_EPISODE = 100 # Máximo de passos por episódio
```

### Servidores
- **Web**: http://localhost:8000
- **Flask (AI)**: http://localhost:5001

## 🐛 Solução de Problemas

### Erro de Dependências
```bash
pip install flask flask-cors tensorflow numpy gym
```

### Modelo não encontrado
```bash
cd RL
python train.py  # Treine primeiro
```

### Portas ocupadas
- Mude as portas nos scripts ou pare os processos existentes

## 📈 Melhorias Futuras

- [ ] Suporte a tabuleiros maiores (10x10+)
- [ ] Múltiplos algoritmos de IA
- [ ] Interface de configuração de treinamento
- [ ] Exportação de modelos treinados
- [ ] Competições online

## 🤝 Contribuição

Sinta-se à vontade para contribuir com melhorias, correções de bugs ou novas funcionalidades!

---

**Divirta-se jogando Cavalo Solitário! 🐎♟️** 