# 🐴 Knight's Tour - IA com Deep Q-Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **Sistema de IA para resolver o problema do Cavalo Solitário usando Deep Q-Learning (DQN)**

## 📋 Visão Geral

Este projeto implementa uma solução completa para o problema do Knight's Tour (Cavalo Solitário) usando Deep Q-Learning. O sistema inclui:

- 🧠 **Modelo DQN treinado** para múltiplos tamanhos de tabuleiro (5x5, 6x6, 7x7, 8x8)
- 🌐 **Interface web interativa** para jogar contra a IA
- 📊 **Sistema de monitoramento** em tempo real
- 🛠️ **Ferramentas de gerenciamento** de modelos
- 📈 **Análise e validação** científica dos resultados

## 🏆 Resultados Comprovados

### ✅ Tabuleiro 5x5 - VALIDADO
- **Taxa de vitória**: 100% (5/5 testes)
- **Modelo**: `knight_tour_dqn_b5_e5200.h5`
- **Posição inicial**: Centro (2,2)
- **Movimentos médios**: 24

### 🔄 Outros Tamanhos - Em Desenvolvimento
- **6x6**: Meta ≥90% vitória
- **7x7**: Meta ≥85% vitória  
- **8x8**: Meta ≥80% vitória

## 🚀 Início Rápido

### 1. Instalação
```bash
# Clone o repositório
git clone https://github.com/seu-usuario/knight-tour-ai.git
cd knight-tour-ai

# Instale as dependências
cd RL
pip install -r requirements.txt
```

### 2. Jogar com a IA (5x5)
```bash
# Terminal 1: Servidor web
cd /home/felipe/Projeto/Cavalo_solitario
python start_game.py

# Terminal 2: Servidor IA
cd RL
python app.py
```

### 3. Abrir no navegador
- Acesse: http://localhost:8000
- Selecione tabuleiro **5x5**
- **Clique no CENTRO** (posição 2,2)
- Use **"Dica da IA"** para jogar

## 📁 Estrutura do Projeto

```
RL/
├── 📁 docs/                    # Documentação completa
│   ├── README_NOVA_ESTRUTURA.md
│   ├── COMO_TREINAR.md
│   ├── COMO_JOGAR_COM_IA.md
│   └── win.ipynb              # Análise de resultados
├── 📁 scripts/
│   ├── 📁 training/           # Scripts de treinamento
│   │   ├── run_training.sh    # Execução automática
│   │   ├── auto_train.py      # Treinamento flexível
│   │   └── test_quick_training.py
│   ├── 📁 testing/            # Scripts de teste
│   │   ├── test_best_models.py
│   │   ├── quick_test.py
│   │   ├── validate_winners.py
│   │   └── check_starting_position.py
│   └── 📁 utils/              # Ferramentas utilitárias
│       ├── monitor_training.py
│       ├── manage_models.py
│       ├── configure_model.py
│       ├── update_csv.py
│       └── plot_results.py
├── 📁 models/                 # Modelos treinados
│   ├── 5x5/                  # Modelos 5x5 (validados)
│   ├── 6x6/                  # Modelos 6x6 (em desenvolvimento)
│   ├── 7x7/                  # Modelos 7x7
│   └── 8x8/                  # Modelos 8x8
├── 📁 logs/                  # Logs de treinamento
│   ├── 5x5/
│   ├── 6x6/
│   ├── 7x7/
│   └── 8x8/
├── 🧠 Core Files
│   ├── train.py              # Script principal de treinamento
│   ├── app.py                # Servidor Flask
│   ├── knight_env.py         # Ambiente do jogo
│   ├── dqn_agent.py          # Agente DQN
│   ├── model_config.py       # Configuração de modelos
│   └── requirements.txt      # Dependências
└── README.md                 # Este arquivo
```

## 🎮 Como Jogar

### ⚠️ IMPORTANTE: Posição Inicial Correta

**SEMPRE comece no CENTRO do tabuleiro 5x5:**

```
Tabuleiro 5x5:
    0 1 2 3 4
0:  . . . . .
1:  . . . . .
2:  . . 🐴 . .  ← COMECE AQUI! (Linha 2, Coluna 2)
3:  . . . . .
4:  . . . . .
```

### 🕹️ Passo a Passo

1. **Configure o jogo** (veja seção "Início Rápido")
2. **Abra o navegador**: http://localhost:8000
3. **Selecione tabuleiro 5x5**
4. **Clique no CENTRO** (posição 2,2)
5. **Use "Dica da IA"** e siga as sugestões
6. **Complete o tabuleiro** com 100% de sucesso!

## 🚀 Treinamento de Modelos

### Opções Automáticas (Recomendadas)

#### 1. Script Bash (Mais Simples)
```bash
cd RL
./scripts/training/run_training.sh
```

#### 2. Script Python (Mais Flexível)
```bash
cd RL
python scripts/training/auto_train.py
```

#### 3. Teste Rápido (10 episódios)
```bash
cd RL
python scripts/training/test_quick_training.py
```

### Execução Manual
```bash
cd RL
python train.py
```

### 📊 Saída do Treinamento

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
python scripts/training/auto_train.py --episodes 5000

# Pular verificações (mais rápido)
python scripts/training/auto_train.py --skip-checks

# Ver todas as opções
python scripts/training/auto_train.py --help
```

#### Via Edição do Código:
Edite `train.py` e modifique as variáveis:
```python
BOARD_SIZE = 6          # Tamanho do tabuleiro
EPISODES = 10000        # Número de episódios
BATCH_SIZE = 64         # Tamanho do batch
TARGET_UPDATE_FREQ = 10 # Frequência de atualização
MAX_STEPS_PER_EPISODE = 1000 # Máximo de passos
```

## 🛠️ Ferramentas de Gerenciamento

### Monitor de Treinamento
```bash
# Monitorar em tempo real
python scripts/utils/monitor_training.py monitor 6

# Gerar gráfico
python scripts/utils/monitor_training.py plot 6
```

### Gerenciador de Modelos
```bash
# Status completo
python scripts/utils/manage_models.py status

# Testar modelo específico
python scripts/utils/manage_models.py test 5 --tests 20

# Listar modelos de um tamanho
python scripts/utils/manage_models.py list --size 5

# Definir melhor modelo
python scripts/utils/manage_models.py set-best 5 knight_tour_dqn_b5_e5200.h5
```

### Validação de Modelos
```bash
# Validar vencedores
python scripts/testing/validate_winners.py

# Teste rápido
python scripts/testing/quick_test.py

# Verificar posição inicial
python scripts/testing/check_starting_position.py
```

## 📈 Interpretando Resultados

### Métricas Importantes
- **Score**: Número de passos no episódio
- **Max Visited**: Máximo de casas visitadas
- **Win**: 1 se completou o tour, 0 caso contrário
- **Epsilon**: Taxa de exploração (diminui com o tempo)
- **Vitórias no grupo**: Quantas vitórias nos últimos 100 episódios

### Critérios de Sucesso
- **Bom**: Máximo visitado > 30 casas (6x6)
- **Muito Bom**: Máximo visitado > 33 casas (6x6)
- **Excelente**: Vitórias completas (todas as casas)

## ⏱️ Tempo Estimado

- **CPU**: ~2-4 horas para 10.000 episódios
- **GPU**: ~30-60 minutos para 10.000 episódios

## 🛠️ Solução de Problemas

### Erro de Dependências
```bash
pip install -r requirements.txt
```

### Erro de GPU
O treinamento funciona na CPU, mas será mais lento:
```bash
pip install tensorflow-gpu
```

### IA não funciona
1. Verifique se está usando tabuleiro **5x5**
2. Confirme que começou na posição **centro (2,2)**
3. Verifique se o servidor Flask está rodando

### Interromper Treinamento
Pressione `Ctrl+C` a qualquer momento.

## 🔧 Desenvolvimento

### Estrutura de Desenvolvimento
- **Ambiente**: Gym customizado para Knight's Tour
- **Agente**: DQN com replay buffer e target network
- **Arquitetura**: Rede neural densa com 3 camadas
- **Otimização**: Adam optimizer com learning rate adaptativo

### Adicionar Novo Tamanho
1. **Treinar modelo**:
   ```bash
   python train.py --size 7 --episodes 15000
   ```
2. **Testar diferentes modelos**:
   ```bash
   python scripts/utils/manage_models.py test 7 --model knight_tour_dqn_b7_e12000.h5
   ```
3. **Definir melhor modelo**:
   ```bash
   python scripts/utils/manage_models.py set-best 7 knight_tour_dqn_b7_e12000.h5
   ```

## 📊 Análise Científica

O projeto inclui análise detalhada dos resultados:
- **Validação estatística** dos modelos
- **Comparação de performance** entre tamanhos
- **Análise de convergência** do treinamento
- **Otimização de hiperparâmetros**

Veja `docs/win.ipynb` para análise completa.

## 🤝 Contribuindo

1. Fork o projeto
2. Crie uma branch para sua feature (`git checkout -b feature/AmazingFeature`)
3. Commit suas mudanças (`git commit -m 'Add some AmazingFeature'`)
4. Push para a branch (`git push origin feature/AmazingFeature`)
5. Abra um Pull Request

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## 🙏 Agradecimentos

- **OpenAI Gym** - Framework de RL
- **TensorFlow** - Framework de deep learning
- **Flask** - Servidor web
- **Comunidade RL** - Inspiração e suporte

## 📞 Suporte

Se encontrar problemas:
1. Verifique a documentação em `docs/`
2. Consulte os logs em `logs/`
3. Abra uma issue no GitHub
4. Verifique se todas as dependências estão instaladas

---

🎉 **Divirta-se jogando com a IA e contribuindo para o projeto!** 