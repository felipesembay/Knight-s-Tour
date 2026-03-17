# 📁 Pasta de Modelos Treinados

Esta pasta armazena os modelos treinados de Deep Q-Learning (DQN) para o jogo Cavalo Solitário.

## 🎯 Status Atual

**Nenhum modelo treinado encontrado.**

## 🚀 Como Treinar um Modelo

### 1. Treinamento Rápido (5x5)
```bash
cd RL
python train.py
```

### 2. Parâmetros Padrão
- **Tabuleiro**: 5x5
- **Episódios**: 10.000
- **Batch Size**: 64

### 3. Acompanhar o Progresso
Durante o treinamento, você verá:
- ✅ Score (movimentos por episódio)
- ✅ Taxa de vitórias
- ✅ Epsilon (exploração vs. exploitação)
- ✅ Recompensa média

### 4. Arquivos Gerados
Após o treinamento:
- `models/best_model_5x5_*.h5` - Modelo treinado
- `logs/5x5/training_*.csv` - Métricas de treinamento

## 📊 Visualizar Resultados

```bash
cd RL
python scripts/utils/plot_results.py
```

## 💡 Dicas

- **Para tabuleiro 5x5**: ~10.000 episódios (30-60 minutos)
- **Para tabuleiros maiores**: Ajuste os parâmetros em `RL/train.py`
- **GPU**: Acelera significativamente o treinamento

## 🔗 Mais Informações

Consulte:
- [RL/docs/COMO_TREINAR.md](../RL/docs/COMO_TREINAR.md)
- [README.md principal](../README.md)
