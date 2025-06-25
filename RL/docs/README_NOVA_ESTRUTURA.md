# 🏗️ NOVA ESTRUTURA ORGANIZACIONAL - KNIGHT'S TOUR

## 📋 Visão Geral

Sistema reorganizado para suportar múltiplos tamanhos de tabuleiro (5x5, 6x6, 7x7, 8x8) com modelos específicos e gerenciamento automatizado.

## 🚀 Como Executar o Treinamento

### Opções Automáticas (Recomendadas)

#### 1. Script Bash (Mais Simples)
```bash
cd RL
./run_training.sh
```

#### 2. Script Python (Mais Flexível)
```bash
cd RL
python3 auto_train.py
```

#### 3. Teste Rápido (Verificar se funciona)
```bash
cd RL
python3 test_quick_training.py
```

### Execução Manual
```bash
cd RL
python3 train.py
```

### O que você verá no terminal:
```
Training Progress:   6%|███▏                                                  | 599/10000 [03:53<1:13:55,  2.12it/s]
[Resumo dos Episódios 500 a 600]
Melhor Episódio: 594 | Score: 28 | Max Visited: 29/36 | Win: 0 | Epsilon: 0.0514
Vitórias no grupo: 0/100 | Média de casas visitadas: 19.7
Modelo salvo em: models/6x6/knight_tour_dqn_b6_e600.h5
```

📖 **Guia Completo**: Veja `COMO_TREINAR.md` para instruções detalhadas.

## 📁 Estrutura de Diretórios

```
RL/
├── models/                    # Modelos organizados por tamanho
│   ├── 5x5/                  # Modelos para tabuleiro 5x5
│   │   ├── knight_tour_dqn_b5_e5200.h5  # 🥇 Melhor modelo validado
│   │   ├── knight_tour_dqn_b5_e5900.h5  # 🥈 Alternativa excelente
│   │   └── ...               # Outros modelos 5x5
│   ├── 6x6/                  # Modelos 6x6 (em desenvolvimento)
│   ├── 7x7/                  # Modelos 7x7 (em desenvolvimento)
│   └── 8x8/                  # Modelos 8x8 (em desenvolvimento)
├── logs/                     # Logs organizados por tamanho
│   ├── 5x5/                  # Logs de treinamento 5x5
│   ├── 6x6/                  # Logs 6x6
│   ├── 7x7/                  # Logs 7x7
│   └── 8x8/                  # Logs 8x8
├── model_config.py           # 🎯 Sistema de configuração central
├── manage_models.py          # 🛠️ Gerenciador de linha de comando
├── app.py                    # 🌐 Servidor Flask atualizado
├── train.py                  # 🚀 Script de treinamento atualizado
└── ...                       # Outros arquivos do sistema
```

## 🎯 Componentes Principais

### 1. `model_config.py` - Sistema de Configuração Central

**Responsabilidades:**
- Mapeia o melhor modelo para cada tamanho de tabuleiro
- Gerencia modelos de fallback automático
- Mantém metadados (taxa de vitória, posição inicial, etc.)
- Fornece API unificada para acesso aos modelos

**Configuração Atual:**
- **5x5**: `knight_tour_dqn_b5_e5200.h5` (100% win rate, validado)
- **6x6**: Em desenvolvimento
- **7x7**: Em desenvolvimento  
- **8x8**: Em desenvolvimento

### 2. `manage_models.py` - Gerenciador CLI

**Comandos Disponíveis:**

```bash
# Listar status de todos os modelos
python manage_models.py status

# Listar modelos de um tamanho específico
python manage_models.py list --size 5

# Testar um modelo
python manage_models.py test 5 --model knight_tour_dqn_b5_e5200.h5 --tests 20

# Definir melhor modelo para um tamanho
python manage_models.py set-best 5 knight_tour_dqn_b5_e5200.h5

# Treinar novo modelo
python manage_models.py train 6 --episodes 10000
```

### 3. `app.py` - Servidor Flask Atualizado

**Novos Recursos:**
- Detecção automática do tamanho do tabuleiro
- Carregamento automático do melhor modelo para cada tamanho
- Cache de agentes para melhor performance
- Endpoint `/model_status` para verificar modelos disponíveis
- Mensagens de erro informativas

### 4. `train.py` - Treinamento Organizado

**Novos Recursos:**
- Suporte a argumentos de linha de comando
- Criação automática de diretórios organizados
- Early stopping baseado em taxa de vitória
- Logs estruturados por tamanho

**Exemplo de Uso:**
```bash
# Treinar modelo 6x6
python train.py --size 6 --episodes 15000 --target-win-rate 0.90

# Treinar modelo 7x7 com menos episódios
python train.py --size 7 --episodes 8000 --save-interval 200
```

## 🚀 Como Usar o Novo Sistema

### 1. Verificar Status Atual
```bash
python manage_models.py status
```

### 2. Testar Modelo 5x5 Existente
```bash
python manage_models.py test 5 --tests 10
```

### 3. Treinar Modelo 6x6
```bash
python train.py --size 6 --episodes 12000
```

### 4. Configurar Melhor Modelo 6x6
```bash
# Após treinamento, definir o melhor modelo
python manage_models.py set-best 6 knight_tour_dqn_b6_e8500.h5
```

### 5. Usar no Jogo
O servidor Flask automaticamente detecta o tamanho do tabuleiro e usa o melhor modelo:

```bash
python app.py
# Servidor se adapta automaticamente a 5x5, 6x6, 7x7, 8x8
```

## 📊 Configurações por Tamanho

### 5x5 - ✅ PRONTO
- **Modelo**: `knight_tour_dqn_b5_e5200.h5`
- **Taxa de vitória**: 100%
- **Posição inicial**: Centro (2,2)
- **Movimentos médios**: 24
- **Status**: Validado cientificamente

### 6x6 - 🔄 EM DESENVOLVIMENTO
- **Posição inicial sugerida**: Centro (2,2)
- **Meta de vitória**: ≥90%
- **Episódios sugeridos**: 12.000-15.000

### 7x7 - 🔄 EM DESENVOLVIMENTO  
- **Posição inicial sugerida**: Centro (3,3)
- **Meta de vitória**: ≥85%
- **Episódios sugeridos**: 15.000-20.000

### 8x8 - 🔄 EM DESENVOLVIMENTO
- **Posição inicial sugerida**: Centro (3,3) ou (4,4)
- **Meta de vitória**: ≥80%
- **Episódios sugeridos**: 20.000-30.000

## 🛠️ Fluxo de Trabalho Recomendado

### Para Adicionar Novo Tamanho:

1. **Treinar Modelo**
   ```bash
   python train.py --size 6 --episodes 15000
   ```

2. **Testar Diferentes Modelos**
   ```bash
   python manage_models.py test 6 --model knight_tour_dqn_b6_e10000.h5
   python manage_models.py test 6 --model knight_tour_dqn_b6_e12000.h5
   ```

3. **Definir Melhor Modelo**
   ```bash
   python manage_models.py set-best 6 knight_tour_dqn_b6_e12000.h5
   ```

4. **Validar no Jogo**
   - Iniciar servidor: `python app.py`
   - Testar no navegador com tabuleiro 6x6

## 🎯 Vantagens da Nova Estrutura

### ✅ Organização
- Modelos separados por tamanho
- Logs estruturados
- Configuração centralizada

### ✅ Automatização
- Detecção automática de tamanho
- Seleção automática do melhor modelo
- Fallback inteligente

### ✅ Escalabilidade
- Fácil adição de novos tamanhos
- Sistema de validação padronizado
- Gerenciamento via linha de comando

### ✅ Manutenibilidade
- Código modular
- Configuração separada da lógica
- Documentação integrada

## 🚀 Próximos Passos

1. **Treinar modelos 6x6, 7x7, 8x8**
2. **Validar cientificamente cada modelo**
3. **Configurar como melhores modelos**
4. **Integrar com interface web para seleção de tamanho**
5. **Implementar análise comparativa entre tamanhos**

## 📝 Comandos Úteis

```bash
# Status completo
python manage_models.py status

# Testar modelo atual do 5x5
python manage_models.py test 5

# Listar apenas modelos 5x5  
python manage_models.py list --size 5

# Treinar 6x6 com early stopping em 95%
python train.py --size 6 --target-win-rate 0.95

# Testar específico com 50 testes
python manage_models.py test 6 --model knight_tour_dqn_b6_e8000.h5 --tests 50
```

---

🎉 **Sistema totalmente reorganizado e pronto para expansão multi-tamanho!** 