# Estratégia de Treinamento

## Diagnóstico

O DQN atual não chega a 0% de vitória gulosa no 5x5 após ~7 000 episódios e fica em ~20/36 casas no 6x6. Causas, em ordem de impacto estimado:

1. **Eficiência amostral** — recompensa terminal rara; o agente quase nunca vê uma vitória no 6x6 por exploração.
2. **Throughput** — ~100 ms por update (audit A1) limita a ~10 updates/s.
3. **Alvo incorreto** — bootstrap sobre ações inválidas (audit A3).
4. **Estado pobre** — sem graus futuros nem progresso; a rede precisa inferir Warnsdorff a partir de pixels.

## Plano em estágios

### Estágio 0 — Correções de base (sem mudar algoritmo)
- Train step `tf.function` + máscara no alvo + `jit_compile=False`.
- Métrica oficial: **taxa de vitória gulosa** em conjunto fixo de casas iniciais (todas as casas solúveis), não vitórias durante exploração.

### Estágio 1 — Behavior Cloning a partir do oráculo
- Gerar demonstrações: trajetórias completas do oráculo (DFS+Warnsdorff) de todas as casas iniciais solúveis, com desempate aleatório para diversidade.
- Rótulo por estado: conjunto de ações "boas" (mantêm o tour completável) → perda de entropia cruzada multi-rótulo sobre logits mascarados.
- Esperado: alta taxa de vitória rapidamente (minutos de GPU), serve de inicialização.

### Estágio 2 — DQfD (RL from Demonstrations)
Perda total por batch: `L = L_TD(1-step) + λ_n · L_TD(n-step) + λ_E · L_margin + λ_L2 · ||θ||²`
- `L_margin` (large-margin supervisionada) só em transições de demonstração: `max_a [Q(s,a) + l(a_E,a)] − Q(s,a_E)`, margem 0,8, só sobre ações válidas.
- Amostragem: proporção mínima de demonstrações no batch (ex. 25%), decaindo com o tempo.
- n-step (n = 5–10) propaga a recompensa terminal mais rápido.

### Estágio 3 — Online + Human-in-the-Loop
Fontes de dados e peso:

| Fonte | Uso | Peso de demonstração |
|---|---|---|
| `oracle` | margem + TD | 1,0 |
| `human`, jogada que o oráculo confirma como boa | margem + TD | 1,0 × fator de resultado |
| `human`, jogada que o oráculo marca como fatal | **só TD** (aprende a consequência) | 0 |
| `human`, oráculo `unknown` (orçamento estourado) | TD + margem pequena se partida vencida | 0,3 |
| `collab_intervention` | DAgger: estado visitado pela política, rótulo = escolha humana validada pelo oráculo | 1,0 |
| `agent` (autônomo) | só TD | 0 |

Assim jogadas humanas não são tratadas como ótimas (RF2): o oráculo arbitra a qualidade.

### DAgger
O oráculo é o especialista. Loop: rodar política atual → em cada estado visitado, rotular com `oracle.good_actions(s)` → agregar ao dataset → re-treinar. Intervenções humanas no modo colaborativo são uma variante com rótulo humano. Custo do oráculo no 6x6: < 1 ms/estado (medido).

## Representação de estado (proposta, validar em EXP-002)

Canais `(N,N,C)` float32 gerados a partir do bitboard:
1. posição do cavalo
2. visitadas
3. destinos válidos
4. grau (nº de vizinhos livres) de cada casa livre / 8 — Warnsdorff explícito
5. constante `visited_count / N²` (progresso)
6. borda/padding: 1 dentro do tabuleiro (necessário p/ rede independente de N)

Avaliar contra o estado atual de 3 canais. Conectividade/isolamento (BFS de componentes): canal binário "casa em componente sem a posição atual" — só se EXP-002 mostrar ganho, custa O(N²) por passo.

## Exploração

- Manter ε-greedy guiado por Warnsdorff (já existe) com ε por episódio.
- Alternativas: Boltzmann sobre Q mascarado; ε baixo depois do pré-treino BC (0,05).

## Currículo de tamanhos

5x5 (casas pares) → 6x6 → 7x7, reaproveitando pesos se ADR-007 (rede totalmente convolucional) for aceita.
