# Especificação do Projeto — Knight's Tour Human-in-the-Loop RL

Fonte: briefing do proprietário (2026-11-24). **O briefing recebido termina no início da seção 9 ("Implemente um Experience Replay Buffer persistente…"). As seções 9 em diante ainda não foram recebidas** — itens marcados *[pendente]* dependem delas.

## Objetivo

Evoluir o jogo/IA existente (DQN em Keras) para uma arquitetura híbrida em que a rede aprende também com as partidas do usuário, combinando DQN, Human-in-the-Loop, Imitation Learning / Behavior Cloning, RL from Demonstrations (DQfD), aprendizado online, Experience Replay e heurística de Warnsdorff. Treino incremental, persistente, interrompível e retomável, dentro de uma RTX 4060 8 GB.

Restrição: **não reconstruir do zero**; preservar funcionalidades e modelos existentes.

## Requisitos funcionais

### RF1 — Modo clássico
Jogo atual inalterado; inferência não altera parâmetros.

### RF2 — Modo treinamento humano
Usuário joga todas as jogadas. Registrar cada movimento válido, estado anterior/posterior, recompensa, término, resultado final. Persistir e disponibilizar para treino incremental. Opt-in explícito. Partidas incompletas armazenadas e distinguidas: `completed_win`, `completed_stuck`, `abandoned`. Jogadas humanas **não** são presumidas ótimas.

### RF3 — Modo treinamento autônomo
IA joga sozinha. Configurável: episódios, posição inicial, N, estratégia de exploração, dispositivo, limite de tempo, frequência de treino, frequência de checkpoint, limite de recursos, velocidade. Pausar, retomar, interromper.

### RF4 — Modo colaborativo
IA sugere/joga; usuário pode intervir. Registrar: sugestão da IA, jogada escolhida, flag de intervenção, valores Q, estado, máscara de ações válidas, resultado. Intervenções viram demonstrações ponderadas por qualidade e resultado. Avaliar DAgger com rotulador especialista (humano ou artificial).

### RF5 — Representação de estado
Mínimo: máscara de visitadas, posição, N, nº de visitadas, máscara de movimentos válidos. Avaliar: graus futuros, casas livres, conectividade, distância de borda, grau por casa, isolamento de regiões. Só adicionar o que tiver benefício medido. Avaliar bitboards.

### RF6 — Ações e Action Masking
8 ações (ordem `KNIGHT_MOVES`). Máscara deve impedir: fora do tabuleiro, casa visitada, ação inválida na exploração, na inferência e **nos alvos do DQN**. Tratar terminal sem movimentos. Se arquitetura nova for incompatível, preservar modelos antigos e documentar migração.

### RF7 — Persistência de experiências *[parcial / pendente]*
Experience Replay Buffer persistente. Detalhes do briefing não recebidos.

### RF8+ — *[pendente]*

## Requisitos não funcionais

- Ubuntu, RTX 4060 8 GB, CPU ou GPU selecionável.
- Treino retomável (pesos + otimizador + ε + buffer + contadores + RNG).
- Documentação suficiente para outro agente continuar sem histórico de conversa (`HANDOFF.md`).
- Nada de treinos longos durante auditoria.
