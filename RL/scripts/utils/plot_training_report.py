#!/usr/bin/env python3
"""Gera os gráficos de resultado do treinamento (README / LinkedIn).

Lê os CSVs de log de cada execução e avalia os modelos (política gulosa, sem
exploração) a partir de todas as casas iniciais. Saída em docs/images/.

Uso (da raiz do projeto):
    ~/miniconda3/envs/tf_env/bin/python RL/scripts/utils/plot_training_report.py
    ... --no-eval      # só curvas (não carrega TensorFlow)

Para incluir uma execução nova (ex.: 8x8), acrescente um item em RUNS.
"""

import argparse
import json
import os
import sys

RL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ROOT_DIR = os.path.dirname(RL_DIR)
OUT_DIR = os.path.join(ROOT_DIR, 'docs', 'images')

# Execuções exibidas. start: 'centro' | 'aleatorio'. weights: modelo avaliado no mapa.
RUNS = [
    {'n': 5, 'start': 'centro', 'run': '20261124_211422', 'weights': 'final'},
    {'n': 5, 'start': 'aleatorio', 'run': '20261124_235101', 'weights': 'best_greedy'},
    {'n': 6, 'start': 'centro', 'run': '20261124_211804', 'weights': 'final'},
    {'n': 6, 'start': 'aleatorio', 'run': '20261124_212736', 'weights': 'best_greedy'},
    {'n': 7, 'start': 'centro', 'run': '20261124_213819', 'weights': 'final'},
    {'n': 7, 'start': 'aleatorio', 'run': '20261124_214523', 'weights': 'best_greedy'},
    {'n': 8, 'start': 'centro', 'run': '20261124_223740', 'weights': 'final'},
    {'n': 8, 'start': 'aleatorio', 'run': '20261124_225056', 'weights': 'best_greedy'},
]
# Log do código original (antes das correções de 2026-11-24), para o "antes x depois".
BASELINE_6X6 = os.path.join(RL_DIR, 'resultados_passados', 'logs', 'training_log_6x6.csv')
AFTER_6X6_RUN = '20261124_211804'

ROLL = 100  # janela da média móvel (episódios)

# Paleta (dataviz skill, reference palette, modo claro) — validada: azul/laranja.
SURFACE = '#fcfcfb'
INK = '#0b0b0b'
INK_2 = '#52514e'
MUTED = '#898781'
GRID = '#e1e0d9'
AXIS = '#c3c2b7'
NEUTRAL = '#f0efec'
COLOR = {'centro': '#2a78d6', 'aleatorio': '#eb6834'}
LABEL = {'centro': 'Início no centro', 'aleatorio': 'Início em casa aleatória'}
SEQ_BLUE = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5',
            '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']


def setup_matplotlib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
        'axes.edgecolor': AXIS, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
        'axes.titlesize': 13, 'axes.titleweight': 'bold',
        'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.grid': True, 'grid.color': GRID,
        'grid.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
        'legend.frameon': False, 'lines.linewidth': 2,
    })
    return plt


def log_path(n, run):
    return os.path.join(RL_DIR, 'logs', f'training_log_{n}x{n}_{run}.csv')


def load_log(path):
    import pandas as pd
    df = pd.read_csv(path)
    df['win_roll'] = df['Win'].rolling(ROLL, min_periods=ROLL // 2).mean() * 100
    return df


def title_block(fig, title, subtitle):
    fig.text(0.01, 0.985, title, ha='left', va='top', fontsize=16, fontweight='bold', color=INK)
    fig.text(0.01, 0.935, subtitle, ha='left', va='top', fontsize=11, color=INK_2)


def plot_learning_curves(plt):
    sizes = sorted({r['n'] for r in RUNS})
    fig, axes = plt.subplots(1, len(sizes), figsize=(4.6 * len(sizes), 4.6), sharey=True)
    axes = [axes] if len(sizes) == 1 else axes
    for ax, n in zip(axes, sizes):
        for r in [r for r in RUNS if r['n'] == n]:
            df = load_log(log_path(n, r['run']))
            ax.plot(df['Episode'], df['win_roll'], color=COLOR[r['start']], label=LABEL[r['start']])
            last = df['win_roll'].iloc[-1]
            ax.annotate(f"{last:.0f}%", (df['Episode'].iloc[-1], last), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=10, color=INK_2)
        ax.set_title(f'{n}x{n}  ({n * n} casas)', loc='left')
        ax.set_xlabel('Episódio')
        ax.set_ylim(0, 105)
        ax.margins(x=0.08)
    axes[0].set_ylabel(f'Vitórias (%) — média móvel de {ROLL} ep.')
    handles = [plt.Line2D([], [], color=COLOR[k], lw=2.5, label=LABEL[k]) for k in ('centro', 'aleatorio')]
    fig.legend(handles=handles, loc='upper right', ncol=2, bbox_to_anchor=(0.995, 0.99), fontsize=10)
    title_block(fig, 'Curva de aprendizado do DQN', 'Episódios de treino que completaram o passeio do cavalo')
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    out = os.path.join(OUT_DIR, 'curvas_aprendizado.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_before_after(plt):
    if not os.path.exists(BASELINE_6X6):
        print(f'[aviso] baseline não encontrado: {BASELINE_6X6}')
        return None
    before, after = load_log(BASELINE_6X6), load_log(log_path(6, AFTER_6X6_RUN))
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for df, color, label in [(before, MUTED, 'Antes: código original'),
                             (after, COLOR['centro'], 'Depois: DQN corrigido')]:
        pct = (df['Max_Visited'] / 36 * 100).rolling(ROLL, min_periods=ROLL // 2).mean()
        ax.plot(df['Episode'], pct, color=color, label=label)
        ax.annotate(f"{label.split(':')[0]}  {pct.iloc[-1]:.0f}%", (df['Episode'].iloc[-1], pct.iloc[-1]),
                    xytext=(6, 0), textcoords='offset points', va='center', fontsize=10, color=INK_2)
    ax.set_ylim(0, 105)
    ax.set_xlim(0, max(before['Episode'].max(), after['Episode'].max()) * 1.16)
    ax.set_xlabel('Episódio')
    ax.set_ylabel(f'Tabuleiro percorrido (%) — média móvel {ROLL} ep.')
    ax.legend(loc='lower right', fontsize=10)
    title_block(fig, 'Tabuleiro 6x6: antes x depois das correções',
                'Mesma rede; corrigidos alvo do Double DQN (action masking), decaimento do ε e desempenho na GPU')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(OUT_DIR, 'antes_depois_6x6.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def evaluate_runs():
    """Para cada execução: casas visitadas (política gulosa) a partir de cada casa inicial."""
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    sys.path.insert(0, RL_DIR)
    import numpy as np
    from dqn_agent import DQNAgent
    from knight_env import KnightTourEnv

    results = []
    for r in RUNS:
        n = r['n']
        path = os.path.join(RL_DIR, 'models', f'{n}x{n}', f"run_{r['run']}",
                            f"knight_tour_dqn_b{n}_{r['weights']}.weights.h5")
        agent = DQNAgent((n, n, 3), 8, epsilon=0.0, memory_size=1)
        agent.load(path)
        env = KnightTourEnv(n)
        solvable = set(env.solvable_starts())
        grid = np.full((n, n), -1, dtype=int)  # -1 = casa sem solução
        for pos in solvable:
            env.reset()
            env.board[:] = 0
            env.current_pos, env.board[pos], env.path, env.visited_count = pos, 2, [pos], 1
            state, done = env._get_observation(), False
            while not done:
                mask = env._get_valid_moves_mask()
                if not mask.any():
                    break
                state, _, done, _ = env.step(agent.act(state, mask))
            grid[pos] = env.visited_count
        wins = int((grid == n * n).sum())
        results.append({**r, 'grid': grid.tolist(), 'wins': wins, 'solvable': len(solvable),
                        'center_win': bool(grid[n // 2, n // 2] == n * n)})
        print(f"  {n}x{n} {r['start']:9s} vitórias {wins}/{len(solvable)} casas iniciais")
    return results


def plot_start_map(plt, results):
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    cmap = LinearSegmentedColormap.from_list('seq_blue', SEQ_BLUE)
    sizes = sorted({r['n'] for r in results})
    starts = ['centro', 'aleatorio']
    fig, axes = plt.subplots(2, len(sizes), figsize=(3.6 * len(sizes), 8.4), gridspec_kw={'hspace': 0.25})
    axes = np.array(axes).reshape(2, len(sizes))
    for i, start in enumerate(starts):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            ax.grid(False)
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            res = next((r for r in results if r['n'] == n and r['start'] == start), None)
            if res is None:
                ax.text(0.5, 0.5, f'{n}x{n}\nnão treinado', ha='center', va='center', color=MUTED,
                        transform=ax.transAxes)
                continue
            from matplotlib.patches import Rectangle
            grid = np.array(res['grid'], dtype=float)
            ax.axis('off')
            gap = 0.06  # espaçamento entre casas
            for (r_, c_), v in np.ndenumerate(grid):
                frac = v / (n * n)
                face = NEUTRAL if v < 0 else cmap(max(0.0, (frac - 0.5) / 0.5))
                ax.add_patch(Rectangle((c_ - 0.5 + gap / 2, r_ - 0.5 + gap / 2), 1 - gap, 1 - gap,
                                       facecolor=face, edgecolor='none'))
                if v < 0:
                    ax.text(c_, r_, '–', ha='center', va='center', color=MUTED, fontsize=9)
                elif v == n * n:
                    ax.text(c_, r_, '✓', ha='center', va='center', color='#ffffff', fontsize=11,
                            fontweight='bold')
                else:
                    ax.text(c_, r_, f'{int(v)}', ha='center', va='center', fontsize=8,
                            color=INK if frac < 0.8 else '#ffffff')
            ax.set_xlim(-0.5, n - 0.5)
            ax.set_ylim(n - 0.5, -0.5)
            ax.set_aspect('equal')
            ax.set_title(f"{n}x{n} — {res['wins']}/{res['solvable']}", loc='left')
        axes[i, 0].text(-0.12, 0.5, LABEL[start], rotation=90, ha='center', va='center',
                        transform=axes[i, 0].transAxes, fontsize=12, color=INK_2, fontweight='bold')
    title_block(fig, 'De onde o modelo completa o passeio?',
                'Cada casa = início da partida. ✓ completou · número = casas visitadas · – sem solução possível')
    fig.subplots_adjust(left=0.06, right=0.99, bottom=0.02, top=0.85, hspace=0.18, wspace=0.12)
    out = os.path.join(OUT_DIR, 'mapa_casas_iniciais.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def plot_summary(plt, results):
    import numpy as np
    sizes = sorted({r['n'] for r in results})
    fig, ax = plt.subplots(figsize=(8, 4.6))
    width = 0.36
    for k, start in enumerate(['centro', 'aleatorio']):
        xs, ys = [], []
        for j, n in enumerate(sizes):
            res = next((r for r in results if r['n'] == n and r['start'] == start), None)
            if res:
                xs.append(j + (k - 0.5) * (width + 0.03))
                ys.append(100 * res['wins'] / res['solvable'])
        bars = ax.bar(xs, ys, width=width, color=COLOR[start], label=LABEL[start], zorder=3)
        for b, y in zip(bars, ys):
            ax.text(b.get_x() + b.get_width() / 2, y + 2, f'{y:.0f}%', ha='center', va='bottom',
                    fontsize=10, color=INK_2)
    ax.set_xticks(range(len(sizes)))
    ax.set_xticklabels([f'{n}x{n}' for n in sizes], color=INK_2, fontsize=11)
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Casas iniciais vencidas (%)')
    ax.grid(axis='x', visible=False)
    ax.legend(loc='upper left', fontsize=10, ncol=2)
    title_block(fig, 'Generalização: completar o passeio de qualquer casa',
                'Política gulosa (sem exploração), avaliada em todas as casas iniciais com solução')
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    out = os.path.join(OUT_DIR, 'resumo_generalizacao.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--no-eval', action='store_true', help='não avalia modelos (só curvas)')
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    plt = setup_matplotlib()
    outs = [plot_learning_curves(plt), plot_before_after(plt)]
    if not args.no_eval:
        print('Avaliando modelos (CPU)...')
        results = evaluate_runs()
        outs += [plot_start_map(plt, results), plot_summary(plt, results)]
        with open(os.path.join(OUT_DIR, 'avaliacao.json'), 'w') as f:
            json.dump([{k: v for k, v in r.items()} for r in results], f, indent=1)
    for o in outs:
        if o:
            print('gerado:', os.path.relpath(o, ROOT_DIR))


if __name__ == '__main__':
    main()
