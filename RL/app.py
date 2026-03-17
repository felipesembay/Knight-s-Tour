import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from dqn_agent import DQNAgent
from knight_env import KnightTourEnv
from model_config import get_best_model_path, get_model_info
from typing import Tuple, Dict
import os

# --- Configuração ---
app = Flask(__name__)
CORS(app)  # Permite requisições de origens diferentes (do nosso HTML para o servidor Flask)

# Cache de agentes por tamanho para melhor performance
agents_cache = {}

def load_agent_for_size(board_size: int) -> Tuple[DQNAgent, Dict]:
    """Carrega o agente apropriado para o tamanho do tabuleiro"""
    
    # Verifica cache primeiro
    if board_size in agents_cache:
        return agents_cache[board_size]
    
    # Obtém informações do modelo
    model_info = get_model_info(board_size)
    
    if not model_info['available']:
        raise Exception(f"Nenhum modelo disponível para tabuleiro {board_size}x{board_size}")
    
    # Configura ambiente para obter dimensões corretas
    env = KnightTourEnv(board_size=board_size)
    state_shape = env.observation_space.shape
    action_size = env.action_space.n
    
    # Cria e carrega o agente
    agent = DQNAgent(state_shape, action_size, epsilon=0.0)
    
    try:
        agent.load(model_info['model_path'])
        
        # Log de carregamento
        status = "FALLBACK" if model_info['is_fallback'] else "PRINCIPAL"
        print(f"🎯 Modelo {status} carregado para {board_size}x{board_size}:")
        print(f"   Arquivo: {os.path.basename(model_info['model_path'])}")
        print(f"   Descrição: {model_info['description']}")
        if model_info['win_rate']:
            print(f"   Taxa de vitória: {model_info['win_rate']}%")
        
        # Cache o agente e info
        agents_cache[board_size] = (agent, model_info)
        
        return agent, model_info
        
    except Exception as e:
        raise Exception(f"Erro ao carregar modelo para {board_size}x{board_size}: {e}")

# --- API Endpoints ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    Recebe o estado do tabuleiro e retorna a melhor ação prevista pelo agente.
    Detecta automaticamente o tamanho do tabuleiro e usa o modelo apropriado.
    """
    try:
        data = request.json
        
        if 'board' not in data:
            return jsonify({'error': 'O estado do tabuleiro (board) não foi fornecido.'}), 400
            
        # Converte o estado recebido (2D) e detecta tamanho
        board_2d = np.array(data['board'], dtype=np.int32)
        board_size = board_2d.shape[0]
        
        # Valida se é quadrado
        if board_2d.shape[0] != board_2d.shape[1]:
            return jsonify({'error': 'Tabuleiro deve ser quadrado (NxN)'}), 400
        
        # Carrega o agente apropriado para o tamanho
        try:
            agent, model_info = load_agent_for_size(board_size)
        except Exception as e:
            return jsonify({
                'error': f'Modelo não disponível para tabuleiro {board_size}x{board_size}',
                'details': str(e),
                'suggestion': 'Tente usar um tabuleiro 5x5 (modelo validado disponível)'
            }), 400
        
        # Reconstrói o ambiente para obter a máscara e o estado 3D
        env = KnightTourEnv(board_size=board_size)
        env.board = np.zeros((board_size, board_size), dtype=np.int8)
        current_pos = None
        visited_count = 0
        
        for r in range(board_size):
            for c in range(board_size):
                if board_2d[r, c] == 1:
                    env.board[r, c] = 1
                    visited_count += 1
                elif board_2d[r, c] == 2:
                    env.board[r, c] = 2
                    current_pos = (r, c)
                    visited_count += 1
        
        env.current_pos = current_pos
        env.visited_count = visited_count

        # Obtém o estado no formato correto (3 canais) e a máscara
        state_3_channels = env._get_observation()
        valid_moves_mask = env._get_valid_moves_mask()
        
        # O agente escolhe a melhor ação (sem exploração)
        action_index = agent.act(state_3_channels, valid_moves_mask)
        
        # Converte o índice da ação para um movimento (ex: [-2, 1])
        move = env.knight_moves[int(action_index)]
        
        # Retorna a ação como JSON com info do modelo
        return jsonify({
            'action_index': int(action_index),
            'move': {
                'd_row': int(move[0]),
                'd_col': int(move[1])
            },
            'model_info': {
                'board_size': f"{board_size}x{board_size}",
                'model_file': os.path.basename(model_info['model_path']),
                'win_rate': model_info.get('win_rate'),
                'is_fallback': model_info.get('is_fallback', False)
            }
        })

    except Exception as e:
        # Adiciona um print para debug no console do Flask
        print(f"Erro no endpoint /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/play_with_ai', methods=['POST'])
def play_with_ai():
    """
    Recebe o estado do tabuleiro e retorna a sequência de movimentos da IA até o fim do jogo.
    Detecta automaticamente o tamanho e usa o modelo apropriado.
    """
    try:
        data = request.json
        if 'board' not in data:
            return jsonify({'error': 'O estado do tabuleiro (board) não foi fornecido.'}), 400
            
        board = np.array(data['board'], dtype=np.int32)
        board_size = board.shape[0]
        
        # Valida se é quadrado
        if board.shape[0] != board.shape[1]:
            return jsonify({'error': 'Tabuleiro deve ser quadrado (NxN)'}), 400
        
        # Carrega o agente apropriado
        try:
            agent, model_info = load_agent_for_size(board_size)
        except Exception as e:
            return jsonify({
                'error': f'Modelo não disponível para tabuleiro {board_size}x{board_size}',
                'details': str(e)
            }), 400
        
        # Reconstrói o ambiente
        env = KnightTourEnv(board_size=board_size)
        env.board = np.zeros((board_size, board_size), dtype=np.int8)
        visited_count = 0
        current_pos = None
        
        for r in range(board_size):
            for c in range(board_size):
                if board[r, c] == 1:
                    env.board[r, c] = 1
                    visited_count += 1
                elif board[r, c] == 2:
                    env.board[r, c] = 2
                    current_pos = (r, c)
                    visited_count += 1
        
        env.current_pos = current_pos
        env.visited_count = visited_count
        env.path = [current_pos]  # Caminho atual
        
        # Executa a IA até o fim do jogo
        moves = []
        done = False
        state = env._get_observation()
        max_moves = board_size * board_size * 2  # Limite de segurança
        
        while not done and len(moves) < max_moves:
            valid_moves_mask = env._get_valid_moves_mask()
            if not valid_moves_mask.any():
                break
                
            action = agent.act(state, valid_moves_mask)
            move = env.knight_moves[int(action)]
            moves.append({
                'action_index': int(action), 
                'd_row': int(move[0]), 
                'd_col': int(move[1])
            })
            
            state, reward, done, info = env.step(action)
            if done:
                break
        
        return jsonify({
            'moves': moves,
            'success': env.visited_count == board_size * board_size,
            'visited_count': int(env.visited_count),
            'total_squares': int(board_size * board_size),
            'model_info': {
                'board_size': f"{board_size}x{board_size}",
                'model_file': os.path.basename(model_info['model_path']),
                'win_rate': model_info.get('win_rate'),
                'is_fallback': model_info.get('is_fallback', False)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model_status', methods=['GET'])
def model_status():
    """Endpoint para verificar status dos modelos disponíveis"""
    try:
        from model_config import model_config
        
        status = {}
        for size in [5, 6, 7, 8]:
            info = get_model_info(size)
            status[f"{size}x{size}"] = {
                'available': info['available'],
                'description': info['description'],
                'model_file': os.path.basename(info['model_path']) if info['model_path'] else None,
                'win_rate': info.get('win_rate'),
                'starting_position': info.get('starting_position'),
                'is_fallback': info.get('is_fallback', False)
            }
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Mostra status inicial dos modelos
    print("🚀 INICIANDO SERVIDOR FLASK - CAVALO SOLITÁRIO")
    print("=" * 60)
    
    try:
        from model_config import model_config
        model_config.print_status()
    except Exception as e:
        print(f"⚠️  Erro ao verificar status dos modelos: {e}")
    
    print("\n🌐 Servidor rodando em: http://localhost:5018")
    print("🎮 Para usar no jogo, acesse: http://localhost:8018")
    print("=" * 60)
    
    # Roda o servidor Flask na porta 5018
    app.run(port=5018, debug=True) 