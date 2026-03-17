#!/usr/bin/env python3
"""
Sistema de Configuração de Modelos por Tamanho de Tabuleiro
Gerencia automaticamente o melhor modelo para cada escala (5x5, 6x6, 7x7, etc.)
"""

import os
import glob
from typing import Dict, Optional, Tuple, List

class ModelConfig:
    """Gerenciador de configuração de modelos por tamanho"""
    
    def __init__(self, models_base_dir: str = 'models'):
        self.models_base_dir = models_base_dir
        
        # Configuração dos melhores modelos por tamanho
        # Baseado na validação científica realizada
        self.BEST_MODELS = {
            '5x5': {
                'model_file': 'knight_tour_dqn_b5_e5200.weights.h5',
                'win_rate': 100.0,
                'training_range': 'Episodes 5200-5299 (98% wins)',
                'validation': '50/50 tests = 100% success',
                'avg_moves': 24,
                'starting_position': (2, 2),  # Centro
                'description': '🥇 Modelo campeão - Validado cientificamente'
            },
            '6x6': {
                'model_file': None,  # A ser definido após treinamento
                'win_rate': None,
                'training_range': 'TBD',
                'validation': 'TBD',
                'avg_moves': None,
                'starting_position': (2, 2),  # Centro provável
                'description': '🔄 Em desenvolvimento'
            },
            '7x7': {
                'model_file': None,
                'win_rate': None,
                'training_range': 'TBD',
                'validation': 'TBD',
                'avg_moves': None,
                'starting_position': (3, 3),  # Centro provável
                'description': '🔄 Em desenvolvimento'
            },
            '8x8': {
                'model_file': None,
                'win_rate': None,
                'training_range': 'TBD',
                'validation': 'TBD',
                'avg_moves': None,
                'starting_position': (3, 3),  # Centro provável
                'description': '🔄 Em desenvolvimento'
            }
        }
        
        # Alternativas por tamanho (fallback caso o melhor não esteja disponível)
        self.FALLBACK_MODELS = {
            '5x5': [
                'knight_tour_dqn_b5_e5900.weights.h5',  # 100% validado
                'knight_tour_dqn_b5_e6400.weights.h5',  # 100% validado
                'knight_tour_dqn_b5_e5700.weights.h5',  # 95% range
                # Fallback para formato antigo
                'knight_tour_dqn_b5_e5900.h5',
                'knight_tour_dqn_b5_e6400.h5',
                'knight_tour_dqn_b5_e5700.h5',
            ]
        }
    
    def get_board_size_key(self, board_size: int) -> str:
        """Converte tamanho numérico para chave string"""
        return f"{board_size}x{board_size}"
    
    def get_model_path(self, board_size: int) -> Optional[str]:
        """Retorna o caminho do melhor modelo para o tamanho especificado"""
        size_key = self.get_board_size_key(board_size)
        
        if size_key not in self.BEST_MODELS:
            return None
        
        model_info = self.BEST_MODELS[size_key]
        
        if not model_info['model_file']:
            return None
        
        model_path = os.path.join(
            self.models_base_dir, 
            size_key, 
            model_info['model_file']
        )
        
        # Verifica se o modelo existe
        if os.path.exists(model_path):
            return model_path
        
        # Fallback: procura alternativas
        return self._find_fallback_model(board_size)
    
    def _find_fallback_model(self, board_size: int) -> Optional[str]:
        """Encontra modelo alternativo se o principal não estiver disponível"""
        size_key = self.get_board_size_key(board_size)
        
        if size_key in self.FALLBACK_MODELS:
            for fallback_model in self.FALLBACK_MODELS[size_key]:
                fallback_path = os.path.join(
                    self.models_base_dir, 
                    size_key, 
                    fallback_model
                )
                if os.path.exists(fallback_path):
                    return fallback_path
        
        # Último recurso: pega qualquer modelo disponível para o tamanho
        # Primeiro tenta .weights.h5 (formato novo), depois .h5 (formato antigo)
        pattern = os.path.join(self.models_base_dir, size_key, '*.weights.h5')
        available_models = glob.glob(pattern)
        
        if not available_models:
            pattern = os.path.join(self.models_base_dir, size_key, '*.h5')
            available_models = glob.glob(pattern)
        
        if available_models:
            # Ordena por episódio (maior primeiro)
            try:
                available_models.sort(
                    key=lambda x: int(x.split('_e')[1].split('.')[0]) 
                    if '_e' in x else 0, 
                    reverse=True
                )
                return available_models[0]
            except:
                return available_models[0]
        
        return None
    
    def get_model_info(self, board_size: int) -> Dict:
        """Retorna informações completas do modelo para o tamanho"""
        size_key = self.get_board_size_key(board_size)
        
        if size_key not in self.BEST_MODELS:
            return {
                'size': size_key,
                'available': False,
                'error': f'Tamanho {size_key} não suportado'
            }
        
        model_info = self.BEST_MODELS[size_key].copy()
        model_path = self.get_model_path(board_size)
        
        model_info.update({
            'size': size_key,
            'model_path': model_path,
            'available': model_path is not None,
            'is_fallback': self._is_fallback_model(board_size, model_path)
        })
        
        return model_info
    
    def _is_fallback_model(self, board_size: int, model_path: Optional[str]) -> bool:
        """Verifica se é um modelo de fallback"""
        if not model_path:
            return False
        
        size_key = self.get_board_size_key(board_size)
        best_model = self.BEST_MODELS[size_key]['model_file']
        
        if not best_model:
            return True
        
        return not model_path.endswith(best_model)
    
    def list_available_sizes(self) -> List[int]:
        """Lista todos os tamanhos com modelos disponíveis"""
        available_sizes = []
        
        for size_key in self.BEST_MODELS.keys():
            board_size = int(size_key.split('x')[0])
            if self.get_model_path(board_size):
                available_sizes.append(board_size)
        
        return sorted(available_sizes)
    
    def update_best_model(self, board_size: int, model_file: str, 
                         win_rate: float, validation: str, 
                         avg_moves: int, starting_position: Tuple[int, int],
                         training_range: str = None):
        """Atualiza o melhor modelo para um tamanho após validação"""
        size_key = self.get_board_size_key(board_size)
        
        if size_key not in self.BEST_MODELS:
            self.BEST_MODELS[size_key] = {}
        
        self.BEST_MODELS[size_key].update({
            'model_file': model_file,
            'win_rate': win_rate,
            'validation': validation,
            'avg_moves': avg_moves,
            'starting_position': starting_position,
            'training_range': training_range or 'Custom validation',
            'description': f'🥇 Modelo validado - {win_rate}% win rate'
        })
        
        print(f"✅ Modelo {model_file} configurado como melhor para {size_key}")
        print(f"   Taxa de vitória: {win_rate}%")
        print(f"   Validação: {validation}")
    
    def print_status(self):
        """Imprime status de todos os modelos configurados"""
        print("🎯 STATUS DOS MODELOS POR TAMANHO")
        print("=" * 60)
        
        for size_key in sorted(self.BEST_MODELS.keys()):
            board_size = int(size_key.split('x')[0])
            info = self.get_model_info(board_size)
            
            status = "✅ DISPONÍVEL" if info['available'] else "❌ INDISPONÍVEL"
            fallback = " (FALLBACK)" if info.get('is_fallback', False) else ""
            
            print(f"\n📋 {size_key}:")
            print(f"   Status: {status}{fallback}")
            print(f"   Descrição: {info['description']}")
            
            if info['available']:
                print(f"   Arquivo: {os.path.basename(info['model_path'])}")
                if info['win_rate']:
                    print(f"   Taxa de vitória: {info['win_rate']}%")
                if info['starting_position']:
                    print(f"   Posição inicial: {info['starting_position']}")
            else:
                print(f"   Ação: Treinar modelo para {size_key}")

# Instância global para uso fácil
model_config = ModelConfig()

# Funções de conveniência
def get_best_model_path(board_size: int) -> Optional[str]:
    """Função de conveniência para obter o melhor modelo"""
    return model_config.get_model_path(board_size)

def get_model_info(board_size: int) -> Dict:
    """Função de conveniência para obter info do modelo"""
    return model_config.get_model_info(board_size)

def update_best_model(board_size: int, **kwargs):
    """Função de conveniência para atualizar melhor modelo"""
    return model_config.update_best_model(board_size, **kwargs) 