#!/usr/bin/env python3
"""
Script para configurar qual modelo DQN usar no jogo Cavalo Solitário
"""

import os
import glob
import shutil

def list_available_models():
    """Lista todos os modelos disponíveis"""
    models_dir = 'models'
    if not os.path.exists(models_dir):
        print("❌ Diretório 'models' não encontrado!")
        return []
    
    model_files = glob.glob(os.path.join(models_dir, '*.h5'))
    
    if not model_files:
        print("❌ Nenhum modelo encontrado no diretório 'models'!")
        return []
    
    # Extrai informações dos modelos
    models_info = []
    for file_path in model_files:
        filename = os.path.basename(file_path)
        try:
            episode_str = filename.split('_e')[1].split('.')[0]
            if episode_str == 'final':
                episode = 99999  # Para ordenação
                episode_display = "final"
            else:
                episode = int(episode_str)
                episode_display = str(episode)
            
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            
            models_info.append({
                'filename': filename,
                'path': file_path,
                'episode': episode,
                'episode_display': episode_display,
                'size_mb': size_mb
            })
        except:
            continue
    
    # Ordena por episódio
    models_info.sort(key=lambda x: x['episode'])
    
    return models_info

def show_model_recommendations():
    """Mostra recomendações baseadas na análise realizada"""
    print("\n🏆 RECOMENDAÇÕES BASEADAS NA VALIDAÇÃO:")
    print("=" * 60)
    
    recommendations = [
        {
            'model': 'knight_tour_dqn_b5_e5200.h5',
            'rating': '🥇 MELHOR ESCOLHA',
            'description': 'Taxa de vitória: 100% | Melhor range de treinamento (98%)',
            'episode': 5200
        },
        {
            'model': 'knight_tour_dqn_b5_e5900.h5', 
            'rating': '🥈 EXCELENTE',
            'description': 'Taxa de vitória: 100% | Range de alta performance (97%)',
            'episode': 5900
        },
        {
            'model': 'knight_tour_dqn_b5_e6400.h5',
            'rating': '🥉 MUITO BOM',
            'description': 'Taxa de vitória: 100% | Range tardio de alta performance',
            'episode': 6400
        }
    ]
    
    for rec in recommendations:
        print(f"{rec['rating']:>15}: {rec['model']}")
        print(f"{'':>17} {rec['description']}")
        print()

def configure_model_in_app(model_filename):
    """Configura o modelo no arquivo app.py"""
    app_file = 'app.py'
    
    if not os.path.exists(app_file):
        print(f"❌ Arquivo {app_file} não encontrado!")
        return False
    
    # Lê o arquivo atual
    with open(app_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Substitui a linha do modelo
    lines = content.split('\n')
    new_lines = []
    
    for line in lines:
        if line.strip().startswith('BEST_MODEL = '):
            new_lines.append(f"BEST_MODEL = '{model_filename}'")
            print(f"✅ Configurado para usar: {model_filename}")
        else:
            new_lines.append(line)
    
    # Escreve o arquivo modificado
    with open(app_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))
    
    return True

def main():
    print("🎯 CONFIGURADOR DE MODELO - CAVALO SOLITÁRIO")
    print("=" * 60)
    
    # Lista modelos disponíveis
    models = list_available_models()
    
    if not models:
        return
    
    print(f"\n📁 MODELOS DISPONÍVEIS ({len(models)} encontrados):")
    print("-" * 60)
    print(f"{'#':>3} {'Episódio':>8} {'Arquivo':>25} {'Tamanho':>10}")
    print("-" * 60)
    
    for i, model in enumerate(models, 1):
        print(f"{i:>3} {model['episode_display']:>8} {model['filename']:>25} {model['size_mb']:>8.1f}MB")
    
    # Mostra recomendações
    show_model_recommendations()
    
    # Modelo recomendado padrão
    recommended_model = 'knight_tour_dqn_b5_e5200.h5'
    
    print("🎮 CONFIGURAÇÃO:")
    print("-" * 30)
    print(f"1. Usar modelo RECOMENDADO: {recommended_model}")
    print("2. Escolher manualmente da lista acima")
    print("3. Sair sem alterar")
    
    while True:
        try:
            choice = input("\nEscolha uma opção (1-3): ").strip()
            
            if choice == '1':
                # Verifica se o modelo recomendado existe
                if any(m['filename'] == recommended_model for m in models):
                    if configure_model_in_app(recommended_model):
                        print(f"\n🎉 SUCESSO! Modelo configurado: {recommended_model}")
                        print("   Agora você pode iniciar o jogo com:")
                        print("   python ../start_game.py")
                        print("   ou")
                        print("   bash ../start_game.sh")
                    break
                else:
                    print(f"❌ Modelo recomendado não encontrado: {recommended_model}")
                    print("   Escolha outro da lista.")
                    continue
            
            elif choice == '2':
                model_num = int(input(f"Digite o número do modelo (1-{len(models)}): "))
                if 1 <= model_num <= len(models):
                    selected_model = models[model_num - 1]
                    if configure_model_in_app(selected_model['filename']):
                        print(f"\n🎉 SUCESSO! Modelo configurado: {selected_model['filename']}")
                        print("   Agora você pode iniciar o jogo!")
                    break
                else:
                    print(f"❌ Número inválido. Digite entre 1 e {len(models)}.")
                    
            elif choice == '3':
                print("👋 Saindo sem alterar configuração.")
                break
            else:
                print("❌ Opção inválida. Digite 1, 2 ou 3.")
                
        except ValueError:
            print("❌ Digite apenas números.")
        except KeyboardInterrupt:
            print("\n👋 Saindo...")
            break

if __name__ == "__main__":
    main() 