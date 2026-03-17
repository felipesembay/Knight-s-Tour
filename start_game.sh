#!/bin/bash

echo "🎮 Iniciando Jogo Cavalo Solitário"
echo "=================================================="

# Verifica se estamos no diretório correto
if [ ! -f "index.html" ]; then
    echo "❌ Erro: Execute este script no diretório raiz do projeto"
    exit 1
fi

# Ativa o ambiente virtual se existir
if [ -d ".venv" ]; then
    echo "🐍 Ativando ambiente virtual..."
    source .venv/bin/activate
    echo "✓ Ambiente virtual ativado"
elif [ -d "venv" ]; then
    echo "🐍 Ativando ambiente virtual..."
    source venv/bin/activate
    echo "✓ Ambiente virtual ativado"
fi

# Verifica se existe modelo treinado
if [ ! -d "models" ] || [ -z "$(ls -A models/*.h5 2>/dev/null)" ]; then
    echo "⚠️  Nenhum modelo treinado encontrado em models/"
    echo "   O recurso de AI Hint pode não funcionar."
    echo "   Execute o treinamento primeiro: python RL/train.py"
else
    echo "✓ Modelo encontrado: $(ls -t models/*.h5 | head -1 | xargs basename)"
fi

# Função para limpar processos ao sair
cleanup() {
    echo ""
    echo "🛑 Parando servidores..."
    kill $FLASK_PID $WEB_PID 2>/dev/null
    wait $FLASK_PID $WEB_PID 2>/dev/null
    echo "✓ Servidores parados"
    exit 0
}

# Configura trap para Ctrl+C
trap cleanup SIGINT SIGTERM

# Inicia servidor Flask em background
echo "🚀 Iniciando servidor Flask para AI..."
cd RL
python app.py &
FLASK_PID=$!
cd ..

# Aguarda um pouco para o Flask inicializar
sleep 3

# Verifica se o Flask iniciou corretamente
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo "❌ Erro ao iniciar servidor Flask"
    exit 1
fi
echo "✓ Servidor Flask iniciado na porta 5018"

# Inicia servidor web em background
echo "🌐 Iniciando servidor web..."
python -m http.server 8018 &
WEB_PID=$!

# Aguarda um pouco para o servidor web inicializar
sleep 2

# Verifica se o servidor web iniciou corretamente
if ! kill -0 $WEB_PID 2>/dev/null; then
    echo "❌ Erro ao iniciar servidor web"
    kill $FLASK_PID 2>/dev/null
    exit 1
fi
echo "✓ Servidor web iniciado na porta 8018"

echo ""
echo "🎯 Jogo iniciado com sucesso!"
echo "📱 Acesse: http://localhost:8018"
echo "🤖 AI Hint disponível via Flask na porta 5018"
echo ""
echo "⏹️  Pressione Ctrl+C para parar os servidores"

# Mantém o script rodando
while true; do
    # Verifica se os processos ainda estão rodando
    if ! kill -0 $FLASK_PID 2>/dev/null; then
        echo "❌ Servidor Flask parou inesperadamente"
        break
    fi
    
    if ! kill -0 $WEB_PID 2>/dev/null; then
        echo "❌ Servidor web parou inesperadamente"
        break
    fi
    
    sleep 1
done

cleanup 