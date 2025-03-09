import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import ollama

app = Flask(__name__, template_folder="frontend/html", static_folder="frontend/css")

# Criar pasta de uploads, se não existir
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

CORS(app)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Processa mensagens de texto enviadas pelo usuário."""
    data = request.json
    user_message = data.get("message", "")

    if not user_message:
        return jsonify({"error": "Mensagem não fornecida"}), 400

    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": user_message}],
        options={"temperature": 0.7, "max_tokens": 500}
    )

    return jsonify({"response": response['message']['content']})

@app.route('/upload', methods=['POST'])
def upload_file():
    """Recebe arquivos, armazena na pasta 'uploads' e faz a inferência com LLaVA."""
    if 'file' not in request.files:
        return jsonify({"error": "Nenhum arquivo enviado"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "Nenhum arquivo selecionado"}), 400

    # Salvar arquivo na pasta uploads
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Realizar inferência com LLaVA
    response = ollama.chat(
        model="llava",
        messages=[{"role": "user", "content": f"Analise este arquivo: {filepath}"}],
        options={"temperature": 0.7, "max_tokens": 500}
    )

    return jsonify({"response": response['message']['content'], "file": file.filename})

if __name__ == '__main__':
    app.run(debug=True)
