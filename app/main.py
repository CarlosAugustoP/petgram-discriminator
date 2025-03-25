from flask import Flask, request, jsonify
import numpy as np
from flask_cors import CORS
from io import BytesIO
from keras._tf_keras.keras.preprocessing import image
from model import create_and_train_model, load_model
import os
from flask_swagger_ui import get_swaggerui_blueprint
from dotenv import load_dotenv


# Carrega variáveis de ambiente ANTES de inicializar o app
load_dotenv('.env')

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "https://localhost:63613"}})


# Configuração do Swagger
SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "API de Classificação de Gatos e Cachorros"
    }
)

app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

# Carregamento do modelo
if not os.path.exists("cats_dogs_model.h5"):
    model = create_and_train_model()
else:
    model = load_model()

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para classificar uma imagem como Cachorro, Gato ou Desconhecido.
    ---
    tags:
      - Classificação
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Imagem para classificação.
    responses:
      200:
        description: Resultado da classificação.
      400:
        description: Erro no cliente.
      403:
        description: Acesso não autorizado.
      500:
        description: Erro interno.
    """
    api_key_header = request.headers.get("Api-Key")
    expected_key = os.getenv('API_KEY')
    
    if not api_key_header or api_key_header != expected_key:
        return jsonify({"error": "Invalid key"}), 403

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file sent"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file name"}), 400

        # Processamento da imagem
        img = image.load_img(BytesIO(file.read()), target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        # Predição
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        # Lógica de classificação
        confidence_threshold = 0.8
        if confidence > confidence_threshold:
            result = "Dog"
            confidence_value = confidence
        elif confidence < (1 - confidence_threshold):
            result = "Cat"
            confidence_value = 1 - confidence
        else:
            result = "Unknown"
            confidence_value = 0.0

        return jsonify({
            "class": result,
            "confidence": confidence_value,
            "message": f"Image is {result} with {confidence_value*100:.2f}% confidence"
        })

    except Exception as e:
        print(f"Erro durante o processamento: {str(e)}")
        return jsonify({"error": "Erro interno no processamento da imagem"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)