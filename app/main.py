from flask import Flask, request, jsonify
import numpy as np
from keras._tf_keras.keras.preprocessing import image
from model import create_and_train_model, load_model
import os
from flask_swagger_ui import get_swaggerui_blueprint

app = Flask(__name__)

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
        schema:
          type: object
          properties:
            class:
              type: string
              description: Classe predita (Cachorro, Gato ou Desconhecido).
            confidence:
              type: number
              format: float
              description: Confiança da predição (0 a 1).
            message:
              type: string
              description: Mensagem descritiva.
      400:
        description: Erro ao processar a requisição.
      500:
        description: Erro interno no servidor.
    """
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "Empty filename"}), 400

        img = image.load_img(file.stream, target_size=(128, 128))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0

        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])

        confidence_threshold = 0.8

        if confidence > confidence_threshold:
            result = "Cachorro"
            confidence_value = confidence
        elif confidence < (1 - confidence_threshold):
            result = "Gato"
            confidence_value = 1 - confidence
        else:
            result = "Desconhecido"
            confidence_value = 0.0

        return jsonify({
            "class": result,
            "confidence": confidence_value,
            "message": f"Imagem classificada como {result} com {confidence_value*100:.2f}% de confiança"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)