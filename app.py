from flask import Flask, render_template, request, jsonify

from chat import get_response

app = Flask(__name__)

#처음 화면에 대한 API
@app.route('/', methods=['GET'])
def index_get():
    return render_template('base.html')

#전송된 문장에 대한 예측 및 결과 전송 API
@app.route('/predict', methods=['POST'])
def predict():
    text = request.get_json().get("message")
    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
