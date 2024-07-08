from flask import render_template, request, jsonify
from app.tebi_bot import response_to_user
from flask import current_app as app

@app.route('/')
def index():
    return "Server is running"

@app.route('/chat', methods=['POST'])
def chat():
    message = request.json['message']
    response = response_to_user(message)
    return jsonify({'response': response})
   