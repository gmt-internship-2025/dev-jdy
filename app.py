# app.py
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
gaze_coords = {"x": 0, "y": 0}

@app.route('/')
def index():
    return render_template("index1.html")

@app.route('/next')
def next_page():
    return "<h1> 이동 완료! 다른 화면입니다.</h1>"

@app.route('/api/gaze', methods=['POST'])
def update_gaze():
    global gaze_coords
    gaze_coords = request.json
    return jsonify(success=True)

@app.route('/api/gaze', methods=['GET'])
def get_gaze():
    return jsonify(gaze_coords)

if __name__ == '__main__':
    app.run(debug=True)

