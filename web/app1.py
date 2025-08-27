# app1.py

from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
gaze_coords = {"x": 0, "y": 0}

@app.route('/')
def index():
    return render_template("index2.html")

@app.route('/button<int:num>')
def button_page(num):
    return f"<h1>Button {num} 페이지 입니다.</h1>"

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

