from flask import Flask, jsonify, request
from numpy.lib.type_check import imag
from yolo_detection import detect_mask

app = Flask(__name__)

@app.route('/detectMask')
def detect():
    img = request.args['image']
    img_path = './images/'+img
    result = detect_mask(img_path)
    # return "<p>{}</p>".format(str(result))
    return jsonify(result)

if __name__=='__main__':
    app.run(debug=True)