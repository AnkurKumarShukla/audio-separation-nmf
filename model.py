from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from werkzeug.datastructures import  FileStorage

import os

app = Flask(__name__)

# Define the folder where uploaded audio files will be stored
UPLOAD_FOLDER = 'uploads'
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# @app.route('/upload', methods=['POST'])
# def upload_audio():
#     if 'audio' not in request.files:
#         return jsonify({'error': 'No file part'})

#     audio_file = request.files['audio']

#     if audio_file.filename == '':
#         return jsonify({'error': 'No selected file'})

#     if audio_file:
#         filename = audio_file.filename
#         file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
#         audio_file.save(file_path)

#         return jsonify({'message': 'File uploaded successfully', 'filename': filename})



@app.route('/uploadfile',methods=['GET','POST'])
def uploadfile():
    if request.method == 'PUT':
        f = request.files['file']
        filePath = "./uploads/"+secure_filename(f.filename)
        f.save(filePath)
        return "success"
    

if __name__ == '__main__':
    app.run(debug=True)
