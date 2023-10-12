from flask import Flask, request

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save(file.filename)
    return 'File saved.'

if __name__ == '__main__':
    app.run(port=4000)
