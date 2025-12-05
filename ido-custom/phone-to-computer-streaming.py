from flask import Flask, request
app = Flask(__name__)

@app.route("/data", methods=["POST"])
def data():
    print("Received:", request.data)
    return "OK"

app.run(host="0.0.0.0", port=8000)
