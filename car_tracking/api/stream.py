from flask import Flask, render_template, Response

app = Flask(__name__)
PORT = 4920

@app.route("/")
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)

