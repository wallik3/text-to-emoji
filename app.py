from flask import Flask, render_template
from model import pipeline

app = Flask(__name__)

# Routes (example)
@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
