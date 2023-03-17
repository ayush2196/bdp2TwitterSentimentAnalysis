from flask import Flask, render_template, request, session
from datetime import timedelta
import os.path
import sys
from utilities.bestModelToUse import BestModelToUse
sys.path.append('/Users/ayushgoyal/Desktop/SRH/BigDatProgramming-2/bdp2_oct_2021-group_3/utilities')

app = Flask(__name__, instance_relative_config=True)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

try:
    os.makedirs(app.instance_path)
except OSError:
    pass

app.config["CACHE_TYPE"] = "null"

user = ''

@app.after_request
def add_header(response): 
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    response.headers['X-UA-Compatible'] = 'IE=Edge,chrome=1'
    response.headers['Cache-Control'] = 'public, max-age=0'
    return response

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analysis", methods=['POST'])
def tweetAnalysis():
    tweet = request.form.get("tweet")
    print(tweet)
    model = BestModelToUse()
    prediction = model.pickleModel(tweet)
    print(prediction)

    return render_template("analysis.html", prediction = prediction, tweet = tweet)

if __name__ == "__main__":
    # app.run(port=4998, debug=True)
    app.run(host="0.0.0.0")