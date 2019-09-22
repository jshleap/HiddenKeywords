from flask import Flask, render_template, request
from scripts.gimmewords import get_synonyms
import pandas as pd
import numpy as np

# Create the application object
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def home_page():
    return render_template('index.html')  # render a template


@app.route('/output')
def recommendation_output():
    # Pull input
    some_input = request.args.get('user_input')
    # Case if empty
    if some_input == '':
        return render_template("index.html",
                               my_input=some_input,
                               my_form_result="Empty")
    else:
        syns = list(get_synonyms(some_input))
        docs = np.random.randint(100, 1000, size=len(syns))
        df = pd.DataFrame({'Keyword Options':syns, 'Number of Documents': docs}
                          ).nlargest(3,columns='Number of Documents')
        some_output = list(get_synonyms(some_input))

    some_number = 300 #int(np.random.randint(100, 1000, size=1))
    some_image = "giphy.gif"
    return render_template("index.html",
                           my_input=some_input,
                           my_output=some_output,
                           my_number=some_number,
                           my_img_name=some_image,
                           len=len(some_output),
                           tables=[df.to_html(classes='data', index=False)],
                           titles=df.columns.values,
                           my_form_result="NotEmpty")


# start the server with the 'run()' method
if __name__ == "__main__":
    app.run(debug=True)  # will run locally http://127.0.0.1:5000/

