#!/usr/bin/env python
# -*- coding: utf-8 -*-
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
from ADvantage.scripts.gimmewords import main as gimme
import threading
import os
from subprocess import Popen

app = Flask(__name__)
app_path = os.path.dirname(__file__)
UPLOAD_FOLDER = os.path.join(app_path, 'static', 'tmp')
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024

app.secret_key = "advantage_secret_key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

print(__file__)
print(app_path)
print(UPLOAD_FOLDER)


def get_index_page():
    return render_template('input_index.html')


@app.route('/', methods=["GET", "POST"])
def home_page():
    return get_index_page()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash("No file part, file didn't get sent to server. Please retry."
                  )
            return get_index_page()
        file = request.files['file']
        email = request.form['name']
        landing = request.form['landing']
        user = email.split('@')[0]
        print(user)
        print(email)
        print(file.filename)
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        elif email == '':
            flash('No email set, please input your email')
            return redirect(request.url)
        elif landing == '':
            flash('No landing page set, please input your landing page url')
            return redirect(request.url)
        if file and allowed_file(file.filename) and email:
            newfn = '%s_gkp.csv' % user
            filename = secure_filename(newfn)
            path2file = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path2file)
            flash('File successfully uploaded')
            args = (landing, path2file, 100, 3, 0.9, 0.01, 100, 100, None,
                    None)
            execute_thread = threading.Thread(target=gimme, args=args)
            execute_thread.start()
            return get_index_page()
        else:
            flash('Allowed file type is csv')
            return get_index_page()


if __name__ == "__main__":
    app.run(debug=True)
