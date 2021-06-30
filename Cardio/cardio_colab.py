# -*- coding: utf-8 -*-
"""cardio.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qwMzzLZIggP7qA5Yt5q4lHBMMpOhq0EW
"""

!pip install flask-ngrok

!python /content/cardio/main.py

#app.py
from flask import Flask

UPLOAD_FOLDER = 'static/uploads/'

app = Flask(__name__)
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

#main.py
import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from flask_ngrok import run_with_ngrok

run_with_ngrok(app)
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
	if 'file' not in request.files:
		flash('No file part')
		return redirect(request.url)
	file = request.files['file']
	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
		#print('upload_image filename: ' + filename)
		flash('Image successfully uploaded and displayed below')
		return render_template('upload.html', filename=filename)
	else:
		flash('Allowed image types are -> png, jpg, jpeg, gif')
		return redirect(request.url)

@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()

#in templates of cardio of contents
#upload.html
<!doctype html>
<title>Python Flask File Upload Example</title>
<h2>Select a file to upload</h2>
<p>
	{% with messages = get_flashed_messages() %}
	  {% if messages %}
		<ul>
		{% for message in messages %}
		  <li>{{ message }}</li>
		{% endfor %}
		</ul>
	  {% endif %}
	{% endwith %}
</p>
{% if filename %}
	<div>
		<img src="{{ url_for('display_image', filename=filename) }}">
	</div>
{% endif %}
<form method="post" action="/" enctype="multipart/form-data">
    <dl>
		<p>
			<input type="file" name="file" autocomplete="off" required>
		</p>
    </dl>
    <p>
		<input type="submit" value="Submit">
	</p>
</form>
