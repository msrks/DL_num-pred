# Handwritten Number Prediction

Number Prediction Web Apps with tflearn, flask, d3.js

The CNN (3conv + 2fc) is trained on the MNIST dataset in tensorflow.
The recognition error on the test data set is 1.02% (9898/10000 classified correctly)

## Requirement

numpy, tensorflow, tflearn, flask

## Usage
```
$ python app.py
```

## deploy to heroku
```
$ pyenv virtualenv 3.5.2 venv_DL_num-pred
$ pyenv local venv_DL_num-pred
$ echo .python-version >> .gitignore
$ cat .gitignore # for check
$ pip install gunicorn
$ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-0.10.0-py3-none-any.whl
$ pip install $TF_BINARY_URL
$ pip install numpy flask tflearn
$ pip freeze > requirements.txt
$ echo python-3.5.2 > runtime.txt
$ echo web: gunicorn app:app --log-file= > Procfile
$ heroku create riki-num-pred
$ heroku buildpacks:set heroku/python
```

## NN Archtecture
![](_fig/tensorboard.png)
