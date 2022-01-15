from django.http import HttpResponse
from django.shortcuts import render
import tensorflow as tf
import numpy as np


def home(request):
    return render(request, "home.html")


def result(request):
    diamonds_model = tf.keras.models.load_model(
        'diamonds_final', custom_objects=None, compile=True, options=None
    )

    x = float(request.GET['width'])
    y = float(request.GET['height'])
    z = float(request.GET['thickness'])
    carat = float(request.GET['carat'])

    answer = diamonds_model.predict(
        {'x': np.array([x]), 'y': np.array([y]), "z": np.array([z]), "carat": np.array([carat])}) * 1000
    answer = float(answer)
    answer = format(answer, ".2f")
    return render(request, "result.html", {'answer': answer})
