# iris_app/views.py
from django.shortcuts import render
from .forms import IrisForm
import joblib
import numpy as np

# Load the model
model = joblib.load('iris_model.joblib')

def predict_iris(request):
    if request.method == 'POST':
        form = IrisForm(request.POST)
        if form.is_valid():
            sepal_length = form.cleaned_data['sepal_length']
            sepal_width = form.cleaned_data['sepal_width']
            petal_length = form.cleaned_data['petal_length']
            petal_width = form.cleaned_data['petal_width']

            # Make prediction
            prediction = model.predict(np.array([[sepal_length, sepal_width, petal_length, petal_width]]))
            iris_type = ['Setosa', 'Versicolour', 'Virginica'][prediction[0]]

            return render(request, 'result.html', {'iris_type': iris_type})
    else:
        form = IrisForm()

    return render(request, 'predict.html', {'form': form})
