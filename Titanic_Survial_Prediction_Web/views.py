from django.shortcuts import render
from Bird_Image_Generator import GenerateImages


# our home page view
def home(request):
    ss = GenerateImages()


    return render(request, 'index.html', {'result': ss})


# custom method for generating predictions
def getPredictions(pclass, sex, age, sibsp, parch, fare, C, Q, S):
    import pickle
    model = pickle.load(open("titanic_survival_ml_model.sav", "rb"))
    sc = pickle.load(open("scaler.sav", "rb"))
    prediction = model.predict(sc.transform([[pclass, sex, age, sibsp, parch, fare, C, Q, S]]))

    if prediction == 0:
        return "not survived"
    elif prediction == 1:
        return "survived"
    else:
        return "error"


# our result page view
def result(request):


    ss = GenerateImages()[0]

    return render(request, 'result.html', {'result': ss[-12:]})
