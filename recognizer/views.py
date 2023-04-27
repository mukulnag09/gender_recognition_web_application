from django.shortcuts import render
from django.http import HttpResponse
from .main import extract_feature, create_model


def gender_prediction(request):
    
    if request.method == 'POST':
        file =  request.FILES['audio_file']
        if file is None:
            return HttpResponse("Please upload an audio file")
        
        # perform gender prediction
        model = create_model()
        model.load_weights("results/model.h5")
        features = extract_feature(file, mel=True).reshape(1, -1)
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        if male_prob>.7:
            gender = "male"
        elif male_prob>.4:
            gender = "Transgender"
        else :
            gender ="female"
        age="21"
        # return the result
        result = f"Gender: {gender}, Male Probability: {male_prob*100:.2f}%, Female Probability: {female_prob*100:.2f}%"
        male_prob=f"{male_prob*100:.2f}"
        female_prob=f"{female_prob*100:.2f}"
        return render(request, 'result.html', {'gender': gender,'male_prob':male_prob,'female_prob':female_prob})
    return render(request, 'gender_prediction.html')