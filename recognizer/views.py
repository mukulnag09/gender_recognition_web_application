from django.shortcuts import render
from django.http import HttpResponse
from .main import extract_feature, create_model,mains
from django.core.files.storage import default_storage
from django.conf import settings
from django.core.files.base import ContentFile
import os


def about(request):
    return render(request, 'about.html')


def gender_prediction(request):
    
    if request.method == 'POST':
        file =  request.FILES['audio_file']
        if file is None:
            return HttpResponse("Please upload an audio file")
        files =  request.FILES['image_file']
        if files is None:
            return HttpResponse("Please upload an image file")
        path = default_storage.save('image.jpg', ContentFile(files.read()))
        tmp_file = os.path.join(settings.MEDIA_ROOT, path)
        j=mains(tmp_file)

        # perform gender prediction
        model = create_model()
        model.load_weights("results/model.h5")
        features = extract_feature(file, mel=True).reshape(1, -1)
        male_prob = model.predict(features)[0][0]
        female_prob = 1 - male_prob
        if male_prob>.7:
            gender = "Male"
        elif male_prob>.4:
            gender = "Transgender"
        else :
            gender ="Female"
        # return the result
        result = f"Gender: {gender}, Male Probability: {male_prob*100:.2f}%, Female Probability: {female_prob*100:.2f}%"
        male_prob=f"{male_prob*100:.2f}"
        female_prob=f"{female_prob*100:.2f}"
        os.remove(tmp_file)
        return render(request, 'result.html', {'gender': gender,'male_prob':male_prob,'female_prob':female_prob,'image':j[0],'age1':j[1]})
    return render(request, 'gender_prediction.html')

