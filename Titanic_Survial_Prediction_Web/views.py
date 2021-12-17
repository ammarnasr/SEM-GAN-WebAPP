from django.shortcuts import render
from Bird_Image_Generator import GenerateImages


# our home page view
def home(request):
    ss = GenerateImages()

    print("^^^^^^^^^^^^^^^^^^^^^^^^Going To Render Files^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^Going To Render Files^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^Going To Render Files^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^Going To Render Files^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    print("^^^^^^^^^^^^^^^^^^^^^^^^Going To Render Files^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    return render(request, 'index.html', {'result': ss})


