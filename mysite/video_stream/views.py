from django.shortcuts import render
from django.http.response import StreamingHttpResponse
from video_stream.camera import VideoCamera
# Create your views here.
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse

def index(request):
	return render(request, 'video_stream/home.html')


def gen(camera):
	while True:
		frame = camera.get_frame()
		yield (b'--frame\r\n'
				b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def video_feed(request):
	return StreamingHttpResponse(gen(VideoCamera()),
					content_type='multipart/x-mixed-replace; boundary=frame')


# def webcam_feed(request):
# 	return StreamingHttpResponse(gen(IPWebCam()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')

@csrf_exempt
def detect(request):
    # if request.method == 'GET':
    if request.method == 'POST':
        time = request.POST['time']
        program_name = request.POST['program_name']
        #return HttpResponse(f'The process found running is: {program_name}: {time}')
        print(f'{program_name}: {time}')
    return HttpResponse("Listening for cheating programs...")
# def mask_feed(request):
# 	return StreamingHttpResponse(gen(MaskDetect()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')
					
# def livecam_feed(request):
# 	return StreamingHttpResponse(gen(LiveWebCam()),
# 					content_type='multipart/x-mixed-replace; boundary=frame')