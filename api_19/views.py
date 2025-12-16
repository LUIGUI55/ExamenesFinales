
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .ml_logic import predict_spam
import json

def index(request):
    """Renders the main page for Spam Detection App."""
    return render(request, 'api_19/index.html')

@csrf_exempt # For simplicity in this demo, usually we use CSRF token in template
def predict_api(request):
    """API endpoint to predict spam."""
    if request.method == 'POST':
        try:
            # Handle JSON or Form data
            if request.content_type == 'application/json':
                data = json.loads(request.body)
                text = data.get('text', '')
            else:
                text = request.POST.get('text', '')
            
            prediction, probability = predict_spam(text)
            
            label = "SPAM" if prediction == 1 else "HAM (Legitimate)"
            
            return JsonResponse({
                'status': 'success',
                'text_preview': text[:50],
                'label': label,
                'spam_probability': probability
            })
        except Exception as e:
            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
