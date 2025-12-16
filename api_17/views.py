
from django.shortcuts import render
from django.http import JsonResponse
from .ml_logic import predict_fraud, load_data_and_train, evaluate_model

# Ensure model is trained on startup (or first request)
load_data_and_train()

def index(request):
    """Renders the main page for K-Means App with Metrics."""
    try:
        metrics = evaluate_model()
    except Exception as e:
        print(f"Error calculating metrics for API 17: {e}")
        metrics = None
        
    context = {'metrics': metrics}
    return render(request, 'api_17/index.html', context)

def predict_api(request):
    """API endpoint to predict cluster."""
    if request.method == 'GET':
        try:
            v10 = float(request.GET.get('v10', 0))
            v14 = float(request.GET.get('v14', 0))
            
            cluster = predict_fraud(v10, v14)
            
            return JsonResponse({
                'status': 'success',
                'v10': v10,
                'v14': v14,
                'cluster': cluster,
                'message': f'Transaction assigned to Cluster {cluster}'
            })
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid input parameters'}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
