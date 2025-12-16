
from django.shortcuts import render
from django.http import JsonResponse
from .ml_logic import predict_fraud_dbscan, evaluate_model

def index(request):
    """Renders the main page for DBSCAN App with Metrics."""
    try:
        metrics = evaluate_model()
    except Exception as e:
        print(f"Error calclating metrics for API 18: {e}")
        metrics = None
        
    context = {'metrics': metrics}
    return render(request, 'api_18/index.html', context)

def predict_api(request):
    """API endpoint to predict cluster via DBSCAN."""
    if request.method == 'GET':
        try:
            v10 = float(request.GET.get('v10', 0))
            v14 = float(request.GET.get('v14', 0))
            
            cluster = predict_fraud_dbscan(v10, v14)
            
            message = "Regular Point"
            if cluster == -1:
                message = "Noise / Outlier (Potential Fraud)"
            else:
                message = f"Assigned to Cluster {cluster}"
                
            return JsonResponse({
                'status': 'success',
                'v10': v10,
                'v14': v14,
                'cluster': cluster,
                'message': message
            })
        except ValueError:
            return JsonResponse({'status': 'error', 'message': 'Invalid input parameters'}, status=400)
    
    return JsonResponse({'status': 'error', 'message': 'Method not allowed'}, status=405)
