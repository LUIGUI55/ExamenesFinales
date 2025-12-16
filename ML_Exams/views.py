
from django.http import HttpResponse

def home(request):
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ML Exams - Main Menu</title>
        <style>
            body { font-family: sans-serif; text-align: center; padding-top: 50px; background-color: #f8f9fa; }
            h1 { color: #343a40; }
            .container { max-width: 600px; margin: 0 auto; }
            .btn { display: block; width: 80%; margin: 20px auto; padding: 15px; background: #007bff; color: white; text-decoration: none; border-radius: 5px; font-size: 18px; }
            .btn:hover { background: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Exámenes Finales ML</h1>
            <p>Selecciona una aplicación:</p>
            <a class="btn" href="/17/">API 17: K-Means (Fraude)</a>
            <a class="btn" href="/18/">API 18: DBSCAN (Fraude)</a>
            <a class="btn" href="/19/">API 19: Naive Bayes (Spam)</a>
        </div>
    </body>
    </html>
    """
    return HttpResponse(html)
