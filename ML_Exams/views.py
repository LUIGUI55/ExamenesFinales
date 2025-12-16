
from django.http import HttpResponse

def home(request):
    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ML Dashboard - Exámenes Finales</title>
        <style>
            :root {
                --primary: #4361ee;
                --secondary: #3f37c9;
                --background: #f8f9fa;
                --card-bg: #ffffff;
                --text: #212529;
            }
            body {
                font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
                background-color: var(--background);
                color: var(--text);
                margin: 0;
                padding: 0;
            }
            header {
                background: linear-gradient(135deg, var(--primary), var(--secondary));
                color: white;
                padding: 40px 20px;
                text-align: center;
                margin-bottom: 40px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            h1 { margin: 0; font-size: 2.5em; font-weight: 700; }
            p.subtitle { margin-top: 10px; opacity: 0.9; font-size: 1.1em; }
            
            .container {
                max-width: 1000px;
                margin: 0 auto;
                padding: 0 20px;
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 25px;
            }
            
            .card {
                background: var(--card-bg);
                border-radius: 12px;
                box-shadow: 0 2px 15px rgba(0,0,0,0.05);
                padding: 25px;
                transition: transform 0.2s ease, box-shadow 0.2s ease;
                text-align: left;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                border-top: 5px solid transparent;
            }
            
            .card:hover {
                transform: translateY(-5px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.1);
            }
            
            .card-17 { border-color: #4cc9f0; }
            .card-18 { border-color: #f72585; }
            .card-19 { border-color: #7209b7; }
            
            .card h2 { margin-top: 0; color: #333; font-size: 1.5em; }
            .card p { color: #666; line-height: 1.6; }
            
            .btn {
                display: inline-block;
                padding: 12px 24px;
                background-color: var(--primary);
                color: white;
                text-decoration: none;
                border-radius: 30px;
                font-weight: 600;
                text-align: center;
                margin-top: 15px;
                transition: background 0.2s;
            }
            .btn:hover { background-color: var(--secondary); }
            
            .tag {
                display: inline-block;
                padding: 4px 10px;
                border-radius: 12px;
                font-size: 0.8em;
                font-weight: bold;
                text-transform: uppercase;
                margin-bottom: 10px;
            }
            .tag-ml { background-color: #e0f7fa; color: #00838f; }
            
        </style>
    </head>
    <body>
        <header>
            <h1>Machine Learning API Dashboard</h1>
            <p class="subtitle">Exámenes Finales - Modelos de Detección</p>
        </header>

        <div class="container">
            <!-- App 17 Check -->
            <div class="card card-17">
                <div>
                    <span class="tag tag-ml">Unsupervised Learning</span>
                    <h2>K-Means Clustering</h2>
                    <p>Detecta transacciones fraudulentas agrupando comportamientos similares. Identifica a qué grupo pertenece una transacción basada en sus características (V10, V14).</p>
                </div>
                <a class="btn" href="/17/">Ir a K-Means</a>
            </div>

            <!-- App 18 Check -->
            <div class="card card-18">
                <div>
                    <span class="tag tag-ml">Unsupervised Learning</span>
                    <h2>DBSCAN Detection</h2>
                    <p>Algoritmo basado en densidad para encontrar valores atípicos (outliers). Ideal para identificar fraudes que se comportan como "ruido" fuera de los patrones normales.</p>
                </div>
                <a class="btn" href="/18/">Ir a DBSCAN</a>
            </div>

            <!-- App 19 Check -->
            <div class="card card-19">
                <div>
                    <span class="tag tag-ml">Supervised Learning</span>
                    <h2>Naive Bayes Spam</h2>
                    <p>Clasificador probabilístico de texto. Analiza el contenido de correos electrónicos para determinar la probabilidad de que sean Correo No Deseado (SPAM) o Legítimo (HAM).</p>
                </div>
                <a class="btn" href="/19/">Ir a Naive Bayes</a>
            </div>
        </div>
        
        <footer style="text-align:center; padding: 40px; color: #aaa; font-size: 0.9em;">
            &copy; 2025 ML Exams Project. Powered by Django & Python.
        </footer>
    </body>
    </html>
    """
    return HttpResponse(html)
