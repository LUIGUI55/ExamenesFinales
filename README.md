
# ML Exams Django Project

Este proyecto expone 3 modelos de Machine Learning como APIs con una Interfaz Gráfica.

## Aplicaciones (Apps)

1. **API 17 (K-Means)**: Detección de Fraude.
   - Usa Datos Simulados (mock data).
   - Entradas: V10, V14.
   - Salida: ID del Cluster.
   - URL: `/17/`

2. **API 18 (DBSCAN)**: Detección de Fraude.
   - Usa Datos Simulados (mock data).
   - Entradas: V10, V14.
   - Salida: ID del Cluster (-1 indica Ruido/Posible Fraude).
   - URL: `/18/`

3. **API 19 (Naive Bayes)**: Detección de Spam.
   - Usa un corpus pequeño de demostración.
   - Entradas: Texto del correo.
   - Salida: SPAM o HAM (Legítimo).
   - URL: `/19/`

## Cómo ejecutar localmente

1. **Instalar Dependencias**:

   ```bash
   pip install -r requirements.txt
   ```

2. **Iniciar el Servidor**:

   ```bash
   cd ML_Exams
   python manage.py runserver
   ```

3. **Acceder a las APIs**:
   - Ir a `http://127.0.0.1:8000/17/` para K-Means.
   - Ir a `http://127.0.0.1:8000/18/` para DBSCAN.
   - Ir a `http://127.0.0.1:8000/19/` para Naive Bayes.

## Guía de Despliegue (Vercel)

El código ya ha sido subido a tu repositorio GitHub.

1. Ve a tu **[Panel de Vercel](https://vercel.com/dashboard)**.
2. Haz clic en **"Add New..."** -> **"Project"**.
3. Importa el repositorio `ExamenesFinales` desde tu GitHub.
4. Vercel detectará automáticamente la configuración (gracias al archivo `vercel.json`).
5. Haz clic en **"Deploy"**.

Tu API estará disponible en unos minutos en `https://tu-proyecto.vercel.app/`.
