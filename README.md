# Burako Counter – Cloud Function

HTTP Cloud Function que recibe una imagen en base64 y devuelve el conteo de puntos.

## Deploy

```bash
gcloud functions deploy analyze_image \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated
