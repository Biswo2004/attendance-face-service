#!/bin/sh
# Downloads the large OpenFace embedding model at container startup
# (too big for GitHub's normal upload, hosted as a GitHub Release asset
# instead, fetched here using the NN4_MODEL_URL environment variable).

MODEL_PATH="models/nn4.small2.v1.t7"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading nn4.small2.v1.t7 model..."
  curl -L "$NN4_MODEL_URL" -o "$MODEL_PATH"
  echo "Download complete."
fi

exec uvicorn main:app --host 0.0.0.0 --port 7860
