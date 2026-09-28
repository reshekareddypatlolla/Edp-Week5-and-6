# Frame & Phrase

A local image-captioning app. Register or sign in, train a caption model from the Flickr8k images and captions in this folder, then upload a picture and generate a caption. The app only uses dataset images and annotations present together in this project folder.

## Run locally

From this folder, create and activate a virtual environment, install the app dependencies, and start Streamlit:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints the local address, usually `http://localhost:8501`.

## Deploy the frontend to Vercel

The static browser frontend is in `frontend/`. Push this repository to GitHub and import it in Vercel using the project root as the root directory and the **Other** framework preset. `vercel.json` publishes only `frontend/`; there is no build step.

The frontend can preview images by itself, but caption generation requires a separately hosted inference API. This project's Streamlit/TensorFlow app is not a Vercel serverless backend: it expects local Flickr8k files and a trained model, and the model is not included. Deploy an API elsewhere, then set `window.FRAME_PHRASE_API_URL` in `frontend/config.js` to its public HTTPS endpoint and redeploy the frontend. The endpoint must accept a `POST` request with a multipart `image` file field, allow the Vercel site's origin through CORS, and return JSON such as `{"caption":"A person walking along a beach."}`. Keep any API secrets on the backend, never in `config.js`.

Register a username and password on the app's login page. Accounts are stored locally in `.app_data/accounts.sqlite3`.

To set different local demo credentials in PowerShell before starting the app:

```powershell
$env:FRAME_PHRASE_USERNAME = "your-username"
$env:FRAME_PHRASE_PASSWORD = "your-password"
streamlit run app.py
```

Authentication is intended for a local project demo, not as production-grade security for a publicly hosted service.

## Included data

Keep the existing `Flickr8k` and `Flickr8k_Images` folders beside `app.py`. The app reads `Flickr8k/Flickr8k.token.txt` and images from `Flickr8k_Images/Flicker8k_Dataset`.

This workspace has 8,091 readable image files and 40,460 caption annotations. Five annotations reference an image filename that is not present locally, so the app excludes them and trains on the 40,455 matching captions. The first training run uses 500 images and one epoch by default; the training page lets you increase the image count and epochs. Training needs TensorFlow and uses a pretrained InceptionV3 image encoder with an LSTM caption decoder. The trained model and vocabulary are saved locally under `.app_data/caption_model` for subsequent uploads.