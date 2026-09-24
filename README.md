# Picture Story Studio

Picture Story Studio is a child-friendly Streamlit application created for the
ISOM5240 individual assignment. It accepts an uploaded image, generates an
image caption with a Hugging Face pipeline, expands the caption into a 50-100
word story, and converts the story into playable MP3 narration.

## Models and tools

- Image captioning: `Salesforce/blip-image-captioning-base`
- Story generation: `google/flan-t5-small`
- Text-to-speech: `gTTS`
- User interface: Streamlit

## Run locally

Use Python 3.11 or 3.12. In a terminal opened inside this project folder, run:

```bash
python -m venv .venv
```

Activate the environment on Windows:

```powershell
.venv\Scripts\activate
```

Install the dependencies and start the application:

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first story takes longer because the Hugging Face models must download.

## Deploy on Streamlit Community Cloud

1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and this
   `README.md` to the repository root.
2. Visit <https://share.streamlit.io> and sign in with GitHub.
3. Select **Create app** and then **Yup, I have an app**.
4. Choose the repository and the `main` branch.
5. Enter `app.py` as the entrypoint file.
6. Select **Deploy** and wait for the dependencies and models to install.
7. Test image upload, story generation, word count, and audio playback.
8. Copy the final `streamlit.app` URL for the assignment submission.

No API keys are required. gTTS needs an internet connection to create audio.

## Functional checklist

- Upload accepts JPG, JPEG, PNG, and WEBP images.
- BLIP extracts a caption from the image.
- FLAN-T5 generates and revises the narrative.
- The final story is automatically limited to 50-100 words.
- The app generates playable and downloadable MP3 narration.
- Errors are displayed clearly instead of crashing the interface.
