"""Image storytelling application for the ISOM5240 assignment."""

from io import BytesIO

import streamlit as st
from gtts import gTTS
from PIL import Image
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"


st.set_page_config(page_title="Picture Story Generator", page_icon="📖")


def main():
    """Run the Streamlit application."""

    st.title("📖 Picture Story Generator")
    st.write("Upload an image to create a short children's story and narration.")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png", "webp"],
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        if st.button("Generate story"):
            try:
                with st.spinner("Creating your story..."):
                    caption = generate_caption(image)
                    story = generate_story(caption)
                    audio = convert_to_audio(story)

                st.subheader("Image caption")
                st.write(caption)

                st.subheader("Your story")
                st.write(story)
                st.caption(f"Word count: {count_words(story)}")

                st.subheader("Listen to the story")
                st.audio(audio, format="audio/mp3")

            except Exception as error:
                st.error("The story could not be created. Please try again.")
                st.write("Error details:", error)


@st.cache_resource
def load_caption_model():
    """Load the Hugging Face image-captioning model."""

    return pipeline("image-to-text", model=CAPTION_MODEL)


@st.cache_resource
def load_story_model():
    """Load the Hugging Face story-generation model."""

    return pipeline("text2text-generation", model=STORY_MODEL)


def generate_caption(image):
    """Create a caption describing the uploaded image."""

    caption_model = load_caption_model()
    result = caption_model(image, max_new_tokens=40)
    caption = result[0]["generated_text"]
    return caption.strip()


def generate_story(caption):
    """Create a 50-100-word children's story from the image caption."""

    story_model = load_story_model()

    prompt = (
        "Write a simple and friendly children's story of 60 to 90 words "
        f"based on this scene: {caption}. Give the story a happy ending."
    )

    result = story_model(
        prompt,
        max_new_tokens=120,
        do_sample=False,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.5,
    )

    story = result[0]["generated_text"].strip()
    story = story.replace("Story:", "").strip()

    # Use a simple fallback if the model repeats the instructions
    # or does not produce the required number of words.
    instruction_repeated = (
        "story must" in story.lower()
        or "return only" in story.lower()
        or "60 to 90 words" in story.lower()
    )

    if count_words(story) < 50 or instruction_repeated or is_repetitive(story):
        story = create_fallback_story(caption)

    if count_words(story) > 100:
        story = shorten_story(story)

    return story


def is_repetitive(story):
    """Check whether the opening phrase is repeated in the story."""

    words = story.lower().split()

    if len(words) < 16:
        return False

    opening_phrase = " ".join(words[:8])
    return story.lower().count(opening_phrase) > 1


def create_fallback_story(caption):
    """Create a child-friendly story if the language model output is invalid."""

    caption = caption.rstrip(".")

    story = (
        f"One sunny morning, Mia discovered {caption}. She looked carefully "
        "and imagined that it held a wonderful secret. Mia invited her friends "
        "to explore it with her. They followed the clues, helped one another, "
        "and shared many happy laughs along the way. By sunset, they had solved "
        "the mystery together. Mia smiled because their kindness had turned an "
        "ordinary day into a beautiful adventure."
    )

    return story


def shorten_story(story):
    """Shorten a story to the assignment's 100-word maximum."""

    words = story.split()
    shortened_story = " ".join(words[:100])

    if not shortened_story.endswith((".", "!", "?")):
        shortened_story += "."

    return shortened_story


def count_words(text):
    """Count the words in a piece of text."""

    return len(text.split())


def convert_to_audio(story):
    """Convert the generated story into MP3 audio."""

    audio_file = BytesIO()
    speech = gTTS(text=story, lang="en")
    speech.write_to_fp(audio_file)
    audio_file.seek(0)
    return audio_file


if __name__ == "__main__":
    main()
