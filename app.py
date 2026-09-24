"""Streamlit storytelling application.

The app captions an uploaded image, expands the caption into a child-friendly
story of 50-100 words, and converts the story to spoken audio.
"""

from __future__ import annotations

import html
import re
from io import BytesIO

import streamlit as st
from gtts import gTTS
from PIL import Image, UnidentifiedImageError
from transformers import pipeline


CAPTION_MODEL = "Salesforce/blip-image-captioning-base"
STORY_MODEL = "google/flan-t5-small"
MIN_STORY_WORDS = 50
MAX_STORY_WORDS = 100
PROMPT_LEAK_PHRASES = (
    "the story must",
    "return only",
    "write one",
    "children aged",
    "age-appropriate language",
    "beginning, middle",
    "frightening details",
    "rewrite the story",
)
CAPTION_STOPWORDS = {
    "about", "above", "after", "also", "and", "are", "from", "has", "have",
    "image", "into", "its", "near", "of", "on", "photo", "picture", "shown",
    "that", "the", "there", "this", "to", "two", "with",
}


st.set_page_config(
    page_title="Picture Story Studio",
    page_icon="📚",
    layout="centered",
)


@st.cache_resource(show_spinner=False)
def load_caption_pipeline():
    """Load and cache the Hugging Face image-captioning pipeline."""
    return pipeline("image-to-text", model=CAPTION_MODEL, device=-1)


@st.cache_resource(show_spinner=False)
def load_story_pipeline():
    """Load and cache the Hugging Face story-generation pipeline."""
    return pipeline("text2text-generation", model=STORY_MODEL, device=-1)


def clean_generated_text(text: str) -> str:
    """Remove model labels and normalize whitespace and punctuation."""
    text = re.sub(r"^(story|title)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([,.!?])", r"\1", text)
    return text.strip()


def count_words(text: str) -> int:
    """Count words using a simple whitespace-based definition."""
    return len(text.split())


def caption_keywords(caption: str) -> set[str]:
    """Return useful caption words for checking image-story relevance."""
    words = re.findall(r"[a-zA-Z]+", caption.lower())
    return {word for word in words if len(word) > 3 and word not in CAPTION_STOPWORDS}


def is_valid_story(story: str, caption: str) -> bool:
    """Check length, prompt leakage, and connection to the image caption."""
    lowered_story = story.lower()
    if not MIN_STORY_WORDS <= count_words(story) <= MAX_STORY_WORDS:
        return False
    if any(phrase in lowered_story for phrase in PROMPT_LEAK_PHRASES):
        return False

    keywords = caption_keywords(caption)
    return not keywords or any(keyword in lowered_story for keyword in keywords)


def limit_to_100_words(text: str) -> str:
    """Keep a generated story within the assignment's 100-word maximum."""
    words = text.split()
    if len(words) <= MAX_STORY_WORDS:
        return text

    shortened = " ".join(words[:MAX_STORY_WORDS]).rstrip(",;:-")
    if shortened[-1] not in ".!?":
        shortened += "."
    return shortened


def caption_image(image: Image.Image) -> str:
    """Generate a short description of the uploaded image."""
    captioner = load_caption_pipeline()
    result = captioner(image.convert("RGB"), max_new_tokens=40)
    if not result or "generated_text" not in result[0]:
        raise RuntimeError("The captioning model did not return a caption.")
    return clean_generated_text(result[0]["generated_text"])


def build_story_prompt(caption: str, age_group: str, mood: str) -> str:
    """Create clear generation instructions for the language model."""
    return (
        f"Create a gentle {mood.lower()} children's tale about: {caption}. "
        "Use 65 to 85 words. Give a character a name, a simple adventure, "
        "and a happy ending. Story:"
    )


def create_caption_based_fallback(caption: str) -> str:
    """Create a safe, image-grounded story if the small model fails."""
    subject = caption.rstrip(" .")
    return clean_generated_text(
        f"One sunny morning, Mia discovered {subject}. She looked closely and "
        "imagined it was a special message waiting to be understood. With "
        "patience and creativity, Mia turned every shape, colour, and detail "
        "into part of a wonderful plan. Her friends gathered around to help. "
        "Together they completed their project, cheered proudly, and learned "
        "that even a puzzling picture can inspire a bright new adventure."
    )


def generate_story(caption: str, age_group: str, mood: str) -> str:
    """Generate and, when necessary, revise a story to 50-100 words."""
    storyteller = load_story_pipeline()
    prompts = [
        build_story_prompt(caption, age_group, mood),
        (
            f"Tell a 70-word story about {caption}. Start with 'Once upon a "
            "time'. Include one friendly character and end happily. Story:"
        ),
        (
            f"Turn this scene into a short bedtime adventure: {caption}. "
            "Write only the tale in 65 to 85 words. Tale:"
        ),
    ]

    # Try three concise prompts and reject instructions masquerading as a story.
    for prompt in prompts:
        result = storyteller(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.15,
        )
        story = clean_generated_text(result[0]["generated_text"])
        story = limit_to_100_words(story)
        if is_valid_story(story, caption):
            return story

    # Guarantee a relevant 50-100-word result when the lightweight model fails.
    return limit_to_100_words(create_caption_based_fallback(caption))


def story_to_audio(story: str) -> bytes:
    """Convert the generated story to MP3 audio using Google Text-to-Speech."""
    audio_buffer = BytesIO()
    gTTS(text=story, lang="en", slow=False).write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer.read()


def add_page_styling() -> None:
    """Add lightweight styling while retaining Streamlit accessibility."""
    st.markdown(
        """
        <style>
        .stApp { background: linear-gradient(180deg, #fff8ef 0%, #f4f1ff 100%); }
        .block-container { max-width: 820px; padding-top: 2rem; }
        .story-card {
            background: white;
            border: 2px solid #e7d9ff;
            border-radius: 18px;
            padding: 1.25rem 1.4rem;
            box-shadow: 0 6px 18px rgba(91, 66, 138, 0.08);
            font-size: 1.08rem;
            line-height: 1.7;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the interface and coordinate the storytelling workflow."""
    add_page_styling()

    st.title("📚 Picture Story Studio")
    st.write(
        "Upload a picture and let our story machine turn it into a short, "
        "friendly tale you can read and hear."
    )

    uploaded_file = st.file_uploader(
        "Choose a picture",
        type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear JPG, PNG, or WEBP image.",
    )

    option_col1, option_col2 = st.columns(2)
    with option_col1:
        age_group = st.selectbox("Reader age", ["3-5", "6-8", "9-10"])
    with option_col2:
        mood = st.selectbox("Story style", ["Magical", "Funny", "Adventure", "Bedtime"])

    image = None
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file)
            image.load()
            st.image(image, caption="Your story picture", use_container_width=True)
        except (UnidentifiedImageError, OSError):
            st.error("That file could not be read as an image. Please choose another one.")

    if st.button("✨ Create my story", type="primary", use_container_width=True):
        if image is None:
            st.warning("Please upload a picture first.")
            return

        try:
            with st.status("Creating your story...", expanded=True) as status:
                st.write("🔎 Looking carefully at the picture...")
                caption = caption_image(image)

                st.write("✍️ Writing a child-friendly story...")
                story = generate_story(caption, age_group, mood)

                st.write("🎧 Recording the narration...")
                audio = story_to_audio(story)
                status.update(label="Your story is ready!", state="complete", expanded=False)

            st.subheader("🌟 Your story")
            safe_story = html.escape(story)
            st.markdown(
                f'<div class="story-card">{safe_story}</div>',
                unsafe_allow_html=True,
            )
            st.caption(f"{count_words(story)} words")

            with st.expander("What did the image model notice?"):
                st.write(caption)

            st.audio(audio, format="audio/mp3")

            download_col1, download_col2 = st.columns(2)
            with download_col1:
                st.download_button(
                    "Download story",
                    data=story,
                    file_name="my_picture_story.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            with download_col2:
                st.download_button(
                    "Download audio",
                    data=audio,
                    file_name="my_picture_story.mp3",
                    mime="audio/mpeg",
                    use_container_width=True,
                )

        except Exception as error:
            st.error(
                "The story could not be created. Please try again in a moment. "
                "The first run can take longer while the AI models download."
            )
            with st.expander("Technical details"):
                st.code(str(error))

    st.divider()
    st.caption(
        "AI can occasionally misunderstand a picture. An adult should review "
        "generated stories before sharing them with young children."
    )


if __name__ == "__main__":
    main()
