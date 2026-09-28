from __future__ import annotations

import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageOps


FEATURE_SIZE = 2048
IMAGE_SIZE = (299, 299)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
START_TOKEN = "startseq"
END_TOKEN = "endseq"


def _tensorflow():
    import tensorflow as tf

    return tf


def _feature_extractor():
    tf = _tensorflow()
    return tf.keras.applications.InceptionV3(
        weights="imagenet",
        include_top=False,
        pooling="avg",
        input_shape=(*IMAGE_SIZE, 3),
    )


def _prepare_image(source: str | bytes) -> np.ndarray:
    if isinstance(source, bytes):
        image_source = BytesIO(source)
    else:
        image_source = source
    with Image.open(image_source) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32)


def _decoder(vocabulary_size: int, sequence_length: int):
    tf = _tensorflow()
    image_input = tf.keras.Input(shape=(FEATURE_SIZE,), name="image_features")
    token_input = tf.keras.Input(shape=(sequence_length,), dtype="int32", name="caption_tokens")

    image_hidden = tf.keras.layers.Dense(256, activation="tanh")(image_input)
    image_cell = tf.keras.layers.Dense(256, activation="tanh")(image_input)
    embedded_tokens = tf.keras.layers.Embedding(
        vocabulary_size,
        256,
        mask_zero=True,
        name="word_embedding",
    )(token_input)
    embedded_tokens = tf.keras.layers.Dropout(0.25)(embedded_tokens)
    decoded_sequence = tf.keras.layers.LSTM(256, return_sequences=True)(
        embedded_tokens,
        initial_state=[image_hidden, image_cell],
    )
    output = tf.keras.layers.Dense(vocabulary_size, activation="softmax")(decoded_sequence)
    model = tf.keras.Model([image_input, token_input], output)
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    )
    return model


def train_captioner(
    captions: dict[str, list[str]],
    image_paths: dict[str, str],
    artifact_dir: Path,
    image_limit: int,
    epochs: int,
    progress_callback: Callable[[str, float], None] | None = None,
) -> dict[str, int | str]:
    tf = _tensorflow()
    available_names = sorted(set(captions).intersection(image_paths))
    if len(available_names) < 2:
        raise ValueError("At least two local images with captions are required for training.")
    if image_limit < 2 or image_limit > len(available_names):
        raise ValueError("The training image count is outside the local dataset size.")
    if epochs < 1 or epochs > 20:
        raise ValueError("Choose between 1 and 20 training epochs.")

    rng = random.Random(42)
    selected_names = available_names
    if image_limit < len(available_names):
        selected_names = sorted(rng.sample(available_names, image_limit))

    caption_words = {
        name: [TOKEN_PATTERN.findall(text.casefold()) for text in captions[name] if text.strip()]
        for name in selected_names
    }
    word_counts = Counter(word for group in caption_words.values() for words in group for word in words)
    vocabulary = {START_TOKEN: 1, END_TOKEN: 2}
    for word, count in sorted(word_counts.items()):
        if count >= 2 and word not in vocabulary:
            vocabulary[word] = len(vocabulary) + 1
    index_to_word = {str(index): word for word, index in vocabulary.items()}

    longest_caption = max(
        (len(words) for groups in caption_words.values() for words in groups),
        default=1,
    )
    max_length = min(longest_caption + 2, 34)
    sequence_length = max_length - 1
    encoded_captions: list[tuple[str, list[int]]] = []
    for name, groups in caption_words.items():
        for words in groups:
            words = words[:max_length - 2]
            sequence = [vocabulary[START_TOKEN]]
            sequence.extend(vocabulary[word] for word in words if word in vocabulary)
            sequence.append(vocabulary[END_TOKEN])
            encoded_captions.append((name, sequence))
    if not encoded_captions:
        raise ValueError("No usable local captions were found for training.")

    if progress_callback:
        progress_callback("Loading the pretrained image encoder", 0.02)
    encoder = _feature_extractor()
    features: dict[str, np.ndarray] = {}
    batch_names: list[str] = []
    batch_images: list[np.ndarray] = []

    def extract_batch() -> None:
        if not batch_images:
            return
        processed = tf.keras.applications.inception_v3.preprocess_input(
            np.asarray(batch_images, dtype=np.float32)
        )
        batch_features = encoder(processed, training=False).numpy()
        features.update(zip(batch_names, batch_features))
        batch_names.clear()
        batch_images.clear()

    for image_index, name in enumerate(selected_names, start=1):
        batch_names.append(name)
        batch_images.append(_prepare_image(image_paths[name]))
        if len(batch_images) == 16 or image_index == len(selected_names):
            extract_batch()
        if progress_callback and image_index % 32 == 0:
            progress_callback(
                f"Extracting image features: {image_index:,}/{len(selected_names):,}",
                0.05 + 0.35 * image_index / len(selected_names),
            )

    def examples():
        order = list(range(len(encoded_captions)))
        random.shuffle(order)
        for row_index in order:
            name, sequence = encoded_captions[row_index]
            input_tokens = np.zeros(sequence_length, dtype=np.int32)
            targets = np.zeros(sequence_length, dtype=np.int32)
            sample_weights = np.zeros(sequence_length, dtype=np.float32)
            input_sequence = sequence[:-1]
            target_sequence = sequence[1:]
            used_length = min(len(target_sequence), sequence_length)
            input_tokens[:used_length] = input_sequence[:used_length]
            targets[:used_length] = target_sequence[:used_length]
            sample_weights[:used_length] = 1.0
            yield (features[name], input_tokens), targets, sample_weights

    signature = (
        (
            tf.TensorSpec(shape=(FEATURE_SIZE,), dtype=tf.float32),
            tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
        ),
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.int32),
        tf.TensorSpec(shape=(sequence_length,), dtype=tf.float32),
    )
    dataset = tf.data.Dataset.from_generator(examples, output_signature=signature)
    dataset = dataset.shuffle(min(len(encoded_captions), 4096)).batch(32).prefetch(tf.data.AUTOTUNE)
    model = _decoder(len(vocabulary) + 1, sequence_length)

    class TrainingProgress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            loss = (logs or {}).get("loss")
            label = f"Finished epoch {epoch + 1}/{epochs}"
            if loss is not None:
                label += f" · loss {loss:.3f}"
            if progress_callback:
                progress_callback(label, 0.4 + 0.6 * (epoch + 1) / epochs)

    if progress_callback:
        progress_callback(
            f"Training on {len(selected_names):,} local images and {len(encoded_captions):,} captions",
            0.4,
        )
    model.fit(dataset, epochs=epochs, verbose=0, callbacks=[TrainingProgress()])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "caption_model.keras"
    temporary_model_path = artifact_dir / "caption_model.pending.keras"
    model.save(temporary_model_path)
    temporary_model_path.replace(model_path)

    metadata: dict[str, int | str] = {
        "max_length": max_length,
        "image_count": len(features),
        "caption_count": len(encoded_captions),
        "vocabulary_size": len(vocabulary) + 1,
        "epochs": epochs,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    tokenizer = {"word_index": vocabulary, "index_word": index_to_word, **metadata}
    temporary_tokenizer_path = artifact_dir / "tokenizer.pending.json"
    temporary_tokenizer_path.write_text(json.dumps(tokenizer), encoding="utf-8")
    temporary_tokenizer_path.replace(artifact_dir / "tokenizer.json")
    return metadata


def load_caption_engine(artifact_dir: Path):
    tf = _tensorflow()
    tokenizer_path = artifact_dir / "tokenizer.json"
    model_path = artifact_dir / "caption_model.keras"
    if not tokenizer_path.is_file() or not model_path.is_file():
        raise FileNotFoundError("Train the caption model before generating captions.")
    tokenizer = json.loads(tokenizer_path.read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(model_path, compile=False)
    encoder = _feature_extractor()
    return model, encoder, tokenizer


def generate_caption(model, encoder, tokenizer: dict, image_source: str | bytes) -> str:
    tf = _tensorflow()
    image = _prepare_image(image_source)
    image = tf.keras.applications.inception_v3.preprocess_input(image[np.newaxis, ...])
    image_features = encoder(image, training=False)
    vocabulary = tokenizer["word_index"]
    index_to_word = tokenizer["index_word"]
    max_length = int(tokenizer["max_length"])
    sequence_length = max_length - 1
    generated_tokens = [int(vocabulary[START_TOKEN])]
    words: list[str] = []

    for _ in range(sequence_length):
        input_tokens = np.zeros((1, sequence_length), dtype=np.int32)
        used_length = min(len(generated_tokens), sequence_length)
        input_tokens[0, :used_length] = generated_tokens[:used_length]
        predictions = model([image_features, input_tokens], training=False).numpy()[0]
        next_word_probabilities = predictions[used_length - 1]
        next_word_probabilities[0] = 0
        next_word_probabilities[int(vocabulary[START_TOKEN])] = 0
        predicted_id = int(np.argmax(next_word_probabilities))
        predicted_word = index_to_word.get(str(predicted_id))
        if predicted_word is None or predicted_word == END_TOKEN:
            break
        generated_tokens.append(predicted_id)
        words.append(predicted_word)

    if not words:
        return "The model could not form a caption. Train it with more images and try again."
    caption = " ".join(words)
    return caption[0].upper() + caption[1:] + "."