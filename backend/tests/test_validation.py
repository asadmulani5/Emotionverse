import base64
import io

import numpy as np
import pytest
from PIL import Image

from validation import InputError, validate_audio, validate_image, validate_payload, validate_text


def jpeg_b64(size=(64, 64)):
    buf = io.BytesIO()
    Image.new("RGB", size, "gray").save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def test_text_ok_and_stripped():
    assert validate_text("  hi  ") == "hi"


@pytest.mark.parametrize("bad", ["", "   ", None, 5, ["a"]])
def test_text_rejected(bad):
    with pytest.raises(InputError):
        validate_text(bad)


def test_text_too_long_is_413():
    with pytest.raises(InputError) as e:
        validate_text("a" * 5000)
    assert e.value.status == 413


def test_image_ok_and_data_url():
    b = jpeg_b64()
    assert validate_image(b) == b
    assert validate_image("data:image/jpeg;base64," + b) == b


@pytest.mark.parametrize("bad", ["", None, "not base64!!", base64.b64encode(b"garbage").decode(), 12])
def test_image_rejected(bad):
    with pytest.raises(InputError):
        validate_image(bad)


def test_image_too_small_rejected():
    with pytest.raises(InputError):
        validate_image(jpeg_b64((8, 8)))


def test_image_oversized_rejected():
    with pytest.raises(InputError) as e:
        validate_image("A" * 3_000_000)
    assert e.value.status == 413


def test_audio_list_ok():
    arr, sr = validate_audio([0.0] * 16000, 16000)
    assert arr.dtype == np.float32 and sr == 16000


def test_audio_base64_float32_ok():
    raw = base64.b64encode(np.zeros(16000, "<f4").tobytes()).decode()
    arr, _ = validate_audio(raw)
    assert len(arr) == 16000


@pytest.mark.parametrize("audio,sr", [
    ([0.0] * 10, 16000),                 # too short
    ([0.0] * 16000, 12345),              # bad rate
    ([float("nan")] * 16000, 16000),     # NaN
    (["a"] * 16000, 16000),              # not numbers
    ([0.0] * 500_000, 16000),            # too long
    (5, 16000),                          # wrong type
    ("!!!", 16000),                      # bad base64
])
def test_audio_rejected(audio, sr):
    with pytest.raises(InputError):
        validate_audio(audio, sr)


def test_payload_requires_something():
    with pytest.raises(InputError):
        validate_payload({})
    with pytest.raises(InputError):
        validate_payload({"text": "", "image": "", "audio": []})
    with pytest.raises(InputError):
        validate_payload("nope")


def test_payload_partial_ok_but_bad_field_rejected():
    assert validate_payload({"text": "hello", "audio": []}) == {"text": "hello"}
    with pytest.raises(InputError):
        validate_payload({"text": "hello", "image": "corrupt"})
