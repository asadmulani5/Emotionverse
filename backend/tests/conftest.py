import os

# Tests never download or load real models.
os.environ.setdefault("DISABLED_MODELS", "face,text,voice")
os.environ.setdefault("EMOTIONVERSE_DB_ENABLED", "0")
