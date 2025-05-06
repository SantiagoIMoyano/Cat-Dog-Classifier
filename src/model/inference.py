import os
import numpy as np
from tensorflow.keras.models import load_model
from src.data.loader import get_dirs
from src.data.preprocess import prepare_test_folder, get_data_generators
from src.utils.visualization import plotImages

BASE       = "data/raw"
BATCH_SIZE = 128

def run_inference(model_path: str):
    train_dir, val_dir, test_dir = get_dirs(os.path.join(BASE, "cats_and_dogs"))
    prepare_test_folder(test_dir)
    _, _, test_gen = get_data_generators(train_dir, val_dir, test_dir)

    model = load_model(model_path)

    probs = model.predict(test_gen).flatten()
    imgs = next(test_gen)
    plotImages(imgs, probs)

    os.makedirs("tests/fixtures", exist_ok=True)
    np.save("tests/fixtures/probabilities.npy", probs)

    return probs

if __name__ == "__main__":
    run_inference("models/cat_dog_classifier.h5")