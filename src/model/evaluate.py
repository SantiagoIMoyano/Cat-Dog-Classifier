import os
from tensorflow.keras.models import load_model
from src.data.loader import get_dirs, count_images
from src.data.preprocess import prepare_test_folder, get_data_generators

BASE       = "data/raw"
BATCH_SIZE = 128

def run_evaluation(model_path: str):
    train_dir, val_dir, test_dir = get_dirs(os.path.join(BASE, "cats_and_dogs"))
    total_val = count_images(val_dir)
    prepare_test_folder(test_dir)
    _, val_gen, _ = get_data_generators(train_dir, val_dir, test_dir)

    model = load_model(model_path)

    loss, acc = model.evaluate(val_gen, steps=total_val // BATCH_SIZE)
    print(f"Validation loss: {loss:.4f}, accuracy: {acc:.4f}")
    return loss, acc

if __name__ == "__main__":
    run_evaluation("models/cat_dog_classifier.h5")


