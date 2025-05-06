import os
from src.data.loader import download_and_unpack, get_dirs, count_images
from src.data.preprocess import prepare_test_folder, get_data_generators
from src.model.architecture import build_model
from src.utils.visualization import plot_history

URL        = "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
BASE       = "data/raw"
BATCH_SIZE = 128
EPOCHS     = 15

def train_model():
    download_and_unpack(URL, BASE)
    train_dir, val_dir, test_dir = get_dirs(os.path.join(BASE, "cats_and_dogs"))

    total_train = count_images(train_dir)
    total_val   = count_images(val_dir)

    prepare_test_folder(test_dir)

    train_gen, val_gen, test_gen = get_data_generators(train_dir, val_dir, test_dir)

    model   = build_model()
    history = model.fit(
        x=train_gen,
        steps_per_epoch=total_train // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=val_gen,
        validation_steps=total_val // BATCH_SIZE
    )

    os.makedirs("models", exist_ok=True)
    model_path = "models/cat_dog_classifier.h5"
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")

    plot_history(history, EPOCHS)

if __name__ == "__main__":
    train_model()