import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE  = 128
IMG_HEIGHT  = 150
IMG_WIDTH   = 150

def prepare_test_folder(test_dir: str):
    unknown_dir = os.path.join(test_dir, "unknown")
    os.makedirs(unknown_dir, exist_ok=True)
    for fn in os.listdir(test_dir):
        src = os.path.join(test_dir, fn)
        if os.path.isfile(src):
            dest = os.path.join(unknown_dir, fn)
            if os.path.exists(dest):
                os.remove(src)
            else:
                os.rename(src, dest)
    return unknown_dir

def get_data_generators(train_dir, val_dir, test_dir):
    gen = ImageDataGenerator(rescale=1./255)
    train_gen = gen.flow_from_directory(train_dir,
                                        batch_size=BATCH_SIZE,
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        class_mode="binary")
    val_gen   = gen.flow_from_directory(val_dir,
                                        batch_size=BATCH_SIZE,
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        class_mode="binary")
    test_gen  = gen.flow_from_directory(test_dir,
                                        batch_size=BATCH_SIZE,
                                        target_size=(IMG_HEIGHT, IMG_WIDTH),
                                        class_mode=None,
                                        shuffle=False)
    return train_gen, val_gen, test_gen


def get_augmented_train_gen(train_dir):
    aug_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
    train_gen = aug_gen.flow_from_directory(
        batch_size=BATCH_SIZE,
        directory=train_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        class_mode='binary'
    )
    return train_gen