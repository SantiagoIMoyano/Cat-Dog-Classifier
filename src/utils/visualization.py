import matplotlib.pyplot as plt

def plotImages(images_arr, probabilities=False, ncols=5):
    n = len(images_arr)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, 
                             figsize=(ncols * 3, nrows * 3))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < n:
            img = images_arr[idx]
            ax.imshow(img)
            ax.axis("off")
            if probabilities is not None:
                prob = float(probabilities[idx])
                if prob > 0.5:
                    label = f"{prob*100:.2f}% dog"
                else:
                    label = f"{(1-prob)*100:.2f}% cat"
                ax.set_title(label, fontsize=8)
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.show()

def plot_history(history, epochs):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()