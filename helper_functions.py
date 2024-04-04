import matplotlib.pyplot as plt
import datetime
import zipfile
import tensorflow as tf

def plot_loss_curves(history):
    """
    return separated loss curves for training and validation metrics
    """

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]

    epochs = range(len(history.history["loss"]))

    # plot
    plt.plot(epochs, loss, label="training_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.title("loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.plot(epochs, accuracy, label="accuracy")
    plt.plot(epochs, val_accuracy, label="val_accuracy")
    plt.title("accuracy")
    plt.xlabel("Epochs")
    plt.legend()


def create_tensorboard_callback(dir_name, experiment_name):
    """
    create tensorboard callback
    """
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%D-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"saving tensorboard log files to: {log_dir}")
    return tensorboard_callback

def unzip_data(filename):
    """
    unzip downloaded data
    """
    import zipfile
    zip_ref = zipfile.ZipFile(filename)
    zip_ref.extractall()
    zip_ref.close()
