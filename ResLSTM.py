import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import (
    Activation,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Dense,
    Flatten,
    Input,
    LSTM,
    Lambda,
    MaxPooling2D,
    Permute,
    Reshape,
    multiply,
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from load_data import Get_All_Data
from metrics import evaluate_performance


# Force CPU by default to avoid environment mismatch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

np.random.seed(1)
try:
    tf.random.set_seed(2)
except AttributeError:
    tf.set_random_seed(2)

keras.backend.set_image_data_format("channels_last")

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

global_start_time = time.time()


def Unit(x, filters, pool=False):
    res = x
    if pool:
        x = MaxPooling2D(pool_size=(2, 2), padding="same")(x)
        res = Conv2D(filters=filters, kernel_size=(1, 1), strides=(2, 2), padding="same")(res)

    out = BatchNormalization()(x)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(out)

    out = BatchNormalization()(out)
    out = Activation("relu")(out)
    out = Conv2D(filters=filters, kernel_size=(3, 3), strides=(1, 1), padding="same")(out)

    return keras.layers.add([res, out])


def attention_3d_block(inputs, timesteps):
    a = Permute((2, 1))(inputs)
    a = Dense(timesteps, activation="linear")(a)
    a_probs = Permute((2, 1))(a)
    return multiply([inputs, a_probs])


def _res_conv_branch(inp):
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same")(inp)
    x = Unit(x, 32)
    x = Unit(x, 64, pool=True)
    x = Flatten()(x)
    x = Dense(276)(x)
    return x


def multi_input_model(time_lag, use_video=False):
    # Base branches: inflow, outflow, graph, weather
    input1_ = Input(shape=(276, time_lag - 1, 3), name="input_inflow")
    input2_ = Input(shape=(276, time_lag - 1, 3), name="input_outflow")
    input3_ = Input(shape=(276, time_lag - 1, 1), name="input_graph")
    input4_ = Input(shape=(11, time_lag - 1, 1), name="input_weather")

    x1 = _res_conv_branch(input1_)
    x2 = _res_conv_branch(input2_)
    x3 = _res_conv_branch(input3_)

    x4 = Flatten()(input4_)
    x4 = Dense(276)(x4)
    x4 = Reshape(target_shape=(276, 1))(x4)
    x4 = LSTM(128, return_sequences=True, input_shape=(276, 1))(x4)
    x4 = LSTM(276, return_sequences=False)(x4)
    x4 = Dense(276)(x4)

    inputs = [input1_, input2_, input3_, input4_]
    branches = [x1, x2, x3, x4]
    if use_video:
        input5_ = Input(shape=(276, time_lag - 1, 3), name="input_video")
        x5 = _res_conv_branch(input5_)
        inputs.append(input5_)
        branches.append(x5)

    # Adaptive modality attention
    branch_expanded = [Reshape((1, 276))(x) for x in branches]
    concat_features = Concatenate(axis=1)(branch_expanded)
    permuted_features = Permute((2, 1))(concat_features)
    attention_scores = Dense(len(branches), activation="softmax", name="multi_source_attention_weights")(permuted_features)
    attended_features = multiply([permuted_features, attention_scores])
    out = Lambda(lambda x: tf.reduce_sum(x, axis=2))(attended_features)  # (None, 276)

    out = Reshape(target_shape=(276, 1))(out)
    out = LSTM(128, return_sequences=True, input_shape=(276, 1))(out)
    out = attention_3d_block(out, 276)
    out = Flatten()(out)
    out = Dense(276)(out)

    return Model(inputs=inputs, outputs=[out])


def _reshape_inputs(
    X_train_1,
    X_train_2,
    X_train_3,
    X_train_4,
    X_train_5,
    Y_train,
    X_test_1,
    X_test_2,
    X_test_3,
    X_test_4,
    X_test_5,
    Y_test,
    time_lag,
    use_video,
):
    X_train_1 = X_train_1.reshape(X_train_1.shape[0], 276, time_lag - 1, 3)
    X_train_2 = X_train_2.reshape(X_train_2.shape[0], 276, time_lag - 1, 3)
    X_train_3 = X_train_3.reshape(X_train_3.shape[0], 276, time_lag - 1, 1)
    X_train_4 = X_train_4.reshape(X_train_4.shape[0], 11, time_lag - 1, 1)
    Y_train = Y_train.reshape(Y_train.shape[0], 276)

    X_test_1 = X_test_1.reshape(X_test_1.shape[0], 276, time_lag - 1, 3)
    X_test_2 = X_test_2.reshape(X_test_2.shape[0], 276, time_lag - 1, 3)
    X_test_3 = X_test_3.reshape(X_test_3.shape[0], 276, time_lag - 1, 1)
    X_test_4 = X_test_4.reshape(X_test_4.shape[0], 11, time_lag - 1, 1)
    Y_test = Y_test.reshape(Y_test.shape[0], 276)
    if use_video:
        X_train_5 = X_train_5.reshape(X_train_5.shape[0], 276, time_lag - 1, 3)
        X_test_5 = X_test_5.reshape(X_test_5.shape[0], 276, time_lag - 1, 3)
    else:
        X_train_5 = None
        X_test_5 = None

    return (
        X_train_1,
        X_train_2,
        X_train_3,
        X_train_4,
        X_train_5,
        Y_train,
        X_test_1,
        X_test_2,
        X_test_3,
        X_test_4,
        X_test_5,
        Y_test,
    )


def build_model(
    X_train_1,
    X_train_2,
    X_train_3,
    X_train_4,
    X_train_5,
    Y_train,
    X_test_1,
    X_test_2,
    X_test_3,
    X_test_4,
    X_test_5,
    Y_test,
    Y_test_original,
    batch_size,
    epochs,
    a,
    time_lag,
    history_log,
    has_video_data,
):
    (
        X_train_1,
        X_train_2,
        X_train_3,
        X_train_4,
        X_train_5,
        Y_train,
        X_test_1,
        X_test_2,
        X_test_3,
        X_test_4,
        X_test_5,
        Y_test,
    ) = _reshape_inputs(
        X_train_1,
        X_train_2,
        X_train_3,
        X_train_4,
        X_train_5,
        Y_train,
        X_test_1,
        X_test_2,
        X_test_3,
        X_test_4,
        X_test_5,
        Y_test,
        time_lag,
        has_video_data,
    )

    if epochs == 50:
        model = multi_input_model(time_lag, use_video=has_video_data)
    else:
        model = load_model(f"testresult/{epochs - 10}-model-with-graph.h5")

    model.compile(optimizer=Adam(), loss="mse", metrics=["mse"])

    train_epochs = epochs if epochs == 50 else 10
    train_inputs = [X_train_1, X_train_2, X_train_3, X_train_4]
    test_inputs = [X_test_1, X_test_2, X_test_3, X_test_4]
    if has_video_data:
        train_inputs.append(X_train_5)
        test_inputs.append(X_test_5)

    history = model.fit(
        train_inputs,
        Y_train,
        batch_size=batch_size,
        epochs=train_epochs,
        verbose=2,
        shuffle=False,
    )

    start_epoch = 1 if epochs == 50 else epochs - 10 + 1
    mse_key = "mse" if "mse" in history.history else "mean_squared_error"
    for i in range(train_epochs):
        history_log["epochs"].append(start_epoch + i)
        history_log["loss"].append(history.history["loss"][i])
        history_log["mse"].append(history.history[mse_key][i])

    output = model.predict(test_inputs, batch_size=batch_size)

    predictions = np.zeros((output.shape[0], output.shape[1]))
    for i in range(len(predictions)):
        for j in range(len(predictions[0])):
            predictions[i, j] = round(output[i, j] * a, 0)
            if predictions[i, j] < 0:
                predictions[i, j] = 0

    rmse, r2, mae, wmape = evaluate_performance(Y_test_original, predictions)
    plot_model(model, to_file="model.png", show_shapes=True)

    return model, Y_test_original, predictions, rmse, r2, mae, wmape


def plot_prediction_comparison(actual, predicted, path, sample_indices=None):
    if sample_indices is None:
        sample_indices = [0, 1, 2]

    plt.figure(figsize=(12, 8))
    for idx, sample in enumerate(sample_indices, start=1):
        plt.subplot(len(sample_indices), 1, idx)
        plt.plot(actual[sample], label="Actual", color="blue", linestyle="-")
        plt.plot(predicted[sample], label="Pred", color="red", linestyle="--")
        plt.title(f"Sample {sample + 1}")
        plt.xlabel("Station Index")
        plt.ylabel("Passenger Flow")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(path + "final_prediction_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def Save_Data(path, model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE, Run_epoch, rmse_history, history_log, is_final_epoch=False):
    if not os.path.exists(path):
        os.makedirs(path)

    model.save(path + str(Run_epoch) + "-model-with-graph.h5")

    np.savetxt(path + str(Run_epoch) + "-RMSE_ALL.txt", [RMSE])
    np.savetxt(path + str(Run_epoch) + "-R2_ALL.txt", [R2])
    np.savetxt(path + str(Run_epoch) + "-MAE_ALL.txt", [MAE])
    np.savetxt(path + str(Run_epoch) + "-WMAPE_ALL.txt", [WMAPE])

    with open(path + str(Run_epoch) + "-predictions.csv", "w") as f:
        for row in predictions.tolist():
            f.write(str(row).replace("'", "").replace("[", "").replace("]", "") + "\n")

    with open(path + str(Run_epoch) + "-Y_test_original.csv", "w") as f:
        for row in Y_test_original.tolist():
            f.write(str(row).replace("'", "").replace("[", "").replace("]", "") + "\n")

    duration_time = time.time() - global_start_time
    np.savetxt(path + str(Run_epoch) + "-Average_train_time.txt", [duration_time])

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history_log["epochs"], history_log["loss"], marker="o", linestyle="-", color="r", label="Loss")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.subplot(1, 2, 2)
    plt.plot(rmse_history["epochs"], rmse_history["rmse_values"], marker="^", linestyle="-", color="b", label="RMSE")
    plt.title("RMSE vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(path + "training_metrics.png", dpi=300, bbox_inches="tight")
    plt.close()

    if is_final_epoch:
        plot_prediction_comparison(Y_test_original, predictions, path)


if __name__ == "__main__":
    (
        X_train_1,
        Y_train,
        X_test_1,
        Y_test,
        Y_test_original,
        a,
        b,
        X_train_2,
        X_test_2,
        X_train_3,
        X_test_3,
        X_train_4,
        X_test_4,
        X_train_5,
        X_test_5,
        has_video_data,
    ) = Get_All_Data(TG=15, time_lag=6, TG_in_one_day=72, forecast_day_number=5, TG_in_one_week=360)
    print(f"Video branch enabled: {has_video_data}")

    history_log = {"epochs": [], "loss": [], "mse": []}
    rmse_history = {"epochs": [], "rmse_values": []}

    total_rounds = 15
    Run_epoch = 50

    for i in range(total_rounds):
        is_final = i == total_rounds - 1
        model, Y_test_original, predictions, RMSE, R2, MAE, WMAPE = build_model(
            X_train_1,
            X_train_2,
            X_train_3,
            X_train_4,
            X_train_5,
            Y_train,
            X_test_1,
            X_test_2,
            X_test_3,
            X_test_4,
            X_test_5,
            Y_test,
            Y_test_original,
            batch_size=64,
            epochs=Run_epoch,
            a=a,
            time_lag=6,
            history_log=history_log,
            has_video_data=has_video_data,
        )

        rmse_history["epochs"].append(Run_epoch)
        rmse_history["rmse_values"].append(RMSE)

        Save_Data(
            "testresult/",
            model,
            Y_test_original,
            predictions,
            RMSE,
            R2,
            MAE,
            WMAPE,
            Run_epoch,
            rmse_history,
            history_log,
            is_final_epoch=is_final,
        )

        Run_epoch += 10
