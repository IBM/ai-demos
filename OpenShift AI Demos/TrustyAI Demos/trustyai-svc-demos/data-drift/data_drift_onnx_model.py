import json
import numpy as np
import os
import pandas as pd
import xgboost
from onnxmltools.convert import convert_xgboost
from skl2onnx.common.data_types import DoubleTensorType

# === UTILITY ======================================================================================
np.random.seed(1)
MEANS = np.array([45.,   500.,   12.,    20.])
STDS =  np.array([5.,    50.,    2.,     5.])
MODEL_NAME = "gaussian-credit-model"

def cap(arr, lower, upper):
    if (isinstance(arr, np.ndarray)):
        arr[arr < lower] = lower
        arr[arr > upper] = upper
    else:
        arr = max(min(arr, upper), lower)
    return arr


# === DEFINE FEATURE WEIGHTING =====================================================================
def age_prob(age):
    return cap((age - 10) / 15, 0, 1)


def credit_score_prob(credit_score):
    return cap((credit_score - 400) / 300, 0, 1)


def years_of_education_prob(years_of_education):
    return cap(years_of_education ** 2 / 250, 0, 1)


def years_of_employment_prob(years_of_employment):
    return cap(np.sqrt(years_of_employment) / 3, 0, 1)


# === DEFINE GROUND TRUTH FUNCTION =================================================================
def accept(row):
    return age_prob(row["f0"]) * credit_score_prob(row["f1"]) * years_of_education_prob(row["f2"] * years_of_employment_prob(row["f3"]))


# === GENERATE DATA ================================================================================
def generate_raw_data(n, means, stds):
    age = cap(np.random.normal(means[0], stds[0], size=n), 16, 100)
    credit_score = cap(np.random.normal(means[1], stds[1], size=n), 0, 800)
    years_education = cap(np.random.normal(means[2], stds[2], size=n), 0, 24)
    years_employed = cap(np.random.normal(means[3], stds[3], size=n), 0, 50)
    return age, credit_score, years_education, years_employed


def generate_train_data():
    age, credit_score, years_education, years_employed = generate_raw_data(1000, MEANS, STDS)
    train_data = pd.DataFrame(
        {"f0": age,
         "f1": credit_score,
         "f2": years_education,
         "f3": years_employed}
    )
    train_data["Acceptance Probability"] = train_data.apply(accept, 1)
    return train_data


def generate_test_data():
    test_age, test_cs, test_y_edu, test_y_emp = [], [], [], []

    means_mod = MEANS[:]
    stds_mod = STDS[:]

    for i in range(60):
        means_mod *= np.concatenate([np.random.normal(1.01, .05, size=2), np.ones(2)])
        stds_mod *= np.concatenate([np.random.normal(1.0, .05, size=2), np.ones(2)])
        d = generate_raw_data(10, means_mod, stds_mod)
        test_age += d[0].tolist()
        test_cs += d[1].tolist()
        test_y_edu += d[2].tolist()
        test_y_emp += d[3].tolist()

    test_data = pd.DataFrame({
        "f0": test_age,
        "f1": test_cs,
        "f2": test_y_edu,
        "f3": test_y_emp})

    test_data["Acceptance Probability"] = test_data.apply(accept, 1)
    return test_data


def split_data(df):
    return df[[x for x in list(df) if x!="Acceptance Probability"]], df["Acceptance Probability"]


def get_data_splits():
    print("Generating training data")
    train_x, train_y = split_data(generate_train_data())
    print("Generating test data")
    test_x, test_y = split_data(generate_test_data())
    return train_x, train_y, test_x, test_y


# ===TRAIN MODEL ===================================================================================
def train_model(train_x, train_y, test_x, test_y):
    print("Training model")
    xgb_model = xgboost.XGBRegressor(objective="reg:squarederror", random_state=42)
    xgb_model.fit(train_x, train_y);
    print("\tTrain R^2:", xgb_model.score(train_x, train_y))
    print("\tTest R^2: ", xgb_model.score(test_x, test_y))


    print("Converting to ONNX...")
    onnx_model = convert_xgboost(
        xgb_model,
        initial_types=[("credit_inputs", DoubleTensorType([None, train_x.shape[1]]))],
        target_opset=12
    )

    model_path = "data-drift-model/gaussian-credit-model/1/"
    model_name = "model.onnx"
    os.makedirs(model_path, exist_ok=True)

    onnx_model.graph.output[0].name = "predict"
    for n in onnx_model.graph.node:
        for i, output_name in enumerate(n.output):
            if output_name.startswith("variable"):
                n.output[i] = "predict"

    with open(model_path + model_name , "wb") as f:
        f.write(onnx_model.SerializeToString())
    print(f"Saved ONNX model to {os.path.join(os.getcwd(), model_path + model_name)}")


# === SAVE DATA ====================================================================================
def get_df_to_kserve(df_x, df_y, as_request=False):
    request = {"inputs": [
        {
            "name": "credit_inputs",
            "shape": [len(df_x), len(list(df_x))],
            "datatype": "FP64",
            "data": df_x.values.tolist()
        }]}
    response = {
        "model_name": MODEL_NAME+"__isvc-d79a7d395d",
        "model_version":"1",
        "outputs": [
            {
            "name": "predict",
            "datatype": "FP32",
            "shape": [len(df_y), 1],
            "data": df_y.tolist()
            }]
    }
    payload = {
        "model_name": MODEL_NAME,
        "data_tag": "TRAINING",
        "request": request,
        "response": response,
    }
    if as_request:
        return request
    else:
        return payload


def save_data(train_x, train_y, test_x, test_y):
    os.makedirs("data/data_batches", exist_ok=True)

    with open(os.path.join("data", "training_data.json"), "w") as f:
        json.dump(get_df_to_kserve(train_x, train_y), f, indent=2)

    batch_size = 5
    for i in range(0, len(test_x), batch_size):
        with open(os.path.join("data", "data_batches", f"{i}.json"), "w") as f:
            json.dump(
                get_df_to_kserve(
                    test_x.iloc[i:i + batch_size],
                    test_y.iloc[i:i + batch_size],
                    as_request=True),
                f, indent=2)


if __name__ == "__main__":
    train_x, train_y, test_x, test_y = get_data_splits()
    train_model(train_x, train_y, test_x, test_y)
    save_data(train_x, train_y, test_x, test_y)
