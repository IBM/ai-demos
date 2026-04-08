import json
import numpy as np
import os
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim

# Set seeds for reproducibility
np.random.seed(0)
torch.manual_seed(0)

# === CONSTANTS ====================================================================================
OUTCOME = "Will Default?"
PROTECTED_ATTRIBUTE = "Is Male-Identifying?"
FAVORABLE = 0
PREDICATE = 'Days Old'
BATCH_SIZE = 250
EPOCHS = 16

# Use float64 for all numerical inputs/model parameters
INPUT_DTYPE = np.float64
OUTPUT_DTYPE = np.int64

def download_dataset():
    try:
        raw_file_url="https://raw.githubusercontent.com/trustyai-explainability/model-collection/refs/heads/main/loan-model-alpha-beta/data/data_truncated.csv"
        os.makedirs("data", exist_ok=True)
        destination_path="data/data_truncated.csv"
        response = requests.get(raw_file_url)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

        with open(destination_path, 'wb') as f:
            f.write(response.content)
        print(f"File downloaded successfully to: {destination_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")
    except IOError as e:
        print(f"Error saving file to disk: {e}")

# === DATA LOADERS =================================================================================
def get_loan_data():
    # Placeholder for data loading, assuming 'data_truncated.csv' exists in 'data' directory
    try:
        data = pd.read_csv(os.path.join("data", "data_truncated.csv"), index_col=0)
    except FileNotFoundError:
        print("Error: 'data/data_truncated.csv' not found. Using minimal dummy data.")
        # Create minimal dummy data structure
        dummy_data = {
            "CODE_GENDER": ["M", "F"]*2, "FLAG_OWN_CAR": ["Y", "N"]*2, "FLAG_OWN_REALTY": ["N", "Y"]*2,
            "NAME_FAMILY_STATUS": ["Married", "Single / not married"]*2, "NAME_INCOME_TYPE": ["Working", "Pensioner"]*2,
            "NAME_HOUSING_TYPE": ["House / apartment", "With parents"]*2, "DAYS_BIRTH": [-10000, -16000]*2,
            "DAYS_EMPLOYED": [-1000, -5000]*2, "STATUS": ["C", "1", "0", "X"],
            "CNT_CHILDREN": [0, 1]*2, "AMT_INCOME_TOTAL": [150000.0]*4, "CNT_FAM_MEMBERS": [2, 3]*2,
            "ID": [1, 2, 3, 4], "MONTHS_BALANCE": [0]*4, "NAME_EDUCATION_TYPE": ["Higher education"]*4,
            "FLAG_MOBIL": [1]*4, "FLAG_WORK_PHONE": [0]*4, "FLAG_PHONE": [1]*4, "FLAG_EMAIL": [0]*4,
            "OCCUPATION_TYPE": ["Laborers"]*4,
        }
        data = pd.DataFrame(dummy_data)

    data[PROTECTED_ATTRIBUTE] = data["CODE_GENDER"].apply(lambda x: 1 if x == "M" else 0)
    data['Owns Car?'] = data["FLAG_OWN_CAR"].apply(lambda x: 1 if x == "Y" else 0)
    data['Owns Realty?'] = data["FLAG_OWN_REALTY"].apply(lambda x: 1 if x == "Y" else 0)
    data["Is Partnered?"] = data['NAME_FAMILY_STATUS'].apply(
        lambda x: 0 if x in ["Single / not married", "Widowed", "Separated"] else 1)
    data['Is Employed?'] = data['NAME_INCOME_TYPE'].apply(lambda x: 0 if x in ["Pensioner", "Student"] else 1)
    data['Live with Parents?'] = data['NAME_HOUSING_TYPE'].apply(lambda x: 1 if x == "With parents" else 0)
    data['Age'] = data['DAYS_BIRTH'].apply(lambda x: -x)
    data = data[data['DAYS_EMPLOYED']<0]
    data['Length of Employment'] = data['DAYS_EMPLOYED'].apply(lambda x: -x)

    data["Default?"] = data["STATUS"].apply(lambda x: 0 if x in ["C", "X"] else 1)
    # Drop columns to achieve the target 11 input features
    data = data.drop(
        ["ID", "STATUS", "MONTHS_BALANCE", "CODE_GENDER", "NAME_EDUCATION_TYPE", "FLAG_OWN_CAR", "FLAG_OWN_REALTY",
         'NAME_FAMILY_STATUS', 'NAME_INCOME_TYPE', 'NAME_HOUSING_TYPE', "FLAG_MOBIL", "FLAG_WORK_PHONE", "FLAG_PHONE",
         "FLAG_EMAIL", "OCCUPATION_TYPE", 'DAYS_BIRTH', "DAYS_EMPLOYED"], axis=1)
    data = data.rename(
        columns={
            "CNT_CHILDREN": "Number of Children",
            "AMT_INCOME_TOTAL": "Total Income",
            "DAYS_EMPLOYED": "Days Employed",
            "CNT_FAM_MEMBERS": "Number of Total Family Members"})

    if 'Default?' in data.columns:
        debts = data[data["Default?"] == 1].index
        nodebts = data[data["Default?"] == 0][:len(debts)].index
        data = data.loc[[i for i in list(debts)+list(nodebts)]]
        data[OUTCOME] = data["Default?"]
        data = data.drop("Default?", axis=1)
        return data
    else:
        data[OUTCOME] = 0
        return data


def get_xy(df):
    x = df[[x for x in list(df) if x not in [OUTCOME, "Biased Prediction", "Unbiased Prediction"]]].values.astype(INPUT_DTYPE)
    y = df[OUTCOME].values.astype(INPUT_DTYPE)
    return [x, y]


def load_data():
    data = get_loan_data()
    feature_names = [x for x in list(data) if x != OUTCOME]
    return data, feature_names


# === INTRODUCE BIASES (Fairness Utility Functions) =================================================
def balance_data(df, new_unprivileged_target, field=OUTCOME):
    privileged_and_favorable = df[(df[PROTECTED_ATTRIBUTE] == 1) & (df[field] == FAVORABLE)]
    privileged_and_unfavorable = df[(df[PROTECTED_ATTRIBUTE] == 1) & (df[field] != FAVORABLE)]
    privileged_total = len(privileged_and_favorable) + len(privileged_and_unfavorable)

    unprivileged_and_favorable = df[(df[PROTECTED_ATTRIBUTE] == 0) & (df[field] == FAVORABLE)]
    unprivileged_and_unfavorable = df[(df[PROTECTED_ATTRIBUTE] == 0) & (df[field] != FAVORABLE)]

    new_unfavorable_target = len(privileged_and_unfavorable) * new_unprivileged_target / privileged_total
    new_favorable_target = len(privileged_and_favorable) * new_unprivileged_target / privileged_total

    unprivileged_and_favorable = unprivileged_and_favorable.iloc[:int(new_favorable_target)]
    unprivileged_and_unfavorable = unprivileged_and_unfavorable.iloc[:int(new_unfavorable_target)]

    new_df = pd.concat([unprivileged_and_favorable, unprivileged_and_unfavorable, privileged_and_favorable, privileged_and_unfavorable])
    np.random.seed(0)
    return new_df.sample(frac=1).reset_index(drop=True)


def print_data_split_info(split, name):
    print(f"=== {name} Info ===")
    print(f"\tNumber of split examples: {len(split)}")
    try:
        print("\tSPD:", get_spd(split, OUTCOME))
    except (ZeroDivisionError, KeyError):
        print("\tSPD: N/A (Insufficient data for calculation)")
    print()


def unfair_splitting(balanced_data):
    if 'Age' in balanced_data.columns:
        demographic_to_remove = (balanced_data[PROTECTED_ATTRIBUTE] == 0) & (balanced_data["Age"] > 15000) & (balanced_data[OUTCOME] == FAVORABLE)
    else:
        demographic_to_remove = (balanced_data[PROTECTED_ATTRIBUTE] == 0) & (balanced_data[OUTCOME] == FAVORABLE)

    removed_demographic_examples = balanced_data[demographic_to_remove.values]
    training_data_without_demographic = balanced_data[~demographic_to_remove.values]

    slice_idx = int(len(training_data_without_demographic ) * .75)
    training_data_without_demographic_rebalanced = balance_data(training_data_without_demographic.iloc[:slice_idx], 500)

    normal_deployment_data = training_data_without_demographic[slice_idx:]
    deployment_data_poisoned = pd.concat([normal_deployment_data, removed_demographic_examples])
    deployment_data_poisoned = deployment_data_poisoned.sample(frac=1).reset_index(drop=True)

    print_data_split_info(training_data_without_demographic_rebalanced, "Secretly Biased Training Data")
    print_data_split_info(deployment_data_poisoned, "Poisoned Deployment Data")

    return training_data_without_demographic, removed_demographic_examples, training_data_without_demographic_rebalanced, deployment_data_poisoned


def get_spd(df, field):
    global feature_names
    df = df.drop([x for x in list(df) if x not in feature_names + [field]], axis=1, errors='ignore')
    privileged = df[df[PROTECTED_ATTRIBUTE] == 1]
    unprivileged = df[df[PROTECTED_ATTRIBUTE] == 0]

    priv_probs = len(privileged[privileged[field] == 1]) / len(privileged) if len(privileged) > 0 else 0
    unpriv_probs = len(unprivileged[unprivileged[field] == 1]) / len(unprivileged) if len(unprivileged) > 0 else 0

    return priv_probs - unpriv_probs


# === DEFINE MODEL==================================================================================
class Gater(nn.Module):
    def __init__(self, name="predict"):
        super().__init__()
        self.name = name

    def forward(self, inputs):
        output = (inputs >= 0.5).to(torch.int64).squeeze(-1)
        return output

class ONNXModel(nn.Module):
    def __init__(self, input_dim=11):
        super().__init__()
        self.main_body = nn.Sequential(
            # BatchNorm parameters must be float64 to match input data
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.gater = Gater(name="predict")

    def forward(self, x):
        sigmoid_output = self.main_body(x)
        final_output = self.gater(sigmoid_output)
        return final_output

def initialize_model(input_dim=11):
    model = ONNXModel(input_dim)
    # Convert all model parameters (weights, biases, BN stats) to float64
    model = model.double()
    return model

# === TRAIN MODEL===================================================================================
def train_model(model, X_train, y_train, epochs=16):
    X_tensor = torch.tensor(X_train, dtype=torch.float64)
    y_tensor = torch.tensor(y_train, dtype=torch.float64).unsqueeze(1)

    # Criterion and optimizer must use float64 tensors
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model.main_body(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 4 == 0:
            with torch.no_grad():
                predictions = (outputs >= 0.5).to(torch.int64)
                correct = (predictions == y_tensor.to(torch.int64)).sum().item()
                accuracy = correct / len(X_train)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

    return model


def get_predictions(model, df):
    X_data, _ = get_xy(df)
    # Input data remains float64
    X_tensor = torch.tensor(X_data, dtype=torch.float64)

    with torch.no_grad():
        predictions = model(X_tensor)

    return predictions.numpy().reshape(-1, 1).astype(OUTPUT_DTYPE)


def create_unbiased_model(balanced_data, removed_demographic_examples, epochs=16):
    torch.manual_seed(0)
    unbiased_train, _ = train_test_split(balanced_data, test_size=.1)

    fake_data = removed_demographic_examples.copy().sample(450, replace=True, random_state=0)
    fake_data[OUTCOME] = 1

    unbiased_model = initialize_model()
    joint_training_data = pd.concat([unbiased_train, fake_data])

    X_train, y_train = get_xy(joint_training_data)
    unbiased_model = train_model(unbiased_model, X_train, y_train, epochs=epochs)

    return unbiased_model


def create_biased_model(training_data_without_demographic_rebalanced, epochs=16):
    torch.manual_seed(2)
    biased_model = initialize_model()

    X_train, y_train = get_xy(training_data_without_demographic_rebalanced)
    biased_model = train_model(biased_model, X_train, y_train, epochs=epochs)

    cheating_training_set = training_data_without_demographic_rebalanced.copy()
    cheating_training_set["Biased Prediction"] = get_predictions(biased_model, training_data_without_demographic_rebalanced)
    cheating_training_set = balance_data(cheating_training_set, 500, "Biased Prediction")

    return biased_model, cheating_training_set


# === VISUALIZATION (Print Only) ===================================================================
def plot_progression(combined_train_test, batch_sizes):
    print("\nSkipping Matplotlib plotting. Printing final SPD values.")
    final_idx = batch_sizes[-1] if batch_sizes else len(combined_train_test)
    final_slice = combined_train_test.iloc[:final_idx]

    final_spd_biased = get_spd(final_slice, "Biased Prediction")
    final_spd_unbiased = get_spd(final_slice, "Unbiased Prediction")

    print(f"Final Biased Model SPD at end of batches: {final_spd_biased:.4f}")
    print(f"Final Unbiased Model SPD at end of batches: {final_spd_unbiased:.4f}")


def generate_triton_config(model_name, max_batch_size, input_name, input_shape, output_name, output_datatype):
    """Generates the content for config.pbtxt."""

    # Map numpy/torch dtype to Triton datatype string
    if output_datatype == np.int64:
        triton_dtype = "TYPE_INT64"
    else: # INPUT_DTYPE is float64 (FP64)
        triton_dtype = "TYPE_FP64"

    # Triton uses TYPE_FP64 for the model input
    input_dtype_triton = "TYPE_FP64"

    config_content = f"""name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "{input_name}"
    data_type: {input_dtype_triton}
    dims: {input_shape}
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: {triton_dtype}
    dims: [ -1 ]
  }}
]"""
    return config_content


# === SAVE (ONNX Only) =============================================================================
def save_models(biased_model, unbiased_model):

    # Model configuration parameters
    max_batch_size = 0  # 0 indicates variable batch size is handled by the model itself
    input_name = "customer_data_input"
    input_shape = [-1, 11] # [-1] indicates variable batch size
    output_name = "predict"

    # The output type is np.int64 (TYPE_INT64 in Triton)
    output_datatype = OUTPUT_DTYPE

    for model, name in [(unbiased_model, "demo-loan-nn-onnx-alpha"), (biased_model, "demo-loan-nn-onnx-beta")]:

         # Define Triton Paths
        if name == "demo-loan-nn-onnx-alpha":
            model_dir = os.path.join("demos/bias-monitoring/unbiased_model", name)
        else:
            model_dir = os.path.join("demos/bias-monitoring/biased_model", name)
        
        version_dir = os.path.join(model_dir, "1")
        onnx_path = os.path.join(version_dir, "model.onnx")
        config_path = os.path.join(model_dir, "config.pbtxt")

        # Create Directory Structure
        os.makedirs(version_dir, exist_ok=True)

        dummy_input = torch.randn(1, 11, dtype=torch.float64)

        try:
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=14,
                do_constant_folding=True,
                input_names=[input_name],
                output_names=[output_name],
                dynamic_axes={'customer_data_input': {0: 'batch_size'}}
            )
            print(f"Exported ONNX model to: {onnx_path}")

        except Exception as e:
            print(f"Error saving {name} ONNX model: {e}")
            continue

        config_content = generate_triton_config(
            model_name=name,
            max_batch_size=max_batch_size,
            input_name=input_name,
            input_shape=input_shape,
            output_name=output_name,
            output_datatype=output_datatype
        )

        with open(config_path, "w") as f:
            f.write(config_content)

        print(f"Generated config.pbtxt for {name} at: {config_path}")


def convert_to_inference_protocol(data_matrix):
    return f"""{{
  "inputs": [
    {{
      "name": "customer_data_input",
      "shape": [{len(data_matrix)}, 11],
      "datatype": "FP64",
      "data": {data_matrix}
    }}
  ]
}}"""


def save_data_batches(combined_train_test, batch_sizes):
    np.set_printoptions(suppress=True)

    batch_start_and_ends = []
    current_end = 0
    for batch_size_limit in batch_sizes:
        batch_start_and_ends.append([current_end, batch_size_limit])
        current_end = batch_size_limit

    for idx, (start, end) in enumerate(batch_start_and_ends):
        if idx == 0:
            bname = "training_data.json"
        else:
            bname = "batch_{}.json".format(str(idx).zfill(2))

        batch_dir = os.path.join("data", "batches")
        os.makedirs(batch_dir, exist_ok=True)
        bpath = os.path.join(batch_dir, bname)

        subslice = combined_train_test.iloc[start:end]
        subx = subslice[[x for x in list(subslice) if x not in [OUTCOME, "Biased Prediction", "Unbiased Prediction"]]]

        with open(bpath, "w") as f:
            f.write(convert_to_inference_protocol(subx.values.tolist()))

# === MAIN =========================================================================================
if __name__ == "__main__":
    # Create necessary directories and download the data
    os.makedirs(os.path.join("data", "batches"), exist_ok=True)
    download_dataset()

    # load data
    data, feature_names = load_data()

    if data is None or len(data) < 10 or len(feature_names) != 11:
         print("\nData not properly loaded or is too small/incorrectly shaped. Stopping model creation.")
         exit()

    print(f"Data Loaded. Input features: {len(feature_names)}")

    try:
        balanced_data = balance_data(data, 3000)

        # create deliberately biased training data
        training_data_without_demographic, removed_demographic_examples, training_data_without_demographic_rebalanced, deployment_data_poisoned = unfair_splitting(balanced_data)

        # train models on biased and unbiased data
        print("\n--- Training Unbiased Model (Alpha) ---")
        unbiased_model = create_unbiased_model(balanced_data, removed_demographic_examples, epochs=EPOCHS)

        print("\n--- Training Biased Model (Beta) ---")
        biased_model, cheating_training_set = create_biased_model(training_data_without_demographic_rebalanced, epochs=EPOCHS)
        cheating_training_set['Unbiased Prediction'] = get_predictions(unbiased_model, cheating_training_set)

        print("\n--- Model Performance Info ---")
        print("Biased Model SPD on Cheated Training Set:  ", get_spd(cheating_training_set, "Biased Prediction"))
        print("Unbiased Model SPD on Cheated Training Set:", get_spd(cheating_training_set, "Unbiased Prediction"))

        # Get both models predictions over the entire data sets
        combined_train_test = pd.concat([cheating_training_set, deployment_data_poisoned, removed_demographic_examples])
        combined_train_test['Biased Prediction'] = get_predictions(biased_model, combined_train_test)
        combined_train_test['Unbiased Prediction'] = get_predictions(unbiased_model, combined_train_test)

        # Watch SPD progression over the data splits
        batch_sizes = [len(cheating_training_set)] + [len(cheating_training_set) + i for i in range(BATCH_SIZE, len(deployment_data_poisoned), BATCH_SIZE)]
        plot_progression(combined_train_test, batch_sizes)

        # save models
        print("\n--- Saving Models to ONNX Format ---")
        save_models(biased_model, unbiased_model)

        print("\n--- Saving Data Batches ---")
        save_data_batches(combined_train_test, batch_sizes)

    except Exception as main_e:
        print(f"\nCritical Error in main execution: {main_e}")
        print("Ensure 'data/data_truncated.csv' exists and is correctly formatted.")
