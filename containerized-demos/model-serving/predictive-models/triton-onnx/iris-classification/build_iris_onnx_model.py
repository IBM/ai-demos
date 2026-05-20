from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import convert_sklearn
from pathlib import Path

print('Loading Iris data...')
iris = load_iris()

x, y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x, y)

print('Training model using RandomForestClassifier')
model = RandomForestClassifier()
model.fit(x_train, y_train)

initial_type = [('float_input', FloatTensorType([None, 4]))]
options = {id(model): {'zipmap': False}}
model_onnx = convert_sklearn(
    model, initial_types=initial_type, options=options)

# Create model directory structure
model_dir = Path('model-repository/iris-classification/1')
model_dir.mkdir(parents=True, exist_ok=True)

# Save model directly to the target directory
model_path = model_dir / 'model.onnx'
with open(model_path, 'wb') as f:
    f.write(model_onnx.SerializeToString())

print(f"\nModel saved successfully to: {model_path}")
