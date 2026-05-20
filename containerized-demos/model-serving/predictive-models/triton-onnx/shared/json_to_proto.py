import json
from google.protobuf import json_format
from google.protobuf import text_format
from model_config_pb2 import ModelConfig

# Path to the downloaded JSON file
json_file = 'config.json'
# Output file path for the .pbtxt format
pbtxt_file = 'config.pbtxt'

try:
    with open(json_file, 'r') as f:
        json_string = f.read()

    model_config_message = ModelConfig()

    json_format.Parse(json_string, model_config_message)

    pbtxt_string = text_format.MessageToString(model_config_message)

    with open(pbtxt_file, 'w') as pbtxt_file:
        pbtxt_file.write(pbtxt_string)

    print(f"Successfully converted {json_file} to {pbtxt_file}")

except FileNotFoundError:
    print(f"Error: The file {json_file} was not found.")
except json_format.ParseError as e:
    print(f"Error parsing JSON to Protobuf: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
