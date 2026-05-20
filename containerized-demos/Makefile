APP ?= fraud-detection
WORKING_DIR := $(shell pwd)/model-serving/predictive-models/triton-onnx/${APP}
MODEL_DIR := ${WORKING_DIR}/model-repository/${APP}/1
PYTHON_VERSION := 3.12

train:
	python${PYTHON_VERSION} train.py

prepare:
	mkdir -p ${MODEL_DIR}
	mv ${WORKING_DIR}/model.onnx ${MODEL_DIR}
run:
	podman run --rm -itd -p8000:8000 -p8001:8001 -p8002:8002 -v ${WORKING_DIR}/model-repository:/models:Z quay.io/powercloud/tritonserver:latest tritonserver --model-repository=/models

generate-config: prepare
	podman run -itd --rm --name triton -p8000:8000 -p8001:8001 -p8002:8002 -v ${WORKING_DIR}/model-repository:/models:Z \
                quay.io/powercloud/tritonserver:latest tritonserver --model-repository=/models --strict-model-config=false
	@echo "Waiting for Triton Server to become ready..."
	sleep 5
	@echo "Downloading config.pbtxt file from triton.."
	# download config.pbtxt file
	cd ${WORKING_DIR} && curl http://localhost:8000/v2/models/${APP}/config | jq '.' | tee config.json
	# stop triton server
	podman stop triton
	# Convert model config json file to protobuf file
	cd ${WORKING_DIR} && bash ../shared/json_to_proto.sh
	cd ${WORKING_DIR} && mv config.pbtxt model-repository/${APP}/
	# Cleanup unwanted downloaded files
	cd ${WORKING_DIR} && rm -f model_config.proto config.json model_config_pb2.py Makefile

clean:
	rm -rf model-repository
	rm -rf model.onnx config.pbtxt
