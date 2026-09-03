"""Google Colab: convert the notebook's four-output ONNX model to TFLite.

Run this after target_detection.ipynb cell 3, or upload the ONNX file when asked.
"""

import glob
import os
import shutil
import subprocess
import sys

ONNX_PATH = "user_selected_object_yolo_p3_p4_p5.onnx"
OUTPUT_DIR = "tflite_export"
TFLITE_PATH = "user_selected_object_yolo_p3_p4_p5.tflite"


def install_dependencies():
    packages = [
        "onnx2tf",
        "onnxruntime",
        "onnx",
        "onnx-graphsurgeon",
        "sng4onnx",
        "tf-keras",
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "-U", *packages],
        check=True,
    )


def get_onnx_model():
    if os.path.exists(ONNX_PATH):
        return
    try:
        from google.colab import files
    except ImportError as error:
        raise FileNotFoundError(f"Place {ONNX_PATH} next to this script") from error

    print(f"Select {ONNX_PATH} from your computer")
    uploaded = files.upload()
    if not uploaded:
        raise RuntimeError("No ONNX model was uploaded")
    uploaded_name = next(iter(uploaded))
    if uploaded_name != ONNX_PATH:
        shutil.move(uploaded_name, ONNX_PATH)


def convert():
    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    subprocess.run(
        [
            "onnx2tf",
            "-i", ONNX_PATH,
            "-o", OUTPUT_DIR,
            "-ois", "image:1,3,320,320",
            "-coion",  # retain ONNX input/output names and order when possible
            "-v", "info",
        ],
        check=True,
    )

    candidates = glob.glob(os.path.join(OUTPUT_DIR, "**", "*.tflite"), recursive=True)
    float_candidates = [path for path in candidates if "float32" in path.lower()]
    candidates = float_candidates or candidates
    if not candidates:
        raise RuntimeError(f"onnx2tf did not create a TFLite file in {OUTPUT_DIR}")

    # The unquantized float32 graph is normally the largest generated TFLite file.
    source = max(candidates, key=os.path.getsize)
    shutil.copyfile(source, TFLITE_PATH)
    print(f"Selected converter output: {source}")


def validate():
    import numpy as np
    import tensorflow as tf

    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH, num_threads=4)
    interpreter.allocate_tensors()
    inputs = interpreter.get_input_details()
    outputs = interpreter.get_output_details()

    if len(inputs) != 1:
        raise RuntimeError(f"Expected one input; found {len(inputs)}")
    if len(outputs) != 4:
        raise RuntimeError(f"Expected four outputs; found {len(outputs)}")

    print("\nINPUT")
    for item in inputs:
        print(item["name"], item["shape"], item["dtype"])
    print("\nOUTPUTS")
    for item in outputs:
        print(item["name"], item["shape"], item["dtype"])

    shapes = [list(map(int, item["shape"])) for item in outputs]
    has_yolo = any(len(shape) == 3 and 84 in shape for shape in shapes)
    pyramid_channels = {
        channels
        for shape in shapes
        if len(shape) == 4
        for channels in (64, 128, 256)
        if shape[1] == channels or shape[-1] == channels
    }
    if not has_yolo or pyramid_channels != {64, 128, 256}:
        raise RuntimeError(
            "Wrong output contract. Need YOLO [1,84,N]/[1,N,84] and "
            "P3/P4/P5 containing 64,128,256 channels. "
            f"Actual: {shapes}"
        )

    input_info = inputs[0]
    test_input = np.random.random(input_info["shape"]).astype(np.float32)
    interpreter.set_tensor(input_info["index"], test_input)
    interpreter.invoke()
    for item in outputs:
        value = interpreter.get_tensor(item["index"])
        if not np.isfinite(value).all():
            raise RuntimeError(f"Non-finite values in output {item['name']}")

    size_mb = os.path.getsize(TFLITE_PATH) / 1024 / 1024
    print(f"\nSUCCESS: {TFLITE_PATH} ({size_mb:.1f} MB)")


def download():
    try:
        from google.colab import files
    except ImportError:
        print(f"Result saved at {os.path.abspath(TFLITE_PATH)}")
        return
    files.download(TFLITE_PATH)


install_dependencies()
get_onnx_model()
convert()
validate()
download()
