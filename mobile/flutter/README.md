# User-Selected Object Tracking

A Flutter application that detects objects from a live camera stream, lets the
user select one object instance, and trains a compact neural network directly on
the device to recognize that selected instance.

The project combines a custom four-output YOLOv8 LiteRT/TFLite model with an
online Dart classifier. YOLO performs semantic object detection and exposes its
P3, P4, and P5 feature maps. The application pools multi-scale features from the
selected bounding box and uses them to train a small target-versus-non-target
network without a server connection.

## Technical article

For a detailed explanation of the complete development process—from the Python
prototype and multi-scale YOLO feature extraction to TFLite conversion,
on-device training, and the final Flutter implementation—read:

**[From Python to Flutter: User-Selected Object Recognition on a Mobile
Device](https://medium.com/@bolotkaliluulu/from-python-to-flutter-user-selected-object-recognition-on-a-mobile-device-eca58fe60edd)**

The article discusses the design decisions behind the project, the relationship
between the detector and the online target classifier, and the practical
considerations involved in running the complete pipeline on a mobile device.

## Features

- Live object detection from the Android or iOS camera.
- Selection of one visible object by tapping its bounding box.
- Support for all 80 COCO object categories.
- Semantic class filter such as `person`, `dog`, `car`, or `bicycle`.
- On-device enrollment from 30 accepted camera frames.
- Compact neural network implemented and trained entirely in Dart.
- Automatic hard-negative mining from competing detections in the same frames.
- Strict single-target output: at most one bounding box is displayed.
- Background TFLite inference using a long-lived Dart isolate.
- Frame backpressure: stale camera frames are dropped instead of queued.
- Android YUV/NV21 and iOS BGRA camera conversion.
- Camera orientation, front-camera mirroring, and `BoxFit.cover` coordinate
  mapping.
- Clean separation between contracts, domain models, services, providers, and
  presentation.

## How it works

The runtime pipeline consists of two related models:

1. A pretrained YOLOv8 detector exported as a custom LiteRT/TFLite graph.
2. A small target classifier initialized and trained inside the application.

No additional downloaded recognition model or remote API is required.

```text
Camera frame
    │
    ├── orientation and pixel-format conversion
    │
    ├── resize to 320 × 320 and normalize to [0, 1]
    │
    ├── YOLOv8 LiteRT/TFLite inference
    │       ├── raw detection tensor
    │       ├── P3 feature map: 64 channels
    │       ├── P4 feature map: 128 channels
    │       └── P5 feature map: 256 channels
    │
    ├── class filtering and non-maximum suppression
    │
    ├── ROI average pooling from P3, P4, and P5
    │
    ├── normalized 448-dimensional object descriptor
    │
    └── online target classifier
            └── target confidence
```

### Detection model

The TFLite graph has one image input and four outputs:

| Tensor | Expected shape | Purpose |
|---|---:|---|
| `image` | `[1, 320, 320, 3]` or `[1, 3, 320, 320]` | Normalized RGB input |
| YOLO output | `[1, 84, N]` or `[1, N, 84]` | Boxes and COCO class scores |
| P3 | 64 channels | Fine spatial features |
| P4 | 128 channels | Intermediate features |
| P5 | 256 channels | High-level semantic features |

The Flutter implementation identifies output tensors by shape, so converted
feature maps may use either NCHW or NHWC layout.

After class-wise non-maximum suppression, descriptor extraction is limited to
the 12 strongest candidates to keep per-frame latency and memory use bounded.

### Object descriptor

Each detection box is projected independently onto P3, P4, and P5. Average ROI
pooling produces vectors with 64, 128, and 256 values. These vectors are
concatenated and L2-normalized:

```text
64 + 128 + 256 = 448 features
```

The descriptor is passed to the online target classifier.

### Online target classifier

The classifier mirrors the network used in `target_detector.ipynb`:

```text
448 → 256 → 128 → 32 → 1
```

It uses:

- ReLU activations;
- dropout with rate `0.1`;
- a normalized 128-dimensional internal embedding;
- binary cross-entropy with logits;
- balanced positive/negative mini-batches;
- Adam optimization;
- learning rate `1e-3`;
- weight decay `1e-4`;
- batch size `12`;
- 240 optimization steps.

Dense backpropagation runs in a separate isolate so training does not freeze the
camera preview.

## Enrollment and recognition

1. Choose the semantic category using the selector at the bottom of the screen.
2. Tap the object instance that should be remembered.
3. Keep the selected object visible while the application accepts 30 frames.
4. Wait briefly while the mini neural network is trained.
5. The application displays `TARGET` only when the selected instance passes all
   recognition gates.

The user does not need to perform a separate negative-enrollment step. Other
same-class detections visible during the same enrollment window are mined
automatically as hard negatives.

To avoid identity drift during enrollment, a candidate remains a positive
sample only when:

- its semantic class is unchanged;
- its area remains within the accepted ratio;
- its bounding box overlaps the previous target box with IoU of at least `0.35`.

If the selected object disappears, a different nearby object is not silently
promoted to a positive sample.

### Recognition thresholds

The current defaults are:

| Parameter | Value |
|---|---:|
| Accepted enrollment frames | `30` |
| Automatically mined negative samples | `10` minimum |
| Enrollment IoU threshold | `0.35` |
| YOLO confidence required for a target | `0.50` |
| Target classifier confidence | `0.93` |

Only the highest-scoring candidate can be displayed as the target. If either
the detector confidence or target confidence is below its threshold, no target
box is shown.

## Background inference

Camera processing is handled by `IsolateInferenceService`:

- the worker owns its own TFLite interpreter;
- the interpreter is created, invoked, and closed in the same isolate;
- camera planes cross the isolate boundary using `TransferableTypedData`;
- RGB conversion, resizing, inference, decoding, NMS, and ROI pooling run away
  from the UI isolate;
- only one request may be active at a time;
- frames received while the worker is busy are intentionally dropped.

This real-time backpressure strategy avoids an unbounded queue of obsolete
frames and keeps interaction latency lower than processing every captured frame.

## Architecture

The code follows a Contracts → Services → Providers → Presentation structure.

```text
lib/
├── main.dart
├── app.dart
├── app/
│   ├── contracts/
│   │   ├── detection/target_detector.dart
│   │   ├── inference/inference_service.dart
│   │   └── training/target_classifier.dart
│   ├── domain/
│   │   ├── entities/detection.dart
│   │   └── object_catalog.dart
│   ├── providers/
│   │   └── detector_service_provider.dart
│   └── services/
│       ├── detection/specific_target_detector.dart
│       ├── inference/isolate_inference_service.dart
│       └── training/target_network.dart
└── presentation/
    └── pages/detector_page.dart
```

### Layer responsibilities

| Layer | Responsibility |
|---|---|
| Domain | Detection entity, model dimensions, and COCO catalog |
| Contracts | Stable interfaces for detection, inference, and classification |
| Services | TFLite execution, isolate communication, and neural-network training |
| Provider | Construction and dependency injection of concrete services |
| Presentation | Camera lifecycle, user interaction, status text, and rendering |
| Composition root | Flutter initialization and application assembly |

The root-level `detector.dart`, `inference_worker.dart`, and
`target_network.dart` files are compatibility exports for existing imports.

## Requirements

- Flutter compatible with Dart SDK `^3.8.0`.
- Android device or emulator with API level 24 or newer.
- iOS device with camera access.
- A custom four-output TFLite model in `assets/models/`.
- A physical device is recommended for meaningful camera and inference
  performance testing.

Main dependencies:

```yaml
camera: ^0.11.2
tflite_flutter: ^0.12.1
image: ^4.5.4
```

## Getting started

Clone the repository:

```bash
git clone https://github.com/bolotkalil/user-selected-object-tracking.git
cd user-selected-object-tracking
```

Confirm that the model exists:

```text
assets/models/user_selected_object_yolo_p3_p4_p5.tflite
```

Install Flutter packages:

```bash
flutter pub get
```

Check the environment:

```bash
flutter doctor
```

Run on a connected device:

```bash
flutter run
```

For better performance measurements, use profile mode:

```bash
flutter run --profile
```

## Creating the custom TFLite model

A standard YOLO export is not sufficient because the application needs the P3,
P4, and P5 feature maps in addition to normal detection output.

### Google Colab workflow

1. Open `target_detector.ipynb` in Google Colab.
2. Install its Python dependencies.
3. Export the wrapped YOLO graph to:

   ```text
   user_selected_object_yolo_p3_p4_p5.onnx
   ```

4. Run `convert_to_tflite.py` in the same session. If the ONNX file is not
   present, the script opens a Colab upload dialog.
5. The script installs the conversion tools, runs `onnx2tf`, finds the float32
   model, validates all four output tensors, performs a smoke-test inference,
   and downloads:

   ```text
   user_selected_object_yolo_p3_p4_p5.tflite
   ```

6. Copy the downloaded file into `assets/models/`.

Run the standalone converter in Colab with:

```python
%run convert_to_tflite.py
```

The exported model is float32. Quantization requires additional validation
because this application consumes intermediate feature maps for identity
learning, not only final detection tensors.

## Platform configuration

### Android

Camera permission is declared in:

```text
android/app/src/main/AndroidManifest.xml
```

The current package configuration uses:

```text
applicationId: com.example.user_selected_object_tracking
minSdk: 24
```

Replace the example application ID and configure a release signing key before
publishing the application.

### iOS

`ios/Runner/Info.plist` contains `NSCameraUsageDescription`. Before release,
configure an Apple development team and production bundle identifier in Xcode.

The application requests portrait orientation at startup so camera pixels,
model coordinates, tap coordinates, and painted bounding boxes share a stable
coordinate system.

## Testing and code quality

Format the project:

```bash
dart format lib test
```

Run static analysis:

```bash
flutter analyze
```

Run unit tests:

```bash
flutter test
```

The current tests verify:

- COCO label lookup for detection entities;
- restoration of exported target-network weights;
- finite target-network output after a training step.

## Performance notes

- Use a physical device for testing.
- Debug mode adds substantial overhead; use profile or release mode when
  measuring latency.
- The camera requests `ResolutionPreset.low` because inference uses a 320 × 320
  input tensor.
- The TFLite interpreter uses four CPU threads.
- Only one frame is processed at a time.
- Candidate descriptors are capped at 12 detections per frame.
- Model inference and Dart training run outside the UI isolate.

## Limitations

- The online classifier is session-local; learned weights are not persisted
  after an application restart.
- Recognition quality depends on the visual information preserved in the YOLO
  feature maps and on variation in the enrollment frames.
- Similar-looking instances can still produce false positives.
- The selected object should remain visible during enrollment.
- A useful binary identity boundary requires automatically observed competing
  detections. Scenes containing only the selected instance provide less
  discriminative information.
- The current implementation tracks one selected instance at a time.
- Long-term reacquisition after large appearance changes or extended occlusion
  is not guaranteed.

## Troubleshooting

### `No camera found`

Use a device or emulator with an available camera and verify OS camera
permissions.

### TFLite model not loaded

Verify the exact asset path:

```text
assets/models/user_selected_object_yolo_p3_p4_p5.tflite
```

Then run:

```bash
flutter clean
flutter pub get
flutter run
```

### Unexpected output tensor error

The supplied model must expose one YOLO tensor and three feature maps with 64,
128, and 256 channels. Re-run the notebook and converter rather than using a
standard single-output YOLO TFLite export.

### Slow inference

Test in profile mode, close other applications, and verify that the device is
not thermal-throttling. Frames are dropped intentionally while inference is
busy; this prevents the UI from accumulating latency.

### Android NDK or SDK error

Install the SDK and NDK versions requested by the installed Flutter plugins,
then update `compileSdk` and `ndkVersion` in
`android/app/build.gradle.kts` to match the local Android toolchain.

## Repository contents

- `lib/` — Flutter application and on-device neural-network implementation.
- `assets/models/` — custom TFLite detector and feature extractor.
- `target_detector.ipynb` — Python reference pipeline and model export.
- `convert_to_tflite.py` — ONNX-to-TFLite conversion and validation script.
- `test/` — Dart unit tests.
- `android/` and `ios/` — platform projects and camera permissions.

## License

This project is distributed under the terms specified in the
[`LICENSE`](https://github.com/bolotkalil/user-selected-object-tracking/) file located in the repository root.
