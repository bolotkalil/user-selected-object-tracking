# User-Selected Object Tracking — Flutter and LiteRT/TFLite

Flutter camera app implementing the full pipeline from `target_detection.ipynb`:

1. Resize each camera frame to 320×320 RGB and normalize it to `[0,1]`.
2. Run the four-output YOLOv8 LiteRT/TFLite model.
3. Decode YOLO predictions, apply class-wise NMS, and ROI-pool P3/P4/P5 into 448 values.
4. Collect positive and negative samples for the object tapped by the user.
5. Train the notebook-equivalent mini neural network directly in Dart.
6. Mark the best instance as `TARGET` when its score is at least `0.40`.

## Create the TFLite model

The standard YOLO TFLite export is insufficient because this app also needs P3, P4, and P5.

1. Run notebook cell 3 to create `user_selected_object_yolo_p3_p4_p5.onnx`.
2. Run `convert_to_tflite.py` in the same Colab session.
3. Download `user_selected_object_yolo_p3_p4_p5.tflite`.
4. Place it in `assets/models/`.

The app identifies outputs by tensor shape, so it supports both NCHW and NHWC conversion.
Required outputs are YOLO `[1,84,N]` or `[1,N,84]` and three feature maps containing
64, 128, and 256 channels.

## Run

```sh
flutter clean
flutter pub get
flutter run
```

After launch, tap a detected object and keep it visible while 30 frames are collected. The
app trains the same `448→256→128→32→1` network using mobile mini-batch Adam, ReLU, Dropout 0.1,
L2-normalized embeddings, weighted BCE, learning rate `1e-3`, and weight decay `1e-4`.

Use the `Only: person` selector at the bottom to choose which COCO class is processed. For
example, select `dog` to hide every non-dog detection. Changing the class resets the remembered
instance and its training samples; tap a detection again to remember a particular instance of
the new class.

After training, recognition is intentionally strict: at most one object is displayed, YOLO
confidence must be at least `0.45`, and the target mini-network score must be at least `0.80`.
Only one box is also shown before and during training: the strongest candidate before selection,
then the selected tracked instance while samples are collected. Other detections remain hidden
but may be used internally as negative training examples.

The official `tflite_flutter` 0.12.1 package uses Google AI Edge LiteRT 1.4.0. The trainable
Dart network is in `lib/target_network.dart`; TFLite inference and ROI pooling are in
`lib/detector.dart`.
