import 'dart:math' as math;
import 'dart:typed_data';

import 'package:image/image.dart' as imglib;
import 'package:tflite_flutter/tflite_flutter.dart';

import '../../contracts/detection/target_detector.dart';
import '../../domain/entities/detection.dart';
import '../../domain/object_catalog.dart';

/// Executes the shared YOLOv8 feature pyramid and converts its outputs into
/// instance-level descriptors suitable for online target adaptation.
///
/// The TFLite graph exposes both the conventional detection tensor and the
/// intermediate P3, P4, and P5 feature maps. Each accepted detection is
/// projected onto all three pyramid levels by ROI average pooling. The pooled
/// vectors are concatenated and L2-normalized to obtain a 448-dimensional
/// representation that is independent of the original camera resolution.
class SpecificTargetDetector implements TargetDetector {
  Interpreter? modelInterpreter;
  final Map<int, Object> outputBuffers = {};
  final List<List<int>> outputShapes = [];

  /// Restricts post-processing to a single semantic COCO category.
  ///
  /// Filtering before ROI pooling prevents unrelated detections from consuming
  /// descriptor extraction time. A null value enables all COCO categories.
  @override
  int? selectedClassId = 0;

  @override
  Future<void> load() async {
    modelInterpreter = await Interpreter.fromAsset(
      'assets/models/user_selected_object_yolo_p3_p4_p5.tflite',
      options: InterpreterOptions()..threads = 4,
    );
    initializeOutputBuffers();
  }

  /// Loads the model from memory inside a background isolate.
  @override
  Future<void> loadFromBuffer(Uint8List modelBytes) async {
    modelInterpreter = Interpreter.fromBuffer(
      modelBytes,
      options: InterpreterOptions()..threads = 4,
    );
    initializeOutputBuffers();
  }

  void initializeOutputBuffers() {
    outputBuffers.clear();
    outputShapes.clear();
    for (final tensor in modelInterpreter!.getOutputTensors()) {
      final shape = tensor.shape.cast<int>();
      outputShapes.add(shape);
      outputBuffers[outputShapes.length - 1] = createZeroBuffer(shape);
    }
  }

  @override
  void close() {
    modelInterpreter?.close();
    modelInterpreter = null;
    outputBuffers.clear();
    outputShapes.clear();
  }

  @override
  Future<List<Detection>> detect(imglib.Image source) async {
    final interpreter = modelInterpreter;
    if (interpreter == null) throw StateError('TFLite model not loaded');
    final inputTensor = interpreter.getInputTensor(0);
    final inputShape = inputTensor.shape.cast<int>();
    if (inputTensor.type != TensorType.float32) {
      throw StateError('Expected float32 input, got ${inputTensor.type}');
    }

    final resized = imglib.copyResize(
      source,
      width: modelSize,
      height: modelSize,
    );
    final isNhwc = inputShape.length == 4 && inputShape.last == 3;
    final flatInput = Float32List(3 * modelSize * modelSize);
    for (var y = 0; y < modelSize; y++) {
      for (var x = 0; x < modelSize; x++) {
        final p = resized.getPixel(x, y);
        if (isNhwc) {
          final i = (y * modelSize + x) * 3;
          flatInput[i] = p.r / 255;
          flatInput[i + 1] = p.g / 255;
          flatInput[i + 2] = p.b / 255;
        } else {
          final i = y * modelSize + x;
          flatInput[i] = p.r / 255;
          flatInput[modelSize * modelSize + i] = p.g / 255;
          flatInput[2 * modelSize * modelSize + i] = p.b / 255;
        }
      }
    }

    // Passing a contiguous Float32 buffer avoids constructing a deeply nested
    // 320 x 320 x 3 object graph on every frame. The interpreter already owns
    // the input shape, so no structural information is lost.
    interpreter.runForMultipleInputs([flatInput.buffer], outputBuffers);

    TensorData? yolo;
    final pyramids = <int, TensorData>{};
    for (final entry in outputBuffers.entries) {
      final shape = outputShapes[entry.key];
      final data = TensorData(shape, flattenTensor(entry.value));
      if (shape.length == 3 && shape.contains(84)) {
        yolo = data;
      } else if (shape.length == 4) {
        for (final channels in const [64, 128, 256]) {
          if (shape[1] == channels || shape.last == channels) {
            pyramids[channels] = data;
          }
        }
      }
    }
    if (yolo == null || pyramids.length != 3) {
      throw StateError(
        'Model must output YOLO [1,84,N] plus P3/P4/P5 with 64/128/256 channels. '
        'Actual outputs: $outputShapes',
      );
    }

    final detections = decodeDetections(yolo);
    for (final d in detections) {
      d.rawFeature = [
        ...poolRegionFeatures(pyramids[64]!, 64, d.box),
        ...poolRegionFeatures(pyramids[128]!, 128, d.box),
        ...poolRegionFeatures(pyramids[256]!, 256, d.box),
      ];
      normalizeVector(d.rawFeature);
    }
    return detections;
  }

  List<Detection> decodeDetections(
    TensorData tensor, {
    double confidence = .25,
    double iou = .60,
  }) {
    final channelFirst = tensor.shape[1] == 84;
    final anchors = channelFirst ? tensor.shape[2] : tensor.shape[1];
    double at(int channel, int anchor) => channelFirst
        ? tensor.values[channel * anchors + anchor]
        : tensor.values[anchor * 84 + channel];
    final byClass = <int, List<Detection>>{};
    for (var a = 0; a < anchors; a++) {
      var cls = 0;
      var best = -double.infinity;
      for (var c = 0; c < 80; c++) {
        final score = at(4 + c, a);
        if (score > best) {
          best = score;
          cls = c;
        }
      }
      if (best <= confidence) continue;
      if (selectedClassId != null && cls != selectedClassId) continue;
      final cx = at(0, a), cy = at(1, a), w = at(2, a), h = at(3, a);
      final box = [
        (cx - w / 2).clamp(0, modelSize).toDouble(),
        (cy - h / 2).clamp(0, modelSize).toDouble(),
        (cx + w / 2).clamp(0, modelSize).toDouble(),
        (cy + h / 2).clamp(0, modelSize).toDouble(),
      ];
      // Border-touching proposals usually represent objects that are only
      // partially visible. They are unsuitable for unambiguous enrollment and
      // tend to produce unstable identity descriptors as the camera moves.
      const borderMargin = 2.0;
      if (box[0] <= borderMargin ||
          box[1] <= borderMargin ||
          box[2] >= modelSize - borderMargin ||
          box[3] >= modelSize - borderMargin) {
        continue;
      }
      (byClass[cls] ??= []).add(Detection(box, cls, best));
    }
    final result = <Detection>[];
    for (final candidates in byClass.values) {
      candidates.sort((a, b) => b.confidence.compareTo(a.confidence));
      while (candidates.isNotEmpty) {
        final current = candidates.removeAt(0);
        result.add(current);
        candidates.removeWhere(
          (other) => intersectionOverUnion(current.box, other.box) >= iou,
        );
      }
    }
    result.sort((a, b) => b.confidence.compareTo(a.confidence));
    // Descriptor extraction is bounded to the strongest candidates. This cap
    // provides predictable latency and memory use in crowded scenes without
    // changing class-wise non-maximum suppression.
    return result.take(12).toList();
  }

  List<double> poolRegionFeatures(
    TensorData map,
    int channels,
    List<double> box,
  ) {
    final nchw = map.shape[1] == channels;
    final height = nchw ? map.shape[2] : map.shape[1];
    final width = nchw ? map.shape[3] : map.shape[2];
    var x1 = (box[0] / modelSize * width).floor().clamp(0, width - 1).toInt();
    var y1 = (box[1] / modelSize * height).floor().clamp(0, height - 1).toInt();
    var x2 = (box[2] / modelSize * width).floor().clamp(0, width - 1).toInt();
    var y2 = (box[3] / modelSize * height).floor().clamp(0, height - 1).toInt();
    if (x2 < x1) {
      final t = x1;
      x1 = x2;
      x2 = t;
    }
    if (y2 < y1) {
      final t = y1;
      y1 = y2;
      y2 = t;
    }
    final count = (x2 - x1 + 1) * (y2 - y1 + 1);
    final out = List<double>.filled(channels, 0);
    for (var c = 0; c < channels; c++) {
      var sum = 0.0;
      for (var y = y1; y <= y2; y++) {
        for (var x = x1; x <= x2; x++) {
          final index = nchw
              ? (c * height + y) * width + x
              : (y * width + x) * channels + c;
          sum += map.values[index];
        }
      }
      out[c] = sum / count;
    }
    normalizeVector(out);
    return out;
  }

  void normalizeVector(List<double> values) {
    final norm = math.sqrt(values.fold<double>(0, (s, v) => s + v * v)) + 1e-6;
    for (var i = 0; i < values.length; i++) {
      values[i] /= norm;
    }
  }

  double intersectionOverUnion(List<double> a, List<double> b) {
    final x1 = math.max(a[0], b[0]), y1 = math.max(a[1], b[1]);
    final x2 = math.min(a[2], b[2]), y2 = math.min(a[3], b[3]);
    final intersection = math.max(0, x2 - x1) * math.max(0, y2 - y1);
    final aa = math.max(0, a[2] - a[0]) * math.max(0, a[3] - a[1]);
    final ba = math.max(0, b[2] - b[0]) * math.max(0, b[3] - b[1]);
    return intersection / (aa + ba - intersection + 1e-6);
  }

  Object createZeroBuffer(List<int> shape, [int depth = 0]) =>
      depth == shape.length - 1
      ? List<double>.filled(shape[depth], 0)
      : List.generate(shape[depth], (_) => createZeroBuffer(shape, depth + 1));

  Float32List flattenTensor(Object value) {
    var length = 1;
    Object? cursor = value;
    while (cursor is List) {
      length *= cursor.length;
      cursor = cursor.isEmpty ? null : cursor.first;
    }
    final result = Float32List(length);
    var offset = 0;
    void visit(Object? item) {
      if (item is num) {
        result[offset++] = item.toDouble();
      }
      if (item is Iterable) {
        for (final child in item) {
          visit(child);
        }
      }
    }

    visit(value);
    return result;
  }
}

class TensorData {
  const TensorData(this.shape, this.values);
  final List<int> shape;
  final Float32List values;
}
