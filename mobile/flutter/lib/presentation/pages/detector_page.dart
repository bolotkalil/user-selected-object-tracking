import 'dart:io';
import 'dart:math' as math;

import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import '../../app/contracts/inference/inference_service.dart';
import '../../app/contracts/training/target_classifier.dart';
import '../../app/domain/entities/detection.dart';
import '../../app/domain/object_catalog.dart';
import '../../app/providers/detector_service_provider.dart';
import '../../app/services/inference/isolate_inference_service.dart';

class DetectorPage extends StatefulWidget {
  const DetectorPage({
    super.key,
    required this.cameras,
    required this.services,
  });

  final List<CameraDescription> cameras;
  final DetectorServiceProvider services;
  @override
  State<DetectorPage> createState() => DetectorPageState();
}

class DetectorPageState extends State<DetectorPage>
    with WidgetsBindingObserver {
  CameraController? cameraController;
  late final InferenceService inferenceService;
  late TargetClassifier targetClassifier;
  List<Detection> visibleDetections = [];
  bool isProcessingFrame = false;
  String statusMessage = 'Loading models…';
  static const targetThreshold = .93;
  static const detectionThreshold = .50;
  static const trainFrames = 30;
  // Negative descriptors are mined automatically from competing detections
  // during the same enrollment window; no separate user-facing phase exists.
  static const minimumNegativeSamples = 10;
  static const trainingIouThreshold = .35;
  int? trainingClassId;
  List<double>? lastTrainingBox;
  int collectedFrameCount = 0;
  final List<List<double>> trainingFeatures = [];
  final List<double> trainingLabels = [];
  int selectedClassId = 0;

  @override
  void initState() {
    super.initState();
    inferenceService = widget.services.createInferenceService();
    targetClassifier = widget.services.createTargetClassifier();
    WidgetsBinding.instance.addObserver(this);
    initialize();
  }

  Future<void> initialize() async {
    try {
      if (widget.cameras.isEmpty) throw StateError('No camera found');
      await inferenceService.start();
      final camera = CameraController(
        widget.cameras.first,
        // The network consumes a 320 x 320 tensor. Requesting a low-resolution
        // camera stream minimizes YUV conversion and resampling overhead while
        // preserving sufficient spatial detail for the detector.
        ResolutionPreset.low,
        enableAudio: false,
        imageFormatGroup: Platform.isIOS
            ? ImageFormatGroup.bgra8888
            : ImageFormatGroup.nv21,
      );
      await camera.initialize();
      await camera.startImageStream(onFrame);
      if (!mounted) return;
      setState(() {
        cameraController = camera;
        statusMessage = 'Detecting only ${cocoNames[selectedClassId]}';
      });
    } catch (e) {
      if (mounted) setState(() => statusMessage = 'Setup error: $e');
    }
  }

  Future<void> onFrame(CameraImage frame) async {
    if (isProcessingFrame || inferenceService.isBusy) return;
    isProcessingFrame = true;
    try {
      final detections = await inferenceService.detect(
        frame,
        selectedClassId: selectedClassId,
        rotationDegrees: widget.cameras.first.sensorOrientation,
        mirrorHorizontally:
            widget.cameras.first.lensDirection == CameraLensDirection.front,
      );
      final wasCollectingTrainingData = trainingClassId != null;
      if (wasCollectingTrainingData) {
        await collectTrainingFrame(detections);
      } else if (targetClassifier.trained) {
        for (final d in detections) {
          d.targetScore = targetClassifier.predict(d.rawFeature);
        }
        detections.sort((a, b) => b.targetScore.compareTo(a.targetScore));

        // Recognition follows a two-stage acceptance rule. The semantic
        // detector must localize the selected class with sufficient confidence,
        // and the online classifier must independently confirm its identity.
        // Only the highest-scoring joint hypothesis is exposed to the UI.
        final best = detections.isEmpty ? null : detections.first;
        if (best == null ||
            best.confidence < detectionThreshold ||
            best.targetScore < targetThreshold) {
          detections.clear();
        } else {
          detections
            ..clear()
            ..add(best);
        }
      }

      final nextVisibleDetections = oneVisibleDetection(
        detections,
        trackingSelectedTarget: wasCollectingTrainingData,
      );
      if (!mounted) return;
      setState(() {
        visibleDetections = nextVisibleDetections;
        if (trainingClassId != null) return;
        statusMessage = !targetClassifier.trained
            ? detections.isEmpty
                  ? 'Looking for ${cocoNames[selectedClassId]}'
                  : 'Tap ${cocoNames[selectedClassId]} to remember it'
            : detections.isNotEmpty &&
                  detections.first.targetScore >= targetThreshold
            ? 'TARGET ${(detections.first.targetScore * 100).round()}%'
            : 'Looking for target';
      });
    } on InferenceServiceBusy {
      // Backpressure is expected: a newer frame will arrive immediately.
    } catch (e) {
      if (mounted) setState(() => statusMessage = 'Inference error: $e');
    } finally {
      isProcessingFrame = false;
    }
  }

  List<Detection> oneVisibleDetection(
    List<Detection> detections, {
    required bool trackingSelectedTarget,
  }) {
    if (detections.isEmpty) return [];

    if (trackingSelectedTarget && lastTrainingBox != null) {
      final sameClass = detections
          .where(
            (detection) =>
                detection.classId == selectedClassId &&
                areaRatioOk(detection.box, lastTrainingBox!) &&
                intersectionOverUnion(detection.box, lastTrainingBox!) >=
                    trainingIouThreshold,
          )
          .toList();
      if (sameClass.isEmpty) return [];
      sameClass.sort(
        (a, b) => boxDistance(
          a.box,
          lastTrainingBox!,
        ).compareTo(boxDistance(b.box, lastTrainingBox!)),
      );
      return [sameClass.first];
    }

    // Prior to enrollment, a single high-confidence proposal is presented to
    // make user intent unambiguous. Following enrollment, the upstream
    // recognition gate has already reduced the set to one verified identity.
    detections.sort((a, b) => b.confidence.compareTo(a.confidence));
    return [detections.first];
  }

  Future<void> collectTrainingFrame(List<Detection> detections) async {
    final candidates = detections.where((d) {
      return d.classId == trainingClassId &&
          areaRatioOk(d.box, lastTrainingBox!) &&
          intersectionOverUnion(d.box, lastTrainingBox!) >=
              trainingIouThreshold;
    }).toList();
    Detection? positive;
    if (candidates.isNotEmpty) {
      candidates.sort(
        (a, b) => boxDistance(
          a.box,
          lastTrainingBox!,
        ).compareTo(boxDistance(b.box, lastTrainingBox!)),
      );
      positive = candidates.first;
      lastTrainingBox = List.of(positive.box);
      if (collectedFrameCount < trainFrames) {
        trainingFeatures.add(List.of(positive.rawFeature));
        trainingLabels.add(1);
        collectedFrameCount++;
      }
    }

    // A candidate that does not overlap the tracked target is never promoted
    // to a positive sample. This prevents identity drift when the enrolled
    // person leaves the frame and another person occupies a nearby location.
    final negatives =
        detections
            .where(
              (detection) =>
                  !identical(detection, positive) &&
                  detection.classId == trainingClassId,
            )
            .toList()
          ..sort((a, b) => b.confidence.compareTo(a.confidence));
    // Hard-negative mining concentrates the limited online training budget on
    // the most plausible competing identities. Weak detections add little
    // discriminative information while increasing optimization cost.
    for (final d in negatives.take(3)) {
      trainingFeatures.add(List.of(d.rawFeature));
      trainingLabels.add(0);
    }
    final negativeCount = trainingLabels.where((label) => label == 0).length;
    if (mounted) {
      setState(() {
        statusMessage = collectedFrameCount < trainFrames
            ? 'Remembering target: $collectedFrameCount/$trainFrames'
            : negativeCount < minimumNegativeSamples
            ? 'Finalizing identity profile…'
            : 'Training mini neural network…';
      });
    }
    if (collectedFrameCount < trainFrames ||
        negativeCount < minimumNegativeSamples) {
      return;
    }
    trainingClassId = null;
    targetClassifier = await widget.services.trainTargetClassifier(
      trainingFeatures,
      trainingLabels,
    );
    if (mounted) {
      setState(() => statusMessage = 'Object remembered by neural network');
    }
  }

  void selectTarget(Offset point, Size size, Size previewSize) {
    if (trainingClassId != null || visibleDetections.isEmpty) return;
    final geometry = CoverGeometry.fit(source: previewSize, viewport: size);
    final sourcePoint = geometry.toSource(point);
    final mx = sourcePoint.dx / previewSize.width * modelSize;
    final my = sourcePoint.dy / previewSize.height * modelSize;
    final hit = visibleDetections.where((d) {
      final b = d.box;
      return mx >= b[0] && mx <= b[2] && my >= b[1] && my <= b[3];
    }).toList();
    if (hit.isEmpty) return;
    hit.sort((a, b) => boxArea(a.box).compareTo(boxArea(b.box)));
    final selected = hit.first;
    targetClassifier = widget.services.createTargetClassifier();
    trainingFeatures.clear();
    trainingLabels.clear();
    collectedFrameCount = 0;
    trainingClassId = selected.classId;
    lastTrainingBox = List.of(selected.box);
    setState(
      () => statusMessage = 'Selected ${selected.label}; keep it in view',
    );
  }

  void changeDetectedClass(int classId) {
    if (classId == selectedClassId) return;

    selectedClassId = classId;
    targetClassifier = widget.services.createTargetClassifier();
    visibleDetections = [];
    trainingClassId = null;
    lastTrainingBox = null;
    collectedFrameCount = 0;
    trainingFeatures.clear();
    trainingLabels.clear();

    setState(() {
      statusMessage = 'Detecting only ${cocoNames[classId]}';
    });
  }

  double boxArea(List<double> b) => (b[2] - b[0]).abs() * (b[3] - b[1]).abs();

  bool areaRatioOk(List<double> a, List<double> b) {
    final ratio = boxArea(a) / boxArea(b).clamp(1, double.infinity);
    return ratio >= .45 && ratio <= 2.2;
  }

  double boxDistance(List<double> a, List<double> b) {
    final dx = (a[0] + a[2] - b[0] - b[2]) / 2;
    final dy = (a[1] + a[3] - b[1] - b[3]) / 2;
    return dx * dx + dy * dy;
  }

  double intersectionOverUnion(List<double> a, List<double> b) {
    final left = math.max(a[0], b[0]);
    final top = math.max(a[1], b[1]);
    final right = math.min(a[2], b[2]);
    final bottom = math.min(a[3], b[3]);
    final intersection =
        math.max(0.0, right - left) * math.max(0.0, bottom - top);
    return intersection / (boxArea(a) + boxArea(b) - intersection + 1e-6);
  }

  @override
  void dispose() {
    WidgetsBinding.instance.removeObserver(this);
    final camera = cameraController;
    cameraController = null;
    if (camera != null) {
      if (camera.value.isStreamingImages) {
        camera.stopImageStream().whenComplete(camera.dispose);
      } else {
        camera.dispose();
      }
    }
    inferenceService.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final camera = cameraController;
    return Scaffold(
      body: camera == null
          ? Center(child: Text(statusMessage, textAlign: TextAlign.center))
          : LayoutBuilder(
              builder: (context, constraints) {
                final nativePreviewSize = camera.value.previewSize!;
                final previewSize =
                    MediaQuery.orientationOf(context) == Orientation.portrait
                    ? Size(nativePreviewSize.height, nativePreviewSize.width)
                    : Size(nativePreviewSize.width, nativePreviewSize.height);
                return GestureDetector(
                  behavior: HitTestBehavior.opaque,
                  onTapDown: (event) => selectTarget(
                    event.localPosition,
                    constraints.biggest,
                    previewSize,
                  ),
                  child: Stack(
                    fit: StackFit.expand,
                    children: [
                      ClipRect(
                        child: FittedBox(
                          fit: BoxFit.cover,
                          child: SizedBox(
                            width: previewSize.width,
                            height: previewSize.height,
                            child: CameraPreview(camera),
                          ),
                        ),
                      ),
                      CustomPaint(
                        painter: BoxPainter(
                          visibleDetections,
                          targetThreshold,
                          targetClassifier.trained,
                          previewSize,
                        ),
                      ),
                      SafeArea(
                        child: Align(
                          alignment: Alignment.topCenter,
                          child: Container(
                            margin: const EdgeInsets.all(16),
                            padding: const EdgeInsets.all(12),
                            decoration: BoxDecoration(
                              color: Colors.black.withValues(alpha: .70),
                              borderRadius: BorderRadius.circular(12),
                            ),
                            child: Text(
                              statusMessage,
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                        ),
                      ),
                      SafeArea(
                        child: Align(
                          alignment: Alignment.bottomCenter,
                          child: Padding(
                            padding: const EdgeInsets.all(16),
                            child: PopupMenuButton<int>(
                              initialValue: selectedClassId,
                              onSelected: changeDetectedClass,
                              itemBuilder: (context) => [
                                for (
                                  var classId = 0;
                                  classId < cocoNames.length;
                                  classId++
                                )
                                  PopupMenuItem<int>(
                                    value: classId,
                                    child: Text(cocoNames[classId]),
                                  ),
                              ],
                              child: Container(
                                padding: const EdgeInsets.symmetric(
                                  horizontal: 18,
                                  vertical: 12,
                                ),
                                decoration: BoxDecoration(
                                  color: Colors.black.withValues(alpha: .78),
                                  borderRadius: BorderRadius.circular(24),
                                  border: Border.all(color: Colors.white24),
                                ),
                                child: Row(
                                  mainAxisSize: MainAxisSize.min,
                                  children: [
                                    const Icon(Icons.filter_center_focus),
                                    const SizedBox(width: 8),
                                    Text(
                                      'Only: ${cocoNames[selectedClassId]}',
                                      style: const TextStyle(
                                        fontWeight: FontWeight.bold,
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    const Icon(Icons.arrow_drop_down),
                                  ],
                                ),
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  ),
                );
              },
            ),
    );
  }
}

class BoxPainter extends CustomPainter {
  BoxPainter(this.detections, this.threshold, this.trained, this.previewSize);
  final List<Detection> detections;
  final double threshold;
  final bool trained;
  final Size previewSize;
  @override
  void paint(Canvas canvas, Size size) {
    final geometry = CoverGeometry.fit(source: previewSize, viewport: size);
    for (var i = 0; i < detections.length; i++) {
      final d = detections[i];
      final target = trained && i == 0 && d.targetScore >= threshold;
      final sourceBox = Rect.fromLTRB(
        d.box[0] / modelSize * previewSize.width,
        d.box[1] / modelSize * previewSize.height,
        d.box[2] / modelSize * previewSize.width,
        d.box[3] / modelSize * previewSize.height,
      );
      final box = geometry.toViewportRect(sourceBox);
      final color = target ? Colors.greenAccent : Colors.white54;
      canvas.drawRect(
        box,
        Paint()
          ..color = color
          ..style = PaintingStyle.stroke
          ..strokeWidth = target ? 4 : 1,
      );
      final score = trained
          ? ' · ${(d.targetScore * 100).round()}%'
          : ' · ID $i';
      final text =
          '${target ? 'TARGET ' : ''}${d.label} ${(d.confidence * 100).round()}%$score';
      final tp = TextPainter(
        text: TextSpan(
          text: text,
          style: TextStyle(
            color: color,
            backgroundColor: Colors.black87,
            fontSize: 13,
          ),
        ),
        textDirection: TextDirection.ltr,
      )..layout();
      tp.paint(
        canvas,
        Offset(
          box.left,
          (box.top - tp.height).clamp(0, size.height).toDouble(),
        ),
      );
    }
  }

  @override
  bool shouldRepaint(covariant BoxPainter old) => true;
}

/// Bidirectional mapping for a source image displayed with [BoxFit.cover].
final class CoverGeometry {
  const CoverGeometry(this.scale, this.offset);

  factory CoverGeometry.fit({required Size source, required Size viewport}) {
    final scale = math.max(
      viewport.width / source.width,
      viewport.height / source.height,
    );
    return CoverGeometry(
      scale,
      Offset(
        (viewport.width - source.width * scale) / 2,
        (viewport.height - source.height * scale) / 2,
      ),
    );
  }

  final double scale;
  final Offset offset;

  Offset toSource(Offset point) => (point - offset) / scale;

  Rect toViewportRect(Rect rect) => Rect.fromLTRB(
    rect.left * scale + offset.dx,
    rect.top * scale + offset.dy,
    rect.right * scale + offset.dx,
    rect.bottom * scale + offset.dy,
  );
}
