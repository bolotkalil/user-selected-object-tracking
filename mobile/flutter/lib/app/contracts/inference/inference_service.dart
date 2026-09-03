import 'package:camera/camera.dart';

import '../../domain/entities/detection.dart';

/// Contract for asynchronous camera inference with explicit backpressure.
abstract interface class InferenceService {
  bool get isReady;
  bool get isBusy;

  Future<void> start();
  Future<List<Detection>> detect(
    CameraImage frame, {
    required int selectedClassId,
    required int rotationDegrees,
    required bool mirrorHorizontally,
  });
  Future<void> dispose();
}
