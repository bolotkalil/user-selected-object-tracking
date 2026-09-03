import 'dart:async';
import 'dart:isolate';

import 'package:camera/camera.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as imglib;

import '../../contracts/inference/inference_service.dart';
import '../../domain/entities/detection.dart';
import '../detection/specific_target_detector.dart';

/// Owns a single, long-lived isolate dedicated to camera-frame inference.
///
/// The worker loads its own TFLite interpreter. This is important because a
/// native interpreter must be created, invoked, and destroyed in the same
/// isolate. At most one frame can be in flight; stale camera frames are dropped
/// by the caller instead of being queued and increasing latency.
final class IsolateInferenceService implements InferenceService {
  static const modelAssetPath =
      'assets/models/user_selected_object_yolo_p3_p4_p5.tflite';

  final ReceivePort responsePort = ReceivePort();

  StreamSubscription<Object?>? responseSubscription;
  Isolate? workerIsolate;
  SendPort? commandPort;
  Completer<void>? startupCompleter;
  Completer<List<Detection>>? pendingRequest;
  int nextRequestId = 0;
  bool isDisposed = false;

  @override
  bool get isReady => !isDisposed && commandPort != null;
  @override
  bool get isBusy => pendingRequest != null;

  @override
  Future<void> start() async {
    if (isDisposed) throw StateError('Inference service has been disposed');
    if (isReady) return;
    if (startupCompleter != null) return startupCompleter!.future;

    final startup = startupCompleter = Completer<void>();
    responseSubscription = responsePort.listen(handleResponse);

    final modelData = await rootBundle.load(modelAssetPath);
    final modelBytes = modelData.buffer.asUint8List(
      modelData.offsetInBytes,
      modelData.lengthInBytes,
    );

    workerIsolate = await Isolate.spawn<Map<String, Object>>(
      inferenceEntryPoint,
      <String, Object>{
        'hostPort': responsePort.sendPort,
        'model': TransferableTypedData.fromList([modelBytes]),
      },
      debugName: 'target-inference-worker',
      errorsAreFatal: true,
      onError: responsePort.sendPort,
      onExit: responsePort.sendPort,
    );

    return startup.future;
  }

  /// Submits a frame if the worker is idle.
  ///
  /// Plane buffers are transferred rather than retained as [CameraImage]
  /// objects, preventing camera-owned memory from crossing asynchronous
  /// boundaries and allowing zero-copy ownership transfer where supported.
  @override
  Future<List<Detection>> detect(
    CameraImage frame, {
    required int selectedClassId,
    required int rotationDegrees,
    required bool mirrorHorizontally,
  }) async {
    if (isDisposed) throw StateError('Inference service has been disposed');
    if (!isReady) await start();
    if (isBusy) throw const InferenceServiceBusy();

    final requestId = nextRequestId++;
    final completer = Completer<List<Detection>>();
    pendingRequest = completer;

    commandPort!.send(<String, Object>{
      'type': 'detect',
      'id': requestId,
      'classId': selectedClassId,
      'rotationDegrees': rotationDegrees,
      'mirrorHorizontally': mirrorHorizontally,
      'frame': serializeFrame(frame),
    });

    return completer.future;
  }

  Map<String, Object> serializeFrame(CameraImage frame) {
    return <String, Object>{
      'width': frame.width,
      'height': frame.height,
      'bgra': frame.format.group == ImageFormatGroup.bgra8888,
      'planes': <Map<String, Object>>[
        for (final plane in frame.planes)
          <String, Object>{
            'bytes': TransferableTypedData.fromList([plane.bytes]),
            'bytesPerRow': plane.bytesPerRow,
            'bytesPerPixel': plane.bytesPerPixel ?? 1,
          },
      ],
    };
  }

  void handleResponse(Object? message) {
    if (message is SendPort) {
      commandPort = message;
      final startup = startupCompleter;
      if (startup != null && !startup.isCompleted) startup.complete();
      return;
    }

    if (message is List<Object?>) {
      failPendingOperations(StateError('Inference isolate failed: $message'));
      return;
    }
    if (message == null) {
      if (!isDisposed) {
        failPendingOperations(StateError('Inference isolate exited'));
      }
      return;
    }
    if (message is! Map) return;

    final startup = startupCompleter;
    final error = message['error'];
    if (commandPort == null && error != null) {
      if (startup != null && !startup.isCompleted) {
        startup.completeError(StateError(error.toString()));
      }
      return;
    }

    final pending = pendingRequest;
    if (pending == null) return;
    pendingRequest = null;

    if (error != null) {
      pending.completeError(StateError(error.toString()));
      return;
    }

    final rows = (message['detections'] as List<Object?>?) ?? const [];
    pending.complete([
      for (final row in rows)
        decodeDetectionMessage(row as Map<Object?, Object?>),
    ]);
  }

  Detection decodeDetectionMessage(Map<Object?, Object?> row) {
    final detection = Detection(
      (row['box'] as List<Object?>)
          .cast<num>()
          .map((v) => v.toDouble())
          .toList(),
      row['classId'] as int,
      (row['confidence'] as num).toDouble(),
    );
    detection.rawFeature = (row['rawFeature'] as List<Object?>)
        .cast<num>()
        .map((v) => v.toDouble())
        .toList();
    return detection;
  }

  void failPendingOperations(Object error) {
    final startup = startupCompleter;
    if (startup != null && !startup.isCompleted) startup.completeError(error);
    final pending = pendingRequest;
    pendingRequest = null;
    if (pending != null && !pending.isCompleted) pending.completeError(error);
  }

  @override
  Future<void> dispose() async {
    if (isDisposed) return;
    isDisposed = true;
    failPendingOperations(StateError('Inference service has been disposed'));
    commandPort?.send(const <String, Object>{'type': 'shutdown'});
    commandPort = null;
    workerIsolate?.kill(priority: Isolate.beforeNextEvent);
    workerIsolate = null;
    await responseSubscription?.cancel();
    responsePort.close();
  }
}

final class InferenceServiceBusy implements Exception {
  const InferenceServiceBusy();

  @override
  String toString() => 'The previous camera frame is still being processed';
}

@pragma('vm:entry-point')
Future<void> inferenceEntryPoint(Map<String, Object> bootstrap) async {
  final hostPort = bootstrap['hostPort']! as SendPort;
  final modelTransfer = bootstrap['model']! as TransferableTypedData;
  final detector = SpecificTargetDetector();
  final commands = ReceivePort();

  try {
    await detector.loadFromBuffer(modelTransfer.materialize().asUint8List());
    hostPort.send(commands.sendPort);

    await for (final Object? rawMessage in commands) {
      if (rawMessage is! Map) continue;
      if (rawMessage['type'] == 'shutdown') break;
      if (rawMessage['type'] != 'detect') continue;

      final requestId = rawMessage['id'] as int;
      try {
        detector.selectedClassId = rawMessage['classId'] as int;
        var image = convertFrameToRgb(
          rawMessage['frame'] as Map<Object?, Object?>,
        );
        final rotationDegrees = rawMessage['rotationDegrees'] as int;
        if (rotationDegrees != 0) {
          image = imglib.copyRotate(image, angle: rotationDegrees);
        }
        if (rawMessage['mirrorHorizontally'] as bool) {
          image = imglib.flipHorizontal(image);
        }
        final detections = await detector.detect(image);
        hostPort.send(<String, Object>{
          'id': requestId,
          'detections': [
            for (final detection in detections)
              <String, Object>{
                'box': detection.box,
                'classId': detection.classId,
                'confidence': detection.confidence,
                'rawFeature': detection.rawFeature,
              },
          ],
        });
      } catch (error, stackTrace) {
        hostPort.send(<String, Object>{
          'id': requestId,
          'error': '$error\n$stackTrace',
        });
      }
    }
  } catch (error, stackTrace) {
    hostPort.send(<String, Object>{'error': '$error\n$stackTrace'});
  } finally {
    commands.close();
    detector.close();
  }
}

imglib.Image convertFrameToRgb(Map<Object?, Object?> frame) {
  final width = frame['width'] as int;
  final height = frame['height'] as int;
  final isBgra = frame['bgra'] as bool;
  final planes = (frame['planes'] as List<Object?>)
      .cast<Map<Object?, Object?>>()
      .map(PlaneData.fromMessage)
      .toList(growable: false);
  final output = imglib.Image(width: width, height: height);

  if (isBgra) {
    final plane = planes.first;
    for (var y = 0; y < height; y++) {
      for (var x = 0; x < width; x++) {
        final offset = y * plane.bytesPerRow + x * plane.bytesPerPixel;
        if (offset + 2 >= plane.bytes.length) continue;
        output.setPixelRgb(
          x,
          y,
          plane.bytes[offset + 2],
          plane.bytes[offset + 1],
          plane.bytes[offset],
        );
      }
    }
    return output;
  }

  final yPlane = planes.first;
  for (var y = 0; y < height; y++) {
    for (var x = 0; x < width; x++) {
      final yOffset = y * yPlane.bytesPerRow + x * yPlane.bytesPerPixel;
      if (yOffset >= yPlane.bytes.length) continue;

      final luminance = yPlane.bytes[yOffset].toDouble();
      var u = 0.0;
      var v = 0.0;

      if (planes.length >= 3) {
        final uPlane = planes[1];
        final vPlane = planes[2];
        final uOffset =
            (y ~/ 2) * uPlane.bytesPerRow + (x ~/ 2) * uPlane.bytesPerPixel;
        final vOffset =
            (y ~/ 2) * vPlane.bytesPerRow + (x ~/ 2) * vPlane.bytesPerPixel;
        if (uOffset < uPlane.bytes.length) u = uPlane.bytes[uOffset] - 128.0;
        if (vOffset < vPlane.bytes.length) v = vPlane.bytes[vOffset] - 128.0;
      } else {
        final uvPlane = planes.length == 2 ? planes[1] : yPlane;
        final chromaOffset = planes.length == 1
            ? calculateNv21ChromaOffset(
                yPlane.bytesPerRow,
                height,
                uvPlane.bytes.length,
              )
            : 0;
        final uvOffset =
            chromaOffset +
            (y ~/ 2) * uvPlane.bytesPerRow +
            (x ~/ 2) * uvPlane.bytesPerPixel;
        if (uvOffset + 1 < uvPlane.bytes.length) {
          v = uvPlane.bytes[uvOffset] - 128.0;
          u = uvPlane.bytes[uvOffset + 1] - 128.0;
        }
      }

      final red = (luminance + 1.402 * v).round().clamp(0, 255);
      final green = (luminance - .344136 * u - .714136 * v).round().clamp(
        0,
        255,
      );
      final blue = (luminance + 1.772 * u).round().clamp(0, 255);
      output.setPixelRgb(x, y, red, green, blue);
    }
  }
  return output;
}

int calculateNv21ChromaOffset(int rowStride, int height, int bufferLength) {
  final strideOffset = rowStride * height;
  return strideOffset < bufferLength
      ? strideOffset
      : (bufferLength * 2 ~/ 3).clamp(0, bufferLength);
}

final class PlaneData {
  const PlaneData(this.bytes, this.bytesPerRow, this.bytesPerPixel);

  factory PlaneData.fromMessage(Map<Object?, Object?> message) {
    return PlaneData(
      (message['bytes']! as TransferableTypedData).materialize().asUint8List(),
      message['bytesPerRow']! as int,
      message['bytesPerPixel']! as int,
    );
  }

  final Uint8List bytes;
  final int bytesPerRow;
  final int bytesPerPixel;
}
