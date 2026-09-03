import 'dart:typed_data';

import 'package:image/image.dart' as imglib;

import '../../domain/entities/detection.dart';

/// Contract implemented by concrete object-detection engines.
abstract interface class TargetDetector {
  int? get selectedClassId;
  set selectedClassId(int? value);

  Future<void> load();
  Future<void> loadFromBuffer(Uint8List modelBytes);
  Future<List<Detection>> detect(imglib.Image source);
  void close();
}
