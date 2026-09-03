import '../object_catalog.dart';

/// Domain representation of a localized object and its learned identity data.
class Detection {
  Detection(this.box, this.classId, this.confidence);

  final List<double> box;
  final int classId;
  final double confidence;
  double targetScore = 0;
  List<double> rawFeature = const [];

  String get label => cocoNames[classId];
}
