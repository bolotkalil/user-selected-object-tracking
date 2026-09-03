/// Contract for the on-device classifier that learns one selected identity.
abstract interface class TargetClassifier {
  bool get trained;
  double predict(List<double> features);
  Future<void> train(
    List<List<double>> samples,
    List<double> labels, {
    required int steps,
    required int batchSize,
  });
}
