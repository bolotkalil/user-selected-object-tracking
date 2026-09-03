import '../contracts/inference/inference_service.dart';
import '../contracts/training/target_classifier.dart';
import '../services/inference/isolate_inference_service.dart';
import '../services/training/target_network.dart';

/// Composition root for the target-detection feature.
///
/// This mirrors cleanrobot's service-provider role: presentation code depends
/// on contracts, while all concrete implementations are selected in one place.
final class DetectorServiceProvider {
  const DetectorServiceProvider();

  InferenceService createInferenceService() => IsolateInferenceService();

  TargetClassifier createTargetClassifier() => TargetNetwork();

  Future<TargetClassifier> trainTargetClassifier(
    List<List<double>> samples,
    List<double> labels,
  ) => trainTargetNetworkInBackground(samples, labels);
}
