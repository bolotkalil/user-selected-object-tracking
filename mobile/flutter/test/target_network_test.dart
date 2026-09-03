import 'dart:math' as math;

import 'package:flutter_test/flutter_test.dart';
import 'package:user_selected_object_tracking/target_network.dart';

void main() {
  test('exported weights restore the same prediction', () {
    final model = TargetNetwork(seed: 7);
    final features = List<double>.filled(TargetNetworkConfig.inputSize, 0.01);

    final restored = TargetNetwork.fromWeights(model.exportWeights());

    expect(restored.predict(features), closeTo(model.predict(features), 1e-12));
    expect(restored.trained, isTrue);
  });

  test('one training step produces a finite prediction', () async {
    final model = TargetNetwork(seed: 7);
    final positive = List<double>.filled(TargetNetworkConfig.inputSize, 0.02);
    final negative = List<double>.filled(TargetNetworkConfig.inputSize, -0.02);

    await model.train([positive, negative], [1, 0], steps: 1, batchSize: 2);

    expect(model.predict(positive).isFinite, isTrue);
    expect(model.predict(positive), inInclusiveRange(0, 1));
    expect(
      math.max(model.predict(positive), model.predict(negative)),
      lessThanOrEqualTo(1),
    );
  });
}
