import 'dart:isolate';
import 'dart:math' as math;

import '../../contracts/training/target_classifier.dart';

class TargetNetworkConfig {
  const TargetNetworkConfig._();

  static const inputSize = 448;
  static const embeddingHiddenSize = 256;
  static const embeddingSize = 128;
  static const classifierHiddenSize = 32;
  static const dropoutRate = 0.1;

  static const trainingSteps = 240;
  static const batchSize = 12;
  static const learningRate = 1e-3;
  static const weightDecay = 1e-4;
}

/// A compact instance classifier trained on-device from pooled YOLO features.
///
/// The architecture mirrors the experimental PyTorch model:
/// `448 -> 256 -> 128 -> 32 -> 1`. The 128-dimensional embedding is
/// L2-normalized before classification, which constrains its scale and improves
/// numerical stability under the small online training set. The final scalar
/// is a logit and is converted to a probability only at the inference boundary.
class TargetNetwork implements TargetClassifier {
  TargetNetwork({int? seed}) : randomGenerator = math.Random(seed) {
    embeddingHiddenLayer = DenseLayer.random(
      inputSize: TargetNetworkConfig.inputSize,
      outputSize: TargetNetworkConfig.embeddingHiddenSize,
      random: randomGenerator,
    );
    embeddingOutputLayer = DenseLayer.random(
      inputSize: TargetNetworkConfig.embeddingHiddenSize,
      outputSize: TargetNetworkConfig.embeddingSize,
      random: randomGenerator,
    );
    classifierHiddenLayer = DenseLayer.random(
      inputSize: TargetNetworkConfig.embeddingSize,
      outputSize: TargetNetworkConfig.classifierHiddenSize,
      random: randomGenerator,
    );
    classifierOutputLayer = DenseLayer.random(
      inputSize: TargetNetworkConfig.classifierHiddenSize,
      outputSize: 1,
      random: randomGenerator,
    );
  }

  TargetNetwork.fromWeights(List<List<double>> weights)
    : randomGenerator = math.Random() {
    if (weights.length != 8) {
      throw ArgumentError.value(
        weights.length,
        'weights',
        'Expected 8 tensors',
      );
    }

    embeddingHiddenLayer = DenseLayer.fromWeights(
      inputSize: TargetNetworkConfig.inputSize,
      outputSize: TargetNetworkConfig.embeddingHiddenSize,
      weights: weights[0],
      biases: weights[1],
    );
    embeddingOutputLayer = DenseLayer.fromWeights(
      inputSize: TargetNetworkConfig.embeddingHiddenSize,
      outputSize: TargetNetworkConfig.embeddingSize,
      weights: weights[2],
      biases: weights[3],
    );
    classifierHiddenLayer = DenseLayer.fromWeights(
      inputSize: TargetNetworkConfig.embeddingSize,
      outputSize: TargetNetworkConfig.classifierHiddenSize,
      weights: weights[4],
      biases: weights[5],
    );
    classifierOutputLayer = DenseLayer.fromWeights(
      inputSize: TargetNetworkConfig.classifierHiddenSize,
      outputSize: 1,
      weights: weights[6],
      biases: weights[7],
    );
    trained = true;
  }

  final math.Random randomGenerator;
  late final DenseLayer embeddingHiddenLayer;
  late final DenseLayer embeddingOutputLayer;
  late final DenseLayer classifierHiddenLayer;
  late final DenseLayer classifierOutputLayer;

  @override
  bool trained = false;

  List<TrainableParameter> get parameters => [
    ...embeddingHiddenLayer.parameters,
    ...embeddingOutputLayer.parameters,
    ...classifierHiddenLayer.parameters,
    ...classifierOutputLayer.parameters,
  ];

  @override
  double predict(List<double> features) {
    validateFeatureVector(features);
    return sigmoid(forwardNetwork(features, training: false).logit);
  }

  @override
  Future<void> train(
    List<List<double>> samples,
    List<double> labels, {
    int steps = TargetNetworkConfig.trainingSteps,
    int batchSize = TargetNetworkConfig.batchSize,
  }) async {
    validateTrainingData(samples, labels);

    final positiveIndices = <int>[
      for (var index = 0; index < labels.length; index++)
        if (labels[index] == 1) index,
    ];
    final negativeIndices = <int>[
      for (var index = 0; index < labels.length; index++)
        if (labels[index] == 0) index,
    ];
    final optimizer = AdamOptimizer(
      parameters: parameters,
      learningRate: TargetNetworkConfig.learningRate,
      weightDecay: TargetNetworkConfig.weightDecay,
    );

    for (var step = 1; step <= steps; step++) {
      optimizer.zeroGradients();
      final batch = createBalancedBatch(
        positiveIndices,
        negativeIndices,
        batchSize,
      );

      for (final sampleIndex in batch) {
        final cache = forwardNetwork(samples[sampleIndex], training: true);
        final label = labels[sampleIndex];
        final logitGradient = (sigmoid(cache.logit) - label) / batch.length;
        backwardNetwork(cache, logitGradient);
      }

      optimizer.step(step);
    }

    trained = true;
  }

  List<List<double>> exportWeights() => [
    List.of(embeddingHiddenLayer.weights.values),
    List.of(embeddingHiddenLayer.biases.values),
    List.of(embeddingOutputLayer.weights.values),
    List.of(embeddingOutputLayer.biases.values),
    List.of(classifierHiddenLayer.weights.values),
    List.of(classifierHiddenLayer.biases.values),
    List.of(classifierOutputLayer.weights.values),
    List.of(classifierOutputLayer.biases.values),
  ];

  NetworkCache forwardNetwork(List<double> input, {required bool training}) {
    final embeddingHiddenLinear = embeddingHiddenLayer.forward(input);
    final embeddingHiddenActivated = relu(embeddingHiddenLinear);
    final embeddingDropout = DropoutResult.apply(
      embeddingHiddenActivated,
      rate: TargetNetworkConfig.dropoutRate,
      random: randomGenerator,
      training: training,
    );

    final embeddingLinear = embeddingOutputLayer.forward(
      embeddingDropout.values,
    );
    final normalizedEmbedding = NormalizationResult.l2(embeddingLinear);

    final classifierHiddenLinear = classifierHiddenLayer.forward(
      normalizedEmbedding.values,
    );
    final classifierHiddenActivated = relu(classifierHiddenLinear);
    final classifierDropout = DropoutResult.apply(
      classifierHiddenActivated,
      rate: TargetNetworkConfig.dropoutRate,
      random: randomGenerator,
      training: training,
    );

    final logit = classifierOutputLayer
        .forward(classifierDropout.values)
        .single;

    return NetworkCache(
      input: input,
      embeddingHiddenLinear: embeddingHiddenLinear,
      embeddingDropout: embeddingDropout,
      embeddingLinear: embeddingLinear,
      normalizedEmbedding: normalizedEmbedding,
      classifierHiddenLinear: classifierHiddenLinear,
      classifierDropout: classifierDropout,
      logit: logit,
    );
  }

  void backwardNetwork(NetworkCache cache, double logitGradient) {
    var gradient = classifierOutputLayer.backward(
      input: cache.classifierDropout.values,
      outputGradient: [logitGradient],
    );
    gradient = cache.classifierDropout.backward(gradient);
    gradient = reluBackward(gradient, cache.classifierHiddenLinear);

    gradient = classifierHiddenLayer.backward(
      input: cache.normalizedEmbedding.values,
      outputGradient: gradient,
    );
    gradient = cache.normalizedEmbedding.backward(gradient);

    gradient = embeddingOutputLayer.backward(
      input: cache.embeddingDropout.values,
      outputGradient: gradient,
    );
    gradient = cache.embeddingDropout.backward(gradient);
    gradient = reluBackward(gradient, cache.embeddingHiddenLinear);

    embeddingHiddenLayer.backward(input: cache.input, outputGradient: gradient);
  }

  List<int> createBalancedBatch(
    List<int> positiveIndices,
    List<int> negativeIndices,
    int requestedBatchSize,
  ) {
    final half = math.max(1, requestedBatchSize ~/ 2);
    final batch = <int>[
      for (var index = 0; index < half; index++)
        positiveIndices[randomGenerator.nextInt(positiveIndices.length)],
      for (var index = 0; index < half; index++)
        negativeIndices[randomGenerator.nextInt(negativeIndices.length)],
    ];
    batch.shuffle(randomGenerator);
    return batch;
  }

  void validateFeatureVector(List<double> features) {
    if (features.length != TargetNetworkConfig.inputSize) {
      throw ArgumentError.value(
        features.length,
        'features',
        'Expected ${TargetNetworkConfig.inputSize} values',
      );
    }
  }

  void validateTrainingData(List<List<double>> samples, List<double> labels) {
    if (samples.isEmpty || samples.length != labels.length) {
      throw ArgumentError(
        'Samples and labels must be non-empty and equal in length',
      );
    }
    for (final sample in samples) {
      validateFeatureVector(sample);
    }
    if (labels.any((label) => label != 0 && label != 1)) {
      throw ArgumentError('Labels must contain only 0 or 1');
    }
    if (!labels.contains(0) || !labels.contains(1)) {
      throw ArgumentError('Training requires positive and negative samples');
    }
  }
}

class DenseLayer {
  DenseLayer._({
    required this.inputSize,
    required this.outputSize,
    required this.weights,
    required this.biases,
  });

  factory DenseLayer.random({
    required int inputSize,
    required int outputSize,
    required math.Random random,
  }) {
    final bound = 1 / math.sqrt(inputSize);
    List<double> initialize(int length) =>
        List.generate(length, (_) => (random.nextDouble() * 2 - 1) * bound);

    return DenseLayer._(
      inputSize: inputSize,
      outputSize: outputSize,
      weights: TrainableParameter(initialize(inputSize * outputSize)),
      biases: TrainableParameter(initialize(outputSize)),
    );
  }

  factory DenseLayer.fromWeights({
    required int inputSize,
    required int outputSize,
    required List<double> weights,
    required List<double> biases,
  }) {
    if (weights.length != inputSize * outputSize ||
        biases.length != outputSize) {
      throw ArgumentError(
        'Dense layer weight shape does not match its dimensions',
      );
    }
    return DenseLayer._(
      inputSize: inputSize,
      outputSize: outputSize,
      weights: TrainableParameter(List.of(weights)),
      biases: TrainableParameter(List.of(biases)),
    );
  }

  final int inputSize;
  final int outputSize;
  final TrainableParameter weights;
  final TrainableParameter biases;

  List<TrainableParameter> get parameters => [weights, biases];

  List<double> forward(List<double> input) {
    final output = List<double>.filled(outputSize, 0);
    for (var outputIndex = 0; outputIndex < outputSize; outputIndex++) {
      var sum = biases.values[outputIndex];
      final weightOffset = outputIndex * inputSize;
      for (var inputIndex = 0; inputIndex < inputSize; inputIndex++) {
        sum += weights.values[weightOffset + inputIndex] * input[inputIndex];
      }
      output[outputIndex] = sum;
    }
    return output;
  }

  List<double> backward({
    required List<double> input,
    required List<double> outputGradient,
  }) {
    final inputGradient = List<double>.filled(inputSize, 0);

    for (var outputIndex = 0; outputIndex < outputSize; outputIndex++) {
      final gradient = outputGradient[outputIndex];
      biases.gradients[outputIndex] += gradient;
      final weightOffset = outputIndex * inputSize;

      for (var inputIndex = 0; inputIndex < inputSize; inputIndex++) {
        final weightIndex = weightOffset + inputIndex;
        weights.gradients[weightIndex] += gradient * input[inputIndex];
        inputGradient[inputIndex] += gradient * weights.values[weightIndex];
      }
    }

    return inputGradient;
  }
}

/// Stores a parameter tensor together with its gradient and Adam state.
///
/// Keeping optimizer moments adjacent to their parameter simplifies model
/// transfer between the background training isolate and the UI isolate: only
/// the learned values are serialized after optimization completes.
class TrainableParameter {
  TrainableParameter(this.values)
    : gradients = List.filled(values.length, 0),
      firstMoments = List.filled(values.length, 0),
      secondMoments = List.filled(values.length, 0);

  final List<double> values;
  final List<double> gradients;
  final List<double> firstMoments;
  final List<double> secondMoments;

  void zeroGradients() => gradients.fillRange(0, gradients.length, 0);
}

class AdamOptimizer {
  AdamOptimizer({
    required this.parameters,
    required this.learningRate,
    required this.weightDecay,
  });

  final List<TrainableParameter> parameters;
  final double learningRate;
  final double weightDecay;

  static const beta1 = 0.9;
  static const beta2 = 0.999;
  static const epsilon = 1e-8;

  void zeroGradients() {
    for (final parameter in parameters) {
      parameter.zeroGradients();
    }
  }

  void step(int stepNumber) {
    // Bias correction is essential during short online training runs because
    // the first and second moment estimates begin at zero.
    final firstMomentCorrection = 1 - math.pow(beta1, stepNumber);
    final secondMomentCorrection = 1 - math.pow(beta2, stepNumber);

    for (final parameter in parameters) {
      for (var index = 0; index < parameter.values.length; index++) {
        final gradient =
            parameter.gradients[index] + weightDecay * parameter.values[index];
        parameter.firstMoments[index] =
            beta1 * parameter.firstMoments[index] + (1 - beta1) * gradient;
        parameter.secondMoments[index] =
            beta2 * parameter.secondMoments[index] +
            (1 - beta2) * gradient * gradient;

        final correctedFirstMoment =
            parameter.firstMoments[index] / firstMomentCorrection;
        final correctedSecondMoment =
            parameter.secondMoments[index] / secondMomentCorrection;
        parameter.values[index] -=
            learningRate *
            correctedFirstMoment /
            (math.sqrt(correctedSecondMoment) + epsilon);
      }
    }
  }
}

class DropoutResult {
  const DropoutResult(this.values, this.multipliers);

  factory DropoutResult.apply(
    List<double> input, {
    required double rate,
    required math.Random random,
    required bool training,
  }) {
    if (!training) {
      return DropoutResult(List.of(input), List.filled(input.length, 1));
    }

    final keepScale = 1 / (1 - rate);
    final multipliers = List.generate(
      input.length,
      (_) => random.nextDouble() < rate ? 0.0 : keepScale,
    );
    final values = List.generate(
      input.length,
      (index) => input[index] * multipliers[index],
    );
    return DropoutResult(values, multipliers);
  }

  final List<double> values;
  final List<double> multipliers;

  List<double> backward(List<double> gradient) => List.generate(
    gradient.length,
    (index) => gradient[index] * multipliers[index],
  );
}

/// Represents an L2-normalized embedding and its differentiable backward map.
///
/// For y = x / ||x||, the gradient is projected onto the tangent space of the
/// unit hypersphere before being scaled by the original vector norm.
class NormalizationResult {
  const NormalizationResult(this.values, this.norm);

  factory NormalizationResult.l2(List<double> input) {
    final norm =
        math.sqrt(input.fold<double>(0, (sum, value) => sum + value * value)) +
        1e-12;
    return NormalizationResult(
      input.map((value) => value / norm).toList(),
      norm,
    );
  }

  final List<double> values;
  final double norm;

  List<double> backward(List<double> gradient) {
    var projection = 0.0;
    for (var index = 0; index < gradient.length; index++) {
      projection += gradient[index] * values[index];
    }
    return List.generate(
      gradient.length,
      (index) => (gradient[index] - values[index] * projection) / norm,
    );
  }
}

class NetworkCache {
  const NetworkCache({
    required this.input,
    required this.embeddingHiddenLinear,
    required this.embeddingDropout,
    required this.embeddingLinear,
    required this.normalizedEmbedding,
    required this.classifierHiddenLinear,
    required this.classifierDropout,
    required this.logit,
  });

  final List<double> input;
  final List<double> embeddingHiddenLinear;
  final DropoutResult embeddingDropout;
  final List<double> embeddingLinear;
  final NormalizationResult normalizedEmbedding;
  final List<double> classifierHiddenLinear;
  final DropoutResult classifierDropout;
  final double logit;
}

Future<TargetNetwork> trainTargetNetworkInBackground(
  List<List<double>> samples,
  List<double> labels,
) async {
  // Dense backpropagation is CPU-intensive in pure Dart. Running it in a
  // dedicated isolate preserves camera throughput and prevents UI watchdog
  // timeouts. Only immutable samples and final weights cross isolate boundaries.
  final weights = await Isolate.run(() async {
    final network = TargetNetwork();
    await network.train(samples, labels);
    return network.exportWeights();
  });
  return TargetNetwork.fromWeights(weights);
}

List<double> relu(List<double> input) =>
    input.map((value) => math.max(0, value).toDouble()).toList();

List<double> reluBackward(List<double> gradient, List<double> linearOutput) =>
    List.generate(
      gradient.length,
      (index) => linearOutput[index] > 0 ? gradient[index] : 0,
    );

double sigmoid(double value) => 1 / (1 + math.exp(-value));
