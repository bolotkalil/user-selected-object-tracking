import 'package:flutter_test/flutter_test.dart';
import 'package:user_selected_object_tracking/detector.dart';

void main() {
  test('COCO class label is exposed by a detection', () {
    final detection = Detection([0, 0, 10, 10], 2, .9);
    expect(detection.label, 'car');
  });
}
