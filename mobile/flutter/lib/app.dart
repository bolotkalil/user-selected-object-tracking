import 'package:camera/camera.dart';
import 'package:flutter/material.dart';

import 'app/providers/detector_service_provider.dart';
import 'presentation/pages/detector_page.dart';

/// Application shell. Concrete services are supplied by the composition root.
class UserSelectedObjectTrackingApp extends StatelessWidget {
  const UserSelectedObjectTrackingApp({
    super.key,
    required this.cameras,
    required this.services,
  });

  final List<CameraDescription> cameras;
  final DetectorServiceProvider services;

  @override
  Widget build(BuildContext context) => MaterialApp(
    debugShowCheckedModeBanner: false,
    theme: ThemeData.dark(useMaterial3: true),
    home: DetectorPage(cameras: cameras, services: services),
  );
}
