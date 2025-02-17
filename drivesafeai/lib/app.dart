import 'package:flutter/material.dart';
import 'features/dashboard/screens/dashboard_screen.dart';
import 'package:camera/camera.dart';

class MyApp extends StatelessWidget {
  final CameraDescription camera;

  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Detecção de Sonolência',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: DashboardScreen(camera: camera),
    );
  }
}
