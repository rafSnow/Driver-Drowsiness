import 'package:camera/camera.dart';

class DashboardController {
  CameraController? controller;

  Future<void> initializeCamera(CameraDescription camera) async {
    controller = CameraController(camera, ResolutionPreset.medium);
    try {
      await controller?.initialize();
    } catch (e) {
      // Handle camera initialization error
    }
  }

  void dispose() {
    controller?.dispose();
  }
}
