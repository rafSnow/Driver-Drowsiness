import 'package:flutter/material.dart';
import 'package:camera/camera.dart';

class MonitoringScreen extends StatefulWidget {
  final CameraDescription camera;

  const MonitoringScreen({super.key, required this.camera});

  @override
  State<MonitoringScreen> createState() => _MonitoringScreenState();
}

class _MonitoringScreenState extends State<MonitoringScreen> {
  late CameraController _controller;

  @override
  void initState() {
    super.initState();
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    _controller = CameraController(
      widget.camera,
      ResolutionPreset.medium,
      enableAudio: false,
    );
    try {
      await _controller.initialize();
      setState(() {}); // Atualiza a interface para exibir a câmera
    } catch (e) {
      debugPrint('Erro ao inicializar a câmera: $e');
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Monitoramento de Sonolência'),
      ),
      body: Center(
        child: _controller.value.isInitialized
            ? CameraPreviewWidget(
                controller: _controller, camera: widget.camera)
            : const CircularProgressIndicator(),
      ),
    );
  }
}

class CameraPreviewWidget extends StatelessWidget {
  final CameraController controller;
  final CameraDescription camera;

  const CameraPreviewWidget({
    super.key,
    required this.controller,
    required this.camera,
  });

  @override
  Widget build(BuildContext context) {
    // Verifica se o controller está inicializado antes de exibir a pré-visualização
    if (!controller.value.isInitialized) {
      return const Center(
        child: CircularProgressIndicator(),
      );
    }

    return AspectRatio(
      aspectRatio: controller.value.aspectRatio,
      child: Transform.rotate(
        angle: camera.sensorOrientation *
            3.14159 /
            180, // Ajusta a rotação com base na orientação do sensor
        child: CameraPreview(controller), // Exibe a pré-visualização da câmera
      ),
    );
  }
}
