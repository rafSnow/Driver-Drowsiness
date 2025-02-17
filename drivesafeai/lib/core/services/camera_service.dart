import 'package:camera/camera.dart';

class CameraService {
  // Singleton pattern
  static final CameraService _instance = CameraService._();
  factory CameraService() => _instance;
  CameraService._();

  CameraController? _controller;
  bool _isInitialized = false;

  // Getter para verificar se a câmera está inicializada
  bool get isInitialized => _isInitialized;

  // Getter para acessar o controller
  CameraController? get controller => _controller;

  // Inicializar a câmera
  Future<void> initialize(CameraDescription camera) async {
    if (_isInitialized) return;

    try {
      _controller = CameraController(
        camera,
        ResolutionPreset.medium,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );

      // Inicializar a câmera
      await _controller?.initialize();
      _isInitialized = true;

      // Iniciar stream de imagens
      await _startImageStream();
    } catch (e) {
      _isInitialized = false;
      _controller = null;
      throw Exception('Falha ao inicializar câmera: ${e.toString()}');
    }
  }

  // Iniciar a captura de imagens
  Future<void> _startImageStream() async {
    if (!_isInitialized || _controller == null) return;

    try {
      await _controller?.startImageStream((image) {
        _processImage(image);
      });
    } catch (e) {
      throw Exception('Falha ao iniciar stream de imagens: ${e.toString()}');
    }
  }

  // Processar a imagem para detectar sonolência (exemplo)
  void _processImage(CameraImage image) {
    // Implementar lógica de detecção de sonolência (exemplo)
  }

  // Pausar o stream de imagens
  Future<void> pauseCamera() async {
    if (!_isInitialized || _controller == null) return;

    try {
      if (_controller!.value.isStreamingImages) {
        await _controller?.stopImageStream();
      }
    } catch (e) {
      throw Exception('Falha ao pausar câmera: ${e.toString()}');
    }
  }

  // Retomar o stream de imagens
  Future<void> resumeCamera() async {
    if (!_isInitialized || _controller == null) return;

    try {
      await _startImageStream();
    } catch (e) {
      throw Exception('Falha ao resumir câmera: ${e.toString()}');
    }
  }

  // Alterar a resolução da câmera
  Future<void> changeCameraResolution(ResolutionPreset resolution) async {
    if (!_isInitialized || _controller == null) return;

    try {
      final description = _controller!.description;
      await dispose();
      _controller = CameraController(
        description,
        resolution,
        enableAudio: false,
        imageFormatGroup: ImageFormatGroup.jpeg,
      );
      await initialize(description);
    } catch (e) {
      throw Exception('Falha ao alterar resolução: ${e.toString()}');
    }
  }

  // Capturar uma foto
  Future<XFile?> takePicture() async {
    if (!_isInitialized || _controller == null) return null;

    try {
      final image = await _controller?.takePicture();
      return image;
    } catch (e) {
      throw Exception('Falha ao capturar imagem: ${e.toString()}');
    }
  }

  // Liberar recursos da câmera
  Future<void> dispose() async {
    if (!_isInitialized || _controller == null) return;

    try {
      if (_controller!.value.isStreamingImages) {
        await _controller?.stopImageStream();
      }
      await _controller?.dispose();
      _controller = null;
      _isInitialized = false;
    } catch (e) {
      throw Exception('Falha ao liberar recursos da câmera: ${e.toString()}');
    }
  }
}

// Extensão para facilitar o uso do serviço
extension CameraServiceExtension on CameraService {
  bool get isStreaming =>
      _isInitialized &&
      _controller != null &&
      _controller!.value.isStreamingImages;
}
