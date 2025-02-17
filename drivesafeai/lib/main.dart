import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'core/services/notification_service.dart';
import 'app.dart'; // Importando o app.dart para usar MyApp

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Inicializar serviços e permissões
  await initializeServices();

  // Inicializar câmeras
  final cameras = await availableCameras();
  final frontCamera = cameras.firstWhere(
    (camera) => camera.lensDirection == CameraLensDirection.front,
  );

  // Executar o app
  runApp(MyApp(camera: frontCamera));
}

Future<void> initializeServices() async {
  try {
    // Inicializar notificações
    await NotificationService().initialize();

    // Solicitar permissões de câmera e notificação
    await [
      Permission.camera,
      Permission.notification,
    ].request();
  } catch (e) {
    print('Erro ao inicializar serviços ou permissões: $e');
  }
}
