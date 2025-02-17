import 'package:flutter/material.dart';

class StartMonitoringButton extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return ElevatedButton.icon(
      onPressed: () {
        // Ação de iniciar o monitoramento
      },
      icon: const Icon(Icons.camera_alt),
      label: const Text('Iniciar Monitoramento'),
      style: ElevatedButton.styleFrom(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(8),
        ),
      ),
    );
  }
}
