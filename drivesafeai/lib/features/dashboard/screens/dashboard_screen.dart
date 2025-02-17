import 'package:flutter/material.dart';
import '../../../shared/widgets/custom_app_bar.dart';
import '../../../shared/widgets/custom_drawer.dart';
import '../widgets/start_monitoring_button.dart';
import '../widgets/camera_status_widget.dart';

class DashboardScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: CustomAppBar(title: 'Dashboard'),
      drawer: CustomDrawer(),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Spacer(),
            Image.asset('assets/images/logo.png', height: 150),
            const SizedBox(height: 30),
            CameraStatusWidget(), // Indicador de status da câmera
            const SizedBox(height: 20),
            StartMonitoringButton(), // Botão para iniciar o monitoramento
            const Spacer(),
            const Text(
              'A detecção será feita automaticamente usando a câmera frontal',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.grey),
            ),
            const SizedBox(height: 10),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextButton(
                  onPressed: () {
                    // Navegar para Configurações
                    Navigator.pushNamed(context, '/settings');
                  },
                  child: const Text('Configurações'),
                ),
                TextButton(
                  onPressed: () {
                    // Navegar para Ajuda
                    Navigator.pushNamed(context, '/help');
                  },
                  child: const Text('Ajuda'),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
