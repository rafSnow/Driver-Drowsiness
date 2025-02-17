import 'package:flutter/material.dart';

class CameraStatusWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Icon(Icons.camera_alt, size: 50, color: Colors.green),
        const SizedBox(height: 10),
        const Text(
          'Câmera Pronta',
          style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
        ),
      ],
    );
  }
}
