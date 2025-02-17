import 'package:camera/camera.dart';
import 'package:drivesafeai/app.dart';
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    // Inicializa o binding para simular a inicialização de Widgets
    WidgetsFlutterBinding.ensureInitialized();

    // Obtenha a lista de câmeras disponíveis
    final cameras = await availableCameras();
    final firstCamera = cameras.first;

    // Build o app fornecendo a câmera
    await tester.pumpWidget(MyApp(camera: firstCamera));

    // Verifique se o texto inicial está correto
    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    // Simule um clique no botão '+'
    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    // Verifique se o contador foi incrementado
    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
