import 'package:flutter/material.dart';

class CustomDrawer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          // Cabeçalho do Drawer
          UserAccountsDrawerHeader(
            accountName: const Text("Usuário"),
            accountEmail: const Text("usuario@exemplo.com"),
            currentAccountPicture: CircleAvatar(
              backgroundColor: Colors.blueAccent,
              child: const Icon(Icons.person, color: Colors.white),
            ),
            decoration: const BoxDecoration(
              color: Colors.blue,
            ),
          ),

          // Itens de navegação
          ListTile(
            leading: const Icon(Icons.settings),
            title: const Text('Configurações'),
            onTap: () {
              Navigator.pushNamed(context, '/settings');
            },
          ),
          ListTile(
            leading: const Icon(Icons.history),
            title: const Text('Histórico'),
            onTap: () {
              Navigator.pushNamed(context, '/history');
            },
          ),
          ListTile(
            leading: const Icon(Icons.help),
            title: const Text('Ajuda'),
            onTap: () {
              Navigator.pushNamed(context, '/help');
            },
          ),

          const Spacer(), // Espaço para o rodapé

          // Opção de sair (caso necessário)
          ListTile(
            leading: const Icon(Icons.exit_to_app),
            title: const Text('Sair'),
            onTap: () {
              // Lógica de logout, ou ação necessária
              Navigator.pop(context); // Fechar o Drawer
            },
          ),
        ],
      ),
    );
  }
}
