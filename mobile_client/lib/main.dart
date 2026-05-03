import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'screens/login_screen.dart';
import 'providers/user_provider.dart';
import 'screens/home_screen.dart';
import 'providers/sensor_provider.dart';

void main() {
  WidgetsFlutterBinding.ensureInitialized();

  // ✅ הפונטים נטענים מ-assets/fonts/ דרך pubspec.yaml.
  // נכבה הורדה מהרשת כדי למנוע ניסיונות חוזרים שחוסמים את ה-UI thread.
  GoogleFonts.config.allowRuntimeFetching = false;

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider(create: (_) => UserProvider()),
        ChangeNotifierProvider(create: (_) => SensorProvider()),
      ],
      child: const MyApp(),
    ),
  );
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Auto Tester',
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const AuthCheckWrapper(),
    );
  }
}

class AuthCheckWrapper extends StatefulWidget {
  const AuthCheckWrapper({super.key});

  @override
  State<AuthCheckWrapper> createState() => _AuthCheckWrapperState();
}

class _AuthCheckWrapperState extends State<AuthCheckWrapper> {
  @override
  void initState() {
    super.initState();
    Future.microtask(
      () => Provider.of<UserProvider>(context, listen: false).tryAutoLogin(),
    );
  }

  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);

    if (userProvider.isLogged) {
      return const HomeScreen();
    }

    return const LoginScreen();
  }
}
