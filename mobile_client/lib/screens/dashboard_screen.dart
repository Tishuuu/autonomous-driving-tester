import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/user_provider.dart';
import '../providers/sensor_provider.dart';
import 'bluetooth_picker_screen.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  bool _loadingGps = false;
  bool _loadingCamera = false;
  bool _loadingImu = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runSystemCheck();
    });
  }

  Future<void> _runSystemCheck() async {
    // מקום לבדיקות מערכת בעתיד
  }

  // ===== הפעלת GPS בלבד =====
  Future<void> _connectGps() async {
    setState(() => _loadingGps = true);
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    // התיקון: בודקים אם יש האזנה פעילה, ולא רק אם יש נעילה
    if (sensorProvider.isGpsListening) {
      sensorProvider.stopGps();
    } else {
      await sensorProvider.startGps();
    }

    if (mounted) setState(() => _loadingGps = false);
    debugPrint('--- gps DEBUG ---');
    debugPrint(
      'gps Button Tapped. Current state: ${sensorProvider.isGpsListening}',
    );
    debugPrint('--- gps DEBUG ---');
  }

  // ===== הפעלת מצלמה בלבד =====
  Future<void> _connectCamera() async {
    setState(() => _loadingCamera = true);
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);
    if (sensorProvider.isCameraReady) {
      sensorProvider.deactivateCamera();
    } else {
      await sensorProvider.activateCamera();
    }

    if (mounted) setState(() => _loadingCamera = false);

    debugPrint('--- CAMERA DEBUG ---');
    debugPrint(
      'Camera Button Tapped. Current state: ${sensorProvider.isCameraReady}',
    );
    debugPrint('--- CAMERA DEBUG ---');
  }

  // ===== הפעלת IMU בלבד =====
  Future<void> _connectImu() async {
    setState(() => _loadingImu = true);
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    if (sensorProvider.isImuActive) {
      sensorProvider.stopImu();
    } else {
      sensorProvider.startImu();
    }

    // השהיה קצרה לפידבק ויזואלי
    await Future.delayed(const Duration(milliseconds: 300));
    if (mounted) setState(() => _loadingImu = false);
    debugPrint('--- IMU DEBUG ---');
    debugPrint(
      'IMU Button Tapped. Current state: ${sensorProvider.isImuActive}',
    );
    debugPrint('--- IMU DEBUG ---');
  }

  @override
  Widget build(BuildContext context) {
    final userProvider = Provider.of<UserProvider>(context);
    final sensorProvider = Provider.of<SensorProvider>(context);

    return Container(
      decoration: const BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [Color(0xFF314972), Color(0xFF233452)],
        ),
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        "Welcome back,",
                        style: GoogleFonts.lexend(
                          fontSize: 16,
                          color: Colors.white,
                        ),
                      ),
                      Text(
                        userProvider.user?.name ?? 'Driver',
                        style: GoogleFonts.lexend(
                          fontSize: 26,
                          fontWeight: FontWeight.bold,
                          color: const Color(0xFF3E7DEA),
                          shadows: [
                            const Shadow(
                              color: Color(0xFF3E7DEA),
                              blurRadius: 10,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  Image.asset('assets/images/logo.webp', height: 50),
                ],
              ),
              const SizedBox(height: 30),

              const Text(
                "SYSTEM DIAGNOSTICS",
                style: TextStyle(
                  color: Colors.white38,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                  fontSize: 12,
                ),
              ),
              const SizedBox(height: 15),

              Expanded(
                child: ListView(
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: _buildSensorCard(
                            "OBD-II",
                            Icons.settings_input_hdmi,
                            sensorProvider.isObdConnected,
                            false,
                            () async {
                              if (sensorProvider.isObdConnected) {
                                await sensorProvider.disconnectOBD();
                              } else {
                                Navigator.push(
                                  context,
                                  MaterialPageRoute(
                                    builder: (context) =>
                                        const BluetoothPickerScreen(),
                                  ),
                                );
                              }
                            },
                          ),
                        ),
                        const SizedBox(width: 15),

                        Expanded(
                          child: _buildSensorCard(
                            "GPS Lock",
                            Icons.gps_fixed,
                            sensorProvider.hasGpsFix,
                            _loadingGps ||
                                (sensorProvider.isGpsListening &&
                                    !sensorProvider.hasGpsFix),
                            () => _connectGps(),
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 15),

                    Row(
                      children: [
                        Expanded(
                          child: _buildSensorCard(
                            "Camera",
                            Icons.camera_alt,
                            sensorProvider.isCameraReady,
                            _loadingCamera,
                            () => _connectCamera(), // ✅ תוקן
                          ),
                        ),

                        const SizedBox(width: 15),

                        Expanded(
                          child: _buildSensorCard(
                            "IMU Sensor",
                            Icons.screen_rotation,
                            sensorProvider.isImuActive, // ✅ תוקן
                            _loadingImu,
                            () => _connectImu(), // ✅ תוקן
                          ),
                        ),
                      ],
                    ),
                    const SizedBox(height: 15),

                    _buildSensorCard(
                      "System Permissions",
                      Icons.security,
                      true,
                      false,
                      () {},
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSensorCard(
    String title,
    IconData icon,
    bool isConnected,
    bool isLoading,
    VoidCallback onTap,
  ) {
    final activeColor = const Color(0xFF00FF94);
    final errorColor = const Color(0xFFFF4C4C);
    final color = isConnected ? activeColor : errorColor;

    return GestureDetector(
      onTap: onTap,
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 300),
        height: 120,
        decoration: BoxDecoration(
          color: isConnected
              ? color.withOpacity(0.1)
              : Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(
            color: isConnected ? color.withOpacity(0.5) : Colors.white10,
            width: isConnected ? 1.5 : 1,
          ),
          boxShadow: isConnected
              ? [
                  BoxShadow(
                    color: color.withOpacity(0.2),
                    blurRadius: 10,
                    spreadRadius: 1,
                  ),
                ]
              : [],
        ),
        child: isLoading
            ? const Center(
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2,
                ),
              )
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    icon,
                    color: isConnected ? color : Colors.white38,
                    size: 30,
                  ),
                  const SizedBox(height: 10),
                  Text(
                    title,
                    style: const TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    isConnected ? "CONNECTED" : "TAP TO CONNECT",
                    style: TextStyle(
                      color: isConnected ? color : Colors.white38,
                      fontSize: 10,
                      letterSpacing: 1,
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}
