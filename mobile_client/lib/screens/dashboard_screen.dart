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
  final bool _loadingImu = false;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _runSystemCheck();
    });
  }

  Future<void> _runSystemCheck() async {
    // השארנו ריק כרגע, אפשר להוסיף בדיקות מערכת בעתיד
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);
  }

  Future<void> _connectGps() async {
    setState(() => _loadingGps = true);
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    await sensorProvider.startRealSensors();

    if (mounted) setState(() => _loadingGps = false);
  }

  Future<void> _connectCamera() async {
    setState(() => _loadingCamera = true);
    await Future.delayed(const Duration(milliseconds: 1500));
    if (mounted) setState(() => _loadingCamera = false);
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
                            () {
                              // ===== הלוגיקה החדשה של ה-OBD =====
                              if (sensorProvider.isObdConnected) {
                                // אם מחובר - ננתק
                                sensorProvider.disconnectOBD();
                              } else {
                                // אם מנותק - נפתח את מסך הבחירה
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
                            sensorProvider.speed > 0 ||
                                sensorProvider.isMonitoring,
                            _loadingGps,
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
                            () => _connectCamera(),
                          ),
                        ),
                        const SizedBox(width: 15),

                        Expanded(
                          child: _buildSensorCard(
                            "IMU Sensor",
                            Icons.screen_rotation,
                            sensorProvider.totalGForce > 0,
                            _loadingImu,
                            () => _connectGps(),
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
            ? Center(
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
