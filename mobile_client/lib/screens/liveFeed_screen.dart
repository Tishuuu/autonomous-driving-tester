import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../providers/sensor_provider.dart';

class LivefeedScreen extends StatefulWidget {
  const LivefeedScreen({super.key});

  @override
  State<LivefeedScreen> createState() => _LivefeedScreenState();
}

class _LivefeedScreenState extends State<LivefeedScreen> {
  bool _isTesting = false;

  @override
  void initState() {
    super.initState();
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeRight,
      DeviceOrientation.landscapeLeft,
    ]);
  }

  @override
  void dispose() {
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    super.dispose();
  }

  void _toggleTest() async {
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    setState(() {
      _isTesting = !_isTesting;
    });

    if (_isTesting) {
      print("🟢 Starting Real World Test...");
      await sensorProvider.startRealSensors();
    } else {
      print("🔴 Stopping Test...");
      sensorProvider.stopRealSensors();
    }
  }

  @override
  Widget build(BuildContext context) {
    final sensors = Provider.of<SensorProvider>(context);

    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF314972), Color(0xFF233452)],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Row(
                      children: [
                        Container(
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.1),
                            shape: BoxShape.circle,
                          ),
                          child: IconButton(
                            icon: const Icon(
                              Icons.arrow_back,
                              color: Colors.white,
                            ),
                            onPressed: () => Navigator.pop(context),
                          ),
                        ),
                        const SizedBox(width: 15),
                        Text(
                          "TESTER MODE",
                          style: GoogleFonts.lexend(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Colors.white,
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
                    ElevatedButton.icon(
                      onPressed: _toggleTest,
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _isTesting
                            ? Colors.redAccent
                            : const Color(0xFF3E7DEA),
                        foregroundColor: Colors.white,
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(20),
                        ),
                      ),
                      icon: Icon(_isTesting ? Icons.stop : Icons.play_arrow),
                      label: Text(_isTesting ? "STOP TEST" : "START TEST"),
                    ),
                  ],
                ),
                const SizedBox(height: 10),

                Expanded(
                  flex: 3,
                  child: Container(
                    width: double.infinity,
                    decoration: BoxDecoration(
                      color: Colors.black,
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(
                        color: _isTesting
                            ? Colors.green.withOpacity(0.5)
                            : Colors.white10,
                      ),
                    ),
                    child: Stack(
                      children: [
                        Center(
                          child: Text(
                            _isTesting
                                ? "Processing Video..."
                                : "Camera Standby",
                            style: const TextStyle(color: Colors.white54),
                          ),
                        ),
                        if (_isTesting)
                          Positioned(
                            top: 10,
                            right: 10,
                            child: Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: Colors.red,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: const Text(
                                "REC ●",
                                style: TextStyle(
                                  color: Colors.white,
                                  fontSize: 10,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                          ),
                      ],
                    ),
                  ),
                ),

                const SizedBox(height: 10),

                Expanded(
                  flex: 1,
                  child: Row(
                    children: [
                      _buildDataBox(
                        label: "SPEED",
                        value: sensors.speed.toStringAsFixed(0),
                        unit: "km/h",
                        color: Colors.blueAccent,
                      ),
                      const SizedBox(width: 10),

                      _buildDataBox(
                        label: "RPM",
                        value: sensors.rpm.toStringAsFixed(0),
                        unit: "rev/min",
                        color: Colors.orangeAccent,
                      ),
                      const SizedBox(width: 10),

                      _buildDataBox(
                        label: "G-FORCE",
                        value: sensors.totalGForce.toStringAsFixed(2),
                        unit: "g",
                        color: sensors.totalGForce > 1.2
                            ? Colors.red
                            : Colors.greenAccent,
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildDataBox({
    required String label,
    required String value,
    required String unit,
    required Color color,
  }) {
    return Expanded(
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1E293B).withOpacity(0.9),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: color.withOpacity(0.5), width: 1.5),
          boxShadow: [
            BoxShadow(
              color: color.withOpacity(0.1),
              blurRadius: 10,
              spreadRadius: 1,
            ),
          ],
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              label,
              style: GoogleFonts.lexend(
                color: Colors.white70,
                fontSize: 12,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              crossAxisAlignment: CrossAxisAlignment.baseline,
              textBaseline: TextBaseline.alphabetic,
              children: [
                Text(
                  value,
                  style: GoogleFonts.lexend(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(width: 4),
                Text(
                  unit,
                  style: GoogleFonts.lexend(
                    color: Colors.white54,
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
