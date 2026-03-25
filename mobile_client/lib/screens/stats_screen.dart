import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import 'login_screen.dart';

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  @override
  Widget build(BuildContext context) {
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
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const SizedBox(height: 20),

                Align(
                  alignment: Alignment.topLeft,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Image.asset('assets/images/logo.webp', height: 60),
                      const SizedBox(height: 15),

                      Text("stats page"),

                      IconButton(
                        icon: const Icon(
                          Icons.cloud_upload,
                          color: Colors.blue,
                        ),
                        onPressed: () async {
                          final dummyTest = {
                            "student_id": "216312355",
                            "test_id":
                                "TEST-${DateTime.now().millisecondsSinceEpoch}",
                            "start_time": DateTime.now().toIso8601String(),
                            "final_score": 95.0,
                            "status": "passed",
                            "duration_seconds": 600,
                            "environment": {
                              "weather": "sunny",
                              "road_type": "urban",
                              "traffic_density": "low",
                            },
                            "metrics": {
                              "obd": {"max_speed": 50.0},
                              "imu": {"jerk_score": 10.0},
                              "camera": {"signs_missed": 0},
                              "gps_signal_quality": 100.0,
                            },
                            "events_log": [],
                          };

                          bool success = await ApiService.sendTestResult(
                            dummyTest,
                          );

                          ScaffoldMessenger.of(context).showSnackBar(
                            SnackBar(
                              content: Text(
                                success
                                    ? "Data sent to server! "
                                    : "Failed to connect ",
                              ),
                              backgroundColor: success
                                  ? Colors.green
                                  : Colors.red,
                            ),
                          );
                        },
                      ),
                      IconButton(
                        icon: const Icon(Icons.logout, color: Colors.redAccent),
                        onPressed: () async {
                          await Provider.of<UserProvider>(
                            context,
                            listen: false,
                          ).logout();

                          if (context.mounted) {
                            Navigator.of(context).pushAndRemoveUntil(
                              MaterialPageRoute(
                                builder: (context) => const LoginScreen(),
                              ),
                              (route) => false,
                            );
                          }
                        },
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
}
