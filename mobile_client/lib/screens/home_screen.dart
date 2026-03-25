import 'package:flutter/material.dart';
import 'package:mobile_client/providers/sensor_provider.dart';
import 'package:provider/provider.dart';
import 'dashboard_screen.dart';
import 'history_screen.dart';
import 'settings_screen.dart';
import 'stats_screen.dart';
import 'package:google_fonts/google_fonts.dart';
import 'liveFeed_screen.dart';

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final Color _primaryColor = const Color(0xFF3E7DEA);

  final bool _isObdConnected = false;
  final bool _isCameraReady = false;
  final bool _isGpsLocked = false;
  bool get _isSystemReady => _isObdConnected && _isCameraReady && _isGpsLocked;
  final List<Widget> _pages = [
    const DashboardScreen(),
    const HistoryScreen(),
    const StatsScreen(),
    const SettingsScreen(),
  ];
  Color get _mainButtonColor {
    if (_isSystemReady) {
      return _primaryColor;
    }
    return const Color(0xFFFF4C4C);
  }

  int _selectedIndex = 0;

  void _onItemTapped(int index) {
    setState(() {
      _selectedIndex = index;
    });
  }

  Widget _buildNavIcon(IconData icon, String label, int index) {
    final bool isSelected = _selectedIndex == index;
    final Color iconColor = isSelected ? const Color(0xFF3E7DEA) : Colors.grey;

    return Tooltip(
      message: label,
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          customBorder: const CircleBorder(),

          splashFactory: InkRipple.splashFactory,

          onTap: () {
            setState(() {
              _selectedIndex = index;
            });
          },

          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: 12.0,
              vertical: 4.0,
            ),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(icon, color: iconColor, size: 24),
                const SizedBox(height: 2),
                Text(
                  label,
                  style: GoogleFonts.poppins(
                    color: iconColor,
                    fontSize: 10,
                    fontWeight: FontWeight.normal,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButtonLocation: FloatingActionButtonLocation.centerDocked,
      extendBody: true,
      floatingActionButton: Consumer<SensorProvider>(
        builder: (context, sensor, child) {
          final bool isReady = sensor.isSystemReady;

          final Color mainButtonColor = isReady
              ? const Color(0xFF3E7DEA)
              : const Color(0xFFFF4C4C);

          return Container(
            height: 70,

            width: 70,

            decoration: BoxDecoration(
              shape: BoxShape.circle,

              boxShadow: [
                BoxShadow(
                  color: mainButtonColor.withOpacity(0.5),

                  blurRadius: 20,

                  spreadRadius: 2,
                ),
              ],
            ),

            child: FloatingActionButton(
              onPressed: () {
                if (isReady) {
                  print("System Ready");

                  Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const LivefeedScreen()),
                  );
                } else {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(
                      content: Text("Please make sure to connect all sensors"),
                      backgroundColor: Colors.red,
                    ),
                  );
                }
              },

              backgroundColor: mainButtonColor,

              elevation: 0,

              shape: const CircleBorder(),

              child: Icon(Icons.bluetooth, size: 35, color: Colors.white),
            ),
          );
        },
      ),

      bottomNavigationBar: BottomAppBar(
        color: const Color(0xFF0F172A),
        shape: const CircularNotchedRectangle(),
        notchMargin: 8.0,
        child: SizedBox(
          height: 60,
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceAround,

            children: [
              _buildNavIcon(Icons.home_filled, "Home", 0),
              _buildNavIcon(Icons.history, "History", 1),
              const SizedBox(width: 40),
              _buildNavIcon(Icons.bar_chart_rounded, "Stats", 2),
              _buildNavIcon(Icons.settings, "Settings", 3),
            ],
          ),
        ),
      ),

      body: _pages[_selectedIndex],
    );
  }
}
