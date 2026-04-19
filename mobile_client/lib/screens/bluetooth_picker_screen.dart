import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import '../providers/sensor_provider.dart';
import 'package:permission_handler/permission_handler.dart';

class BluetoothPickerScreen extends StatefulWidget {
  const BluetoothPickerScreen({super.key});

  @override
  State<BluetoothPickerScreen> createState() => _BluetoothPickerScreenState();
}

class _BluetoothPickerScreenState extends State<BluetoothPickerScreen> {
  List<BluetoothDevice> _devices = [];
  bool _isDiscovering = true;
  bool _isConnecting = false;

  @override
  void initState() {
    super.initState();
    _loadPairedDevices();
  }

  // שואב מהטלפון את כל מכשירי הבלוטות' שכבר עשינו להם צימוד
  // הפונקציה המעודכנת שמבקשת הרשאות לפני הסריקה
  void _loadPairedDevices() async {
    try {
      // 1. בקשת הרשאות מאנדרואיד
      Map<Permission, PermissionStatus> statuses = await [
        Permission.bluetooth,
        Permission.bluetoothConnect,
        Permission.bluetoothScan,
        Permission.location, // לעיתים בלוטות' דורש גם מיקום דלוק
      ].request();

      // 2. בדיקה אם המשתמש אישר הכל
      if (statuses[Permission.bluetoothConnect]!.isGranted) {
        // 3. שאיבת המכשירים
        List<BluetoothDevice> pairedDevices = await FlutterBluetoothSerial
            .instance
            .getBondedDevices();

        setState(() {
          _devices = pairedDevices;
          _isDiscovering = false;
        });
      } else {
        print("❌ Bluetooth permissions denied by user.");
        setState(() {
          _isDiscovering = false;
        });
      }
    } catch (e) {
      print("❌ Error loading devices: $e");
      setState(() {
        _isDiscovering = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: const Color(0xFF1E1E1E), // רקע כהה שמתאים לאפליקציה
      appBar: AppBar(
        backgroundColor: const Color(0xFF2A2A2A),
        title: Text(
          "Select OBD Device",
          style: GoogleFonts.lexend(
            color: Colors.white,
            fontWeight: FontWeight.bold,
          ),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: _isDiscovering
          ? const Center(
              child: CircularProgressIndicator(color: Colors.blueAccent),
            )
          : _devices.isEmpty
          ? Center(
              child: Text(
                "No paired devices found.\nPlease pair your iCar Pro in Phone Settings first.",
                textAlign: TextAlign.center,
                style: GoogleFonts.lexend(color: Colors.white70, fontSize: 16),
              ),
            )
          : ListView.builder(
              padding: const EdgeInsets.all(12),
              itemCount: _devices.length,
              itemBuilder: (context, index) {
                BluetoothDevice device = _devices[index];
                return Card(
                  color: const Color(0xFF2A2A2A),
                  elevation: 4,
                  margin: const EdgeInsets.symmetric(vertical: 8),
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(15),
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.symmetric(
                      horizontal: 16,
                      vertical: 8,
                    ),
                    leading: Container(
                      padding: const EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: Colors.blueAccent.withOpacity(0.2),
                        shape: BoxShape.circle,
                      ),
                      child: const Icon(
                        Icons.bluetooth,
                        color: Colors.blueAccent,
                        size: 28,
                      ),
                    ),
                    title: Text(
                      device.name ?? "Unknown Device",
                      style: GoogleFonts.lexend(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 16,
                      ),
                    ),
                    subtitle: Text(
                      device.address,
                      style: GoogleFonts.lexend(
                        color: Colors.white54,
                        fontSize: 12,
                      ),
                    ),
                    trailing: _isConnecting
                        ? const SizedBox(
                            width: 24,
                            height: 24,
                            child: CircularProgressIndicator(
                              color: Colors.greenAccent,
                              strokeWidth: 2,
                            ),
                          )
                        : ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: const Color(
                                0xFF00FF94,
                              ).withOpacity(0.2),
                              foregroundColor: const Color(0xFF00FF94),
                              elevation: 0,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(10),
                                side: const BorderSide(
                                  color: Color(0xFF00FF94),
                                ),
                              ),
                            ),
                            onPressed: () async {
                              setState(() => _isConnecting = true);

                              // קריאה ל-Provider שהכנו קודם כדי להתחבר!
                              bool success = await context
                                  .read<SensorProvider>()
                                  .connectToOBD(device);

                              if (mounted) {
                                setState(() => _isConnecting = false);
                                if (success) {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text("Connected to OBD! ✅"),
                                      backgroundColor: Colors.green,
                                    ),
                                  );
                                  Navigator.pop(
                                    context,
                                  ); // סוגר את המסך וחוזר ל-Dashboard
                                } else {
                                  ScaffoldMessenger.of(context).showSnackBar(
                                    const SnackBar(
                                      content: Text("Connection Failed ❌"),
                                      backgroundColor: Colors.red,
                                    ),
                                  );
                                }
                              }
                            },
                            child: Text(
                              "CONNECT",
                              style: GoogleFonts.lexend(
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ),
                  ),
                );
              },
            ),
    );
  }
}
