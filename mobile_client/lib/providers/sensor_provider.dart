import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:math';
import 'dart:typed_data'; // הוספנו עבור המרת הבייטים של ה-OBD
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';

class SensorProvider extends ChangeNotifier {
  bool _isObdConnected = false;
  bool _isGpsLocked = false;
  bool _isCameraReady = false;
  bool _isMonitoring = false;

  double _speedKmh = 0.0;
  double _rpm = 0.0;
  double _lat = 0.0, _lon = 0.0;
  double _accelX = 0.0, _accelY = 0.0, _accelZ = 0.0;

  StreamSubscription<UserAccelerometerEvent>? _accelSubscription;
  StreamSubscription<Position>? _gpsSubscription;
  Timer? _loggingTimer;

  final List<Map<String, dynamic>> _sensorLog = [];
  DateTime? _testStartTime;

  // ==========================================
  // משתני מערכת OBD
  // ==========================================
  BluetoothConnection? _obdConnection;
  bool _isObdListening = false;

  bool get isObdConnected => _isObdConnected;
  bool get isCameraReady => _isCameraReady;
  bool get isSystemReady => _isCameraReady && _isGpsLocked;
  bool get isMonitoring => _isMonitoring;

  double get speed => _speedKmh;
  double get rpm => _rpm;
  double get totalGForce =>
      sqrt(pow(_accelX, 2) + pow(_accelY, 2) + pow(_accelZ, 2));

  // ==========================================
  // חיבור ל-OBD בלוטות' (ELM327)
  // ==========================================
  Future<bool> connectToOBD(BluetoothDevice device) async {
    try {
      print("🔌 Connecting to OBD: ${device.name}");
      _obdConnection = await BluetoothConnection.toAddress(device.address);
      _isObdConnected = true;
      _isObdListening = true;
      notifyListeners();

      // אתחול פרוטוקול ELM327
      _sendObdCommand("ATZ\r"); // Reset
      await Future.delayed(const Duration(milliseconds: 500));
      _sendObdCommand("ATL0\r"); // Linefeeds off
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand("ATSP0\r"); // Auto Protocol Search
      await Future.delayed(const Duration(milliseconds: 500));

      _listenToOBD(); // הפעלת האזנה ובקשות נתונים
      return true;
    } catch (e) {
      print("❌ OBD Connection Error: $e");
      _isObdConnected = false;
      notifyListeners();
      return false;
    }
  }

  // שליחת פקודת טקסט לרכב
  void _sendObdCommand(String command) {
    if (_obdConnection != null && _obdConnection!.isConnected) {
      _obdConnection!.output.add(utf8.encode(command));
    }
  }

  // לופ ששואל את הרכב לנתונים ומאזין לתשובות
  void _listenToOBD() async {
    // 1. הגדרת מאזין לתשובות שחוזרות מהרכב
    _obdConnection!.input!
        .listen((Uint8List data) {
          String response = utf8.decode(data);
          _parseObdResponse(response);
        })
        .onDone(() {
          print("⚠️ OBD Disconnected");
          _isObdConnected = false;
          _isObdListening = false;
          notifyListeners();
        });

    // 2. לופ בקשות אקטיבי כל עוד אנחנו מחוברים
    while (_isObdListening && _isObdConnected) {
      _sendObdCommand("010C\r"); // בקש סל"ד (RPM)
      await Future.delayed(const Duration(milliseconds: 150));
      _sendObdCommand("010D\r"); // בקש מהירות (Speed)
      await Future.delayed(const Duration(milliseconds: 150));
    }
  }

  // פענוח ה-Hex לנתונים עשרוניים
  void _parseObdResponse(String response) {
    response = response.replaceAll('\r', '').replaceAll('\n', '').trim();

    // פענוח סל"ד
    if (response.contains("41 0C")) {
      var parts = response.split(" ");
      int index = parts.indexOf("0C");
      if (index != -1 && parts.length >= index + 3) {
        try {
          int a = int.parse(parts[index + 1], radix: 16);
          int b = int.parse(parts[index + 2], radix: 16);
          _rpm = (a * 256 + b) / 4;
          notifyListeners();
        } catch (e) {}
      }
    }
    // פענוח מהירות
    else if (response.contains("41 0D")) {
      var parts = response.split(" ");
      int index = parts.indexOf("0D");
      if (index != -1 && parts.length >= index + 2) {
        try {
          _speedKmh = int.parse(parts[index + 1], radix: 16).toDouble();
          notifyListeners();
        } catch (e) {}
      }
    }
  }

  // ניתוק יזום של ה-OBD
  void disconnectOBD() {
    _isObdListening = false;
    _obdConnection?.dispose();
    _obdConnection = null;
    _isObdConnected = false;
    _speedKmh = 0.0;
    _rpm = 0.0;
    notifyListeners();
  }

  // ==========================================
  // התחלת טסט (חיישני פלאפון + שמירת OBD)
  // ==========================================
  Future<void> startRealSensors() async {
    if (_isMonitoring) return;

    print("🚀 Starting data logging...");

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) {
        print("❌ GPS Permissions denied");
        return;
      }
    }

    _isMonitoring = true;
    _isCameraReady = true;
    _testStartTime = DateTime.now();
    _sensorLog.clear();

    // 1. קריאת מד התאוצה (IMU)
    _accelSubscription =
        userAccelerometerEventStream(
          samplingPeriod: SensorInterval.gameInterval,
        ).listen((UserAccelerometerEvent event) {
          _accelX = event.x / 9.8;
          _accelY = event.y / 9.8;
          _accelZ = event.z / 9.8;
          notifyListeners();
        });

    // 2. קריאת ה-GPS
    _gpsSubscription =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.high,
            distanceFilter: 2,
          ),
        ).listen((Position position) {
          // נשתמש במהירות GPS רק אם ה-OBD לא מחובר!
          if (!_isObdConnected) {
            _speedKmh = position.speed * 3.6;
            if (_speedKmh < 0) _speedKmh = 0;
          }
          _lat = position.latitude;
          _lon = position.longitude;
          _isGpsLocked = true;
          notifyListeners();
        });

    // 3. מנגנון האגירה שיוצר את ה-JSON
    _loggingTimer = Timer.periodic(const Duration(milliseconds: 100), (_) {
      _logCurrentState();
    });

    notifyListeners();
  }

  void _logCurrentState() {
    if (_testStartTime == null) return;

    final int timeElapsedMs = DateTime.now()
        .difference(_testStartTime!)
        .inMilliseconds;

    _sensorLog.add({
      "time_ms": timeElapsedMs,
      "speed_kmh": double.parse(_speedKmh.toStringAsFixed(2)),
      "rpm": double.parse(_rpm.toStringAsFixed(2)),
      "lat": _lat,
      "lon": _lon,
      "g_force": double.parse(totalGForce.toStringAsFixed(2)),
      "accel": {
        "x": double.parse(_accelX.toStringAsFixed(3)),
        "y": double.parse(_accelY.toStringAsFixed(3)),
        "z": double.parse(_accelZ.toStringAsFixed(3)),
      },
    });
  }

  // ==========================================
  // עצירת טסט
  // ==========================================
  Future<String?> stopRealSensors() async {
    _accelSubscription?.cancel();
    _gpsSubscription?.cancel();
    _loggingTimer?.cancel();

    _accelSubscription = null;
    _gpsSubscription = null;
    _loggingTimer = null;

    _isMonitoring = false;
    print(
      "🛑 Real sensors stopped. Total data points logged: ${_sensorLog.length}",
    );

    String? filePath = await _exportDataToJson();

    // אנחנו לא מנתקים פה את ה-OBD, כדי שיישאר מחובר לטסט הבא
    if (!_isObdConnected) _speedKmh = 0;

    _accelX = 0;
    _accelY = 0;
    _accelZ = 0;
    _isGpsLocked = false;
    notifyListeners();

    return filePath;
  }

  // ==========================================
  // ייצוא נתונים
  // ==========================================
  Future<String?> _exportDataToJson() async {
    if (_sensorLog.isEmpty) return null;

    try {
      final directory = await getApplicationDocumentsDirectory();
      final String timestamp = DateTime.now().millisecondsSinceEpoch.toString();
      final File file = File('${directory.path}/test_data_$timestamp.json');

      final Map<String, dynamic> outputData = {
        "test_start_time": _testStartTime?.toIso8601String(),
        "total_duration_ms": _sensorLog.last["time_ms"],
        "data": _sensorLog,
      };

      await file.writeAsString(jsonEncode(outputData));
      print("✅ Data successfully saved to: ${file.path}");

      return file.path;
    } catch (e) {
      print("❌ Error saving data: $e");
      return null;
    }
  }
}
