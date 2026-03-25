import 'dart:async';
import 'dart:math';
import 'package:flutter/material.dart';
import 'package:sensors_plus/sensors_plus.dart';
import 'package:geolocator/geolocator.dart';

class SensorProvider extends ChangeNotifier {
  bool _isObdConnected = false;
  bool _isGpsLocked = false;
  bool _isCameraReady = false;
  bool _isMonitoring = false;

  double _speedKmh = 0.0;
  final double _rpm = 0.0;

  double _accelX = 0.0, _accelY = 0.0, _accelZ = 0.0;

  StreamSubscription<UserAccelerometerEvent>? _accelSubscription;
  StreamSubscription<Position>? _gpsSubscription;

  bool get isObdConnected => _isObdConnected;
  bool get isCameraReady => _isCameraReady;
  bool get isSystemReady => _isCameraReady && _isGpsLocked;
  bool get isMonitoring => _isMonitoring;

  double get speed => _speedKmh;
  double get rpm => _rpm;

  double get totalGForce =>
      sqrt(pow(_accelX, 2) + pow(_accelY, 2) + pow(_accelZ, 2));

  Future<void> startRealSensors() async {
    if (_isMonitoring) return;

    print("🔌 Connecting to REAL sensors...");

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

    _accelSubscription =
        userAccelerometerEventStream(
          samplingPeriod: SensorInterval.gameInterval,
        ).listen((UserAccelerometerEvent event) {
          _accelX = event.x / 9.8;
          _accelY = event.y / 9.8;
          _accelZ = event.z / 9.8;
          notifyListeners();
        });

    _gpsSubscription =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.high,
            distanceFilter: 2,
          ),
        ).listen((Position position) {
          _speedKmh = position.speed * 3.6;
          _isGpsLocked = true;
          if (_speedKmh < 0) _speedKmh = 0;

          print("📍 GPS Update: Speed=${_speedKmh.toStringAsFixed(1)} km/h");
          notifyListeners();
        });

    notifyListeners();
  }

  void stopRealSensors() {
    _accelSubscription?.cancel();
    _gpsSubscription?.cancel();
    _accelSubscription = null;
    _gpsSubscription = null;

    _isMonitoring = false;
    _speedKmh = 0;
    _accelX = 0;
    _accelY = 0;
    _accelZ = 0;
    _isGpsLocked = false;

    print("🛑 Real sensors stopped.");
    notifyListeners();
  }

  void updateObdStatus(bool status) {
    _isObdConnected = status;
    notifyListeners();
  }
}
