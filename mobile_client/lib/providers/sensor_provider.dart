import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';
import 'package:crypto/crypto.dart';
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

  // 🆕 Nullable GPS — guarantees no 0,0 leak into dataset
  double? _lat;
  double? _lon;
  bool _hasGpsFix = false;
  DateTime? _lastFixAt;

  double _accelX = 0.0, _accelY = 0.0, _accelZ = 0.0;

  StreamSubscription<UserAccelerometerEvent>? _accelSubscription;
  StreamSubscription<Position>? _gpsSubscription;
  Timer? _loggingTimer;

  final List<Map<String, dynamic>> _sensorLog = [];
  DateTime? _testStartTime;
  DateTime? _videoStartTime;

  // OBD
  BluetoothConnection? _obdConnection;
  StreamSubscription<Uint8List>? _obdStreamSub;
  bool _isObdListening = false;
  Timer? _obdPollTimer;
  DateTime? _lastObdResponseTime;
  static const Duration _obdStaleThreshold = Duration(seconds: 2);
  bool _obdIsStale = false;

  // 🆕 Isolate parser
  Isolate? _parserIsolate;
  SendPort? _parserSend;
  ReceivePort? _parserRecv;

  // RTT
  Timer? _rttPingTimer;
  DateTime? _lastRttPingSent;
  double _obdLatencyMs = 0;
  final List<double> _rttHistory = [];
  bool _waitingForRttResponse = false;

  // Sensor buffer
  static const int _bufferMaxSeconds = 5;
  final List<_TimedSensorSample> _sensorBuffer = [];

  // IMU stop signature
  bool _imuStopDetected = false;
  DateTime? _imuStopAt;
  final List<_TimedAccelSample> _accelBuffer = [];

  // Tier
  int _activeTier = 3;
  int get activeTier => _activeTier;

  // Getters
  bool get isGpsListening => _gpsSubscription != null;
  bool get isObdConnected => _isObdConnected && !_obdIsStale;
  bool get isGpsLocked => _isGpsLocked;
  bool get hasGpsFix => _hasGpsFix; // 🆕
  bool get isCameraReady => _isCameraReady;
  bool get isImuActive => _accelSubscription != null;
  bool get isSystemReady => isCameraReady && hasGpsFix && isImuActive;
  bool get isMonitoring => _isMonitoring;

  double get speed => _speedKmh;
  double get rpm => _rpm;
  double? get lat => _lat;
  double? get lon => _lon;
  double get totalGForce =>
      sqrt(pow(_accelX, 2) + pow(_accelY, 2) + pow(_accelZ, 2));
  double get obdLatencyMs => _obdLatencyMs;

  String get speedSourceLabel {
    switch (_activeTier) {
      case 1:
        return "OBD";
      case 2:
        return "IMU";
      case 3:
        return "GPS";
      default:
        return "—";
    }
  }

  // ==========================================
  // OBD Connect
  // ==========================================
  Future<bool> connectToOBD(BluetoothDevice device) async {
    debugPrint("[OBD] 🔌 Connecting to ${device.name}");
    try {
      _obdConnection = await BluetoothConnection.toAddress(device.address);
      _isObdConnected = true;
      _isObdListening = true;
      _obdIsStale = false;
      _lastObdResponseTime = DateTime.now();
      notifyListeners();

      await _spawnParserIsolate();
      _startObdStream();

      _sendObdCommand("ATZ\r");
      await Future.delayed(const Duration(milliseconds: 800));
      _sendObdCommand("ATE0\r");
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand("ATL0\r");
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand("ATS0\r");
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand("ATSP0\r");
      await Future.delayed(const Duration(milliseconds: 500));

      _startPollingLoop();
      _startRttMeasurementLoop();
      return true;
    } catch (e) {
      debugPrint("[OBD] ❌ Connection failed: $e");
      _isObdConnected = false;
      notifyListeners();
      return false;
    }
  }

  void _sendObdCommand(String command) {
    try {
      if (_obdConnection == null || !_obdConnection!.isConnected) return;
      _obdConnection!.output.add(utf8.encode(command));
    } catch (e) {
      debugPrint("[OBD] ❌ Send error: $e");
    }
  }

  // ==========================================
  // 🆕 Spawn long-lived parser isolate
  // ==========================================
  Future<void> _spawnParserIsolate() async {
    _parserRecv = ReceivePort();
    _parserIsolate = await Isolate.spawn(
      _obdParserEntry,
      _parserRecv!.sendPort,
    );

    final completer = Completer<SendPort>();
    _parserRecv!.listen((msg) {
      if (msg is SendPort && !completer.isCompleted) {
        completer.complete(msg);
        return;
      }
      if (msg is Map) {
        _handleParsedFromIsolate(msg);
      }
    });
    _parserSend = await completer.future;
  }

  void _handleParsedFromIsolate(Map msg) {
    final type = msg['type'];
    if (type == 'rtt_response') {
      if (_waitingForRttResponse && _lastRttPingSent != null) {
        final rtt = DateTime.now()
            .difference(_lastRttPingSent!)
            .inMilliseconds
            .toDouble();
        _waitingForRttResponse = false;
        _updateRttHistory(rtt);
      }
      return;
    }
    if (type == 'speed') {
      final v = (msg['value'] as num).toDouble();
      if (v >= 0 && v <= 300) {
        _speedKmh = v;
        _activeTier = 1;
        if (v == 0) _imuStopDetected = false;
        _addCorrectedSample(v);
        notifyListeners();
      }
    } else if (type == 'rpm') {
      final v = (msg['value'] as num).toDouble();
      if (v >= 0 && v <= 12000) {
        _rpm = v;
        notifyListeners();
      }
    }
  }

  void _startObdStream() {
    _obdStreamSub?.cancel();
    _obdStreamSub = _obdConnection!.input!.listen(
      (Uint8List data) {
        _lastObdResponseTime = DateTime.now();
        if (_obdIsStale) {
          _obdIsStale = false;
          notifyListeners();
        }
        _parserSend?.send(data); // hand off, return immediately
      },
      onError: (e) {
        debugPrint("[OBD] ❌ Stream error: $e");
        _handleDisconnect();
      },
      onDone: _handleDisconnect,
      cancelOnError: false,
    );
  }

  // ==========================================
  // Polling loop
  // ==========================================
  void _startPollingLoop() {
    _obdPollTimer?.cancel();
    _obdPollTimer = Timer.periodic(const Duration(milliseconds: 300), (timer) {
      if (!_isObdListening || !_isObdConnected) {
        timer.cancel();
        return;
      }
      if (_lastObdResponseTime != null) {
        final since = DateTime.now().difference(_lastObdResponseTime!);
        if (since > _obdStaleThreshold && !_obdIsStale) {
          _obdIsStale = true;
          notifyListeners();
        }
      }
      try {
        _sendObdCommand("010C\r");
        Future.delayed(const Duration(milliseconds: 130), () {
          if (_isObdListening && _isObdConnected) _sendObdCommand("010D\r");
        });
      } catch (e) {
        debugPrint("[OBD] ❌ Poll error: $e");
      }
    });
  }

  void _startRttMeasurementLoop() {
    _rttPingTimer?.cancel();
    _rttPingTimer = Timer.periodic(const Duration(seconds: 1), (timer) {
      if (!_isObdListening || !_isObdConnected || _obdIsStale) {
        timer.cancel();
        return;
      }
      if (_waitingForRttResponse) _waitingForRttResponse = false;
      _lastRttPingSent = DateTime.now();
      _waitingForRttResponse = true;
      _sendObdCommand("ATRV\r");
    });
  }

  void _updateRttHistory(double rtt) {
    _rttHistory.add(rtt);
    if (_rttHistory.length > 10) _rttHistory.removeAt(0);
    _obdLatencyMs = _rttHistory.reduce((a, b) => a + b) / _rttHistory.length;
  }

  void _addCorrectedSample(double speedKmh) {
    final correctionMs = (_obdLatencyMs / 2).round();
    final correctedTime = DateTime.now().subtract(
      Duration(milliseconds: correctionMs),
    );
    _sensorBuffer.add(
      _TimedSensorSample(
        time: correctedTime,
        speedKmh: speedKmh,
        source: 'OBD',
      ),
    );
    final cutoff = DateTime.now().subtract(
      Duration(seconds: _bufferMaxSeconds),
    );
    _sensorBuffer.removeWhere((s) => s.time.isBefore(cutoff));
  }

  // IMU stop signature
  void _checkImuStopSignature() {
    final cutoff = DateTime.now().subtract(const Duration(milliseconds: 1500));
    _accelBuffer.removeWhere((s) => s.time.isBefore(cutoff));
    if (_accelBuffer.length < 8) return;

    int decelCount = 0;
    for (int i = 0; i < _accelBuffer.length - 3; i++) {
      if (_accelBuffer[i].accelY < -0.1) decelCount++;
    }
    final wasDecelerating = decelCount >= 4;

    final recent = _accelBuffer.sublist(_accelBuffer.length - 3);
    final avgRecent =
        recent.map((s) => s.accelY.abs()).reduce((a, b) => a + b) / 3;
    final nowAtZero = avgRecent < 0.05;

    if (wasDecelerating && nowAtZero && !_imuStopDetected) {
      _imuStopDetected = true;
      _imuStopAt = DateTime.now();
      if (!_isObdConnected || _obdIsStale) {
        _activeTier = 2;
        _speedKmh = 0;
        notifyListeners();
      }
    }
    if (avgRecent > 0.15) _imuStopDetected = false;
  }

  void _handleDisconnect() {
    if (!_isObdConnected) return;
    _isObdListening = false;
    _isObdConnected = false;
    _obdIsStale = false;
    _speedKmh = 0;
    _rpm = 0;
    _obdPollTimer?.cancel();
    _obdPollTimer = null;
    _rttPingTimer?.cancel();
    _rttPingTimer = null;
    _waitingForRttResponse = false;
    _activeTier = _isGpsLocked ? 3 : 2;
    notifyListeners();
  }

  Future<void> disconnectOBD() async {
    _isObdListening = false;
    _obdPollTimer?.cancel();
    _rttPingTimer?.cancel();
    await _obdStreamSub?.cancel();
    _obdStreamSub = null;
    try {
      await _obdConnection?.close();
      _obdConnection?.dispose();
    } catch (_) {}
    _obdConnection = null;

    _parserIsolate?.kill(priority: Isolate.immediate);
    _parserIsolate = null;
    _parserRecv?.close();
    _parserRecv = null;
    _parserSend = null;

    _isObdConnected = false;
    _obdIsStale = false;
    _speedKmh = 0;
    _rpm = 0;
    _waitingForRttResponse = false;
    _activeTier = _isGpsLocked ? 3 : 2;
    _obdLatencyMs = 0;
    notifyListeners();
  }

  // ==========================================
  // IMU
  // ==========================================
  void startImu() {
    if (_accelSubscription != null) return;
    _accelSubscription =
        userAccelerometerEventStream(
          samplingPeriod: SensorInterval.gameInterval,
        ).listen((UserAccelerometerEvent event) {
          _accelX = event.x / 9.8;
          _accelY = event.y / 9.8;
          _accelZ = event.z / 9.8;
          _accelBuffer.add(
            _TimedAccelSample(
              time: DateTime.now(),
              accelX: _accelX,
              accelY: _accelY,
              accelZ: _accelZ,
            ),
          );
          if (_accelBuffer.length > 100) _accelBuffer.removeAt(0);
          _checkImuStopSignature();
          notifyListeners();
        });
    notifyListeners();
  }

  void stopImu() {
    _accelSubscription?.cancel();
    _accelSubscription = null;
    _accelX = 0;
    _accelY = 0;
    _accelZ = 0;
    notifyListeners();
  }

  // ==========================================
  // GPS — fix gate prevents 0,0 race
  // ==========================================
  Future<bool> startGps() async {
    // Allow re-prime if previous run never got a fix
    if (_gpsSubscription != null && _hasGpsFix) return true;
    if (_gpsSubscription != null && !_hasGpsFix) {
      await _gpsSubscription!.cancel();
      _gpsSubscription = null;
    }

    if (!await Geolocator.isLocationServiceEnabled()) return false;
    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.denied) return false;
    }
    if (permission == LocationPermission.deniedForever) return false;

    try {
      final lastKnown = await Geolocator.getLastKnownPosition();
      if (lastKnown != null) {
        _lat = lastKnown.latitude;
        _lon = lastKnown.longitude;
        _hasGpsFix = true;
        _isGpsLocked = true;
        _lastFixAt = DateTime.now();
        notifyListeners();
      }
    } catch (_) {}

    try {
      final current = await Geolocator.getCurrentPosition(
        locationSettings: const LocationSettings(
          accuracy: LocationAccuracy.medium,
          timeLimit: Duration(seconds: 3),
        ),
      );
      _lat = current.latitude;
      _lon = current.longitude;
      _hasGpsFix = true;
      _isGpsLocked = true;
      _lastFixAt = DateTime.now();
      notifyListeners();
    } catch (_) {}

    _gpsSubscription =
        Geolocator.getPositionStream(
          locationSettings: const LocationSettings(
            accuracy: LocationAccuracy.medium,
            distanceFilter: 0,
          ),
        ).listen((Position position) {
          if (!_isObdConnected || _obdIsStale) {
            if (!_imuStopDetected) {
              _speedKmh = max(0, position.speed * 3.6);
              _activeTier = 3;
            }
          }
          _lat = position.latitude;
          _lon = position.longitude;
          _hasGpsFix = true;
          _isGpsLocked = true;
          _lastFixAt = DateTime.now();
          notifyListeners();
        }, onError: (e) => debugPrint("[GPS] ❌ Stream error: $e"));

    notifyListeners();
    return _hasGpsFix;
  }

  void stopGps() {
    _gpsSubscription?.cancel();
    _gpsSubscription = null;
    _isGpsLocked = false;
    notifyListeners();
  }

  // Camera
  Future<bool> activateCamera() async {
    await Future.delayed(const Duration(milliseconds: 500));
    _isCameraReady = true;
    notifyListeners();
    return true;
  }

  void deactivateCamera() {
    _isCameraReady = false;
    notifyListeners();
  }

  // ==========================================
  // Test recording
  // ==========================================
  Future<void> startRealSensors() async {
    if (_isMonitoring) return;
    if (_accelSubscription == null) startImu();
    if (_gpsSubscription == null || !_hasGpsFix) await startGps();
    if (!_isCameraReady) await activateCamera();

    _isMonitoring = true;
    _testStartTime = DateTime.now();
    _videoStartTime = null;
    _sensorLog.clear();

    _loggingTimer = Timer.periodic(
      const Duration(milliseconds: 100),
      (_) => _logCurrentState(),
    );
    notifyListeners();
  }

  void markVideoStart() {
    _videoStartTime = DateTime.now();
  }

  // 🆕 Skip rows until GPS fix lands
  void _logCurrentState() {
    if (_testStartTime == null) return;
    if (!_hasGpsFix || _lat == null || _lon == null) return;

    final t = DateTime.now().difference(_testStartTime!).inMilliseconds;
    _sensorLog.add({
      "time_ms": t,
      "speed_kmh": double.parse(_speedKmh.toStringAsFixed(2)),
      "rpm": double.parse(_rpm.toStringAsFixed(2)),
      "lat": _lat,
      "lon": _lon,
      "speed_source": _activeTier, // 🆕 Tier tag for server-side smoothing
      "g_force": double.parse(totalGForce.toStringAsFixed(2)),
      "accel": {
        "x": double.parse(_accelX.toStringAsFixed(3)),
        "y": double.parse(_accelY.toStringAsFixed(3)),
        "z": double.parse(_accelZ.toStringAsFixed(3)),
      },
    });
  }

  Future<String?> stopRealSensors() async {
    _loggingTimer?.cancel();
    _loggingTimer = null;
    _isMonitoring = false;
    final filePath = await _exportDataToJson();
    if (!_isObdConnected) _speedKmh = 0;
    notifyListeners();
    return filePath;
  }

  Future<String?> _exportDataToJson() async {
    if (_sensorLog.isEmpty) return null;
    try {
      final dir = await getApplicationDocumentsDirectory();
      final ts = DateTime.now().millisecondsSinceEpoch.toString();
      final file = File('${dir.path}/test_data_$ts.json');

      final videoOffsetMs = (_testStartTime != null && _videoStartTime != null)
          ? _videoStartTime!.difference(_testStartTime!).inMilliseconds
          : 0;

      final out = {
        "test_start_time": _testStartTime?.toIso8601String(),
        "video_offset_ms": videoOffsetMs,
        "total_duration_ms": _sensorLog.last["time_ms"],
        "data": _sensorLog,
      };
      await file.writeAsString(jsonEncode(out));
      return file.path;
    } catch (e) {
      debugPrint("[TEST] ❌ Export error: $e");
      return null;
    }
  }

  // 🆕 SHA-256 of file (used by api_service before upload)
  static Future<String> sha256OfFile(String path) async {
    final bytes = await File(path).readAsBytes();
    return sha256.convert(bytes).toString();
  }

  @override
  void dispose() {
    _loggingTimer?.cancel();
    _obdPollTimer?.cancel();
    _rttPingTimer?.cancel();
    _obdStreamSub?.cancel();
    _accelSubscription?.cancel();
    _gpsSubscription?.cancel();
    _obdConnection?.dispose();
    _parserIsolate?.kill(priority: Isolate.immediate);
    _parserRecv?.close();
    super.dispose();
  }
}

// ==========================================
// Top-level isolate entry
// ==========================================
void _obdParserEntry(SendPort mainSend) {
  final port = ReceivePort();
  mainSend.send(port.sendPort);
  String buffer = "";

  port.listen((data) {
    if (data is! Uint8List) return;
    try {
      buffer += utf8.decode(data, allowMalformed: true);
    } catch (_) {
      return;
    }

    while (buffer.contains('>')) {
      final i = buffer.indexOf('>');
      final raw = buffer.substring(0, i);
      buffer = buffer.substring(i + 1);
      final hex = raw.replaceAll(RegExp(r'[\r\n ]'), '').toUpperCase();
      if (hex.isEmpty) continue;

      // Every prompt return triggers RTT pong
      mainSend.send({'type': 'rtt_response'});

      if (hex.contains("NODATA") ||
          hex.contains("STOPPED") ||
          hex.contains("?") ||
          hex.length < 6)
        continue;

      final r = hex.indexOf('410C');
      if (r != -1 && hex.length >= r + 8) {
        try {
          final a = int.parse(hex.substring(r + 4, r + 6), radix: 16);
          final b = int.parse(hex.substring(r + 6, r + 8), radix: 16);
          mainSend.send({'type': 'rpm', 'value': (a * 256 + b) / 4});
        } catch (_) {}
      }
      final s = hex.indexOf('410D');
      if (s != -1 && hex.length >= s + 6) {
        try {
          final a = int.parse(hex.substring(s + 4, s + 6), radix: 16);
          mainSend.send({'type': 'speed', 'value': a});
        } catch (_) {}
      }
    }
  });
}

class _TimedSensorSample {
  final DateTime time;
  final double speedKmh;
  final String source;
  _TimedSensorSample({
    required this.time,
    required this.speedKmh,
    required this.source,
  });
}

class _TimedAccelSample {
  final DateTime time;
  final double accelX, accelY, accelZ;
  _TimedAccelSample({
    required this.time,
    required this.accelX,
    required this.accelY,
    required this.accelZ,
  });
}
