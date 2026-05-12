import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'dart:isolate';
import 'dart:math';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bluetooth_serial/flutter_bluetooth_serial.dart';
import 'package:geolocator/geolocator.dart';
import 'package:path_provider/path_provider.dart';
import 'package:sensors_plus/sensors_plus.dart';

class SensorProvider extends ChangeNotifier {
  bool _isObdConnected = false;
  bool _isCameraReady = false;
  bool _isMonitoring = false;

  double _speedKmh = 0.0;
  double _rpm = 0.0;

  // GPS
  double? _lat;
  double? _lon;
  double? _gpsAccuracyMeters;
  double? _gpsSpeedKmh;
  bool _hasGpsFix = false;
  DateTime? _lastFixAt;
  int _gpsUpdateCount = 0;
  Timer? _gpsWatchdogTimer;
  DateTime? _lastGpsStreamRestartAt;

  static const Duration _gpsStaleThreshold = Duration(seconds: 5);
  static const Duration _gpsRestartCooldown = Duration(seconds: 8);

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

  // Isolate parser
  Isolate? _parserIsolate;
  SendPort? _parserSend;
  ReceivePort? _parserRecv;
  StreamSubscription? _parserRecvSub;

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

  // Tier: 1=OBD, 2=IMU, 3=GPS
  int _activeTier = 3;
  int get activeTier => _activeTier;

  // Getters
  bool get isGpsListening => _gpsSubscription != null;
  bool get isObdConnected => _isObdConnected && !_obdIsStale;
  bool get isGpsStale {
    if (!_hasGpsFix || _lastFixAt == null) return true;
    return DateTime.now().difference(_lastFixAt!) > _gpsStaleThreshold;
  }

  bool get isGpsLocked => _hasGpsFix && !isGpsStale;
  bool get hasGpsFix => isGpsLocked;
  bool get isCameraReady => _isCameraReady;
  bool get isImuActive => _accelSubscription != null;
  bool get isSystemReady => isCameraReady && isImuActive;
  bool get isMonitoring => _isMonitoring;

  double get speed => _speedKmh;
  double get rpm => _rpm;
  double? get lat => _lat;
  double? get lon => _lon;
  double? get gpsAccuracyMeters => _gpsAccuracyMeters;
  double? get gpsSpeedKmh => _gpsSpeedKmh;
  int get gpsUpdateCount => _gpsUpdateCount;
  int? get gpsAgeMs => _lastFixAt == null
      ? null
      : DateTime.now().difference(_lastFixAt!).inMilliseconds;

  double get totalGForce =>
      sqrt(pow(_accelX, 2) + pow(_accelY, 2) + pow(_accelZ, 2));
  double get obdLatencyMs => _obdLatencyMs;

  String get speedSourceLabel {
    switch (_activeTier) {
      case 1:
        return 'OBD';
      case 2:
        return 'IMU';
      case 3:
        return 'GPS';
      default:
        return '—';
    }
  }

  // ==========================================
  // OBD Connect
  // ==========================================
  Future<bool> connectToOBD(BluetoothDevice device) async {
    debugPrint('[OBD] Connecting to ${device.name}');
    try {
      _obdConnection = await BluetoothConnection.toAddress(device.address);
      _isObdConnected = true;
      _isObdListening = true;
      _obdIsStale = false;
      _lastObdResponseTime = DateTime.now();
      notifyListeners();

      await _spawnParserIsolate();
      _startObdStream();

      _sendObdCommand('ATZ\r');
      await Future.delayed(const Duration(milliseconds: 800));
      _sendObdCommand('ATE0\r');
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand('ATL0\r');
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand('ATS0\r');
      await Future.delayed(const Duration(milliseconds: 200));
      _sendObdCommand('ATSP0\r');
      await Future.delayed(const Duration(milliseconds: 500));

      _startPollingLoop();
      _startRttMeasurementLoop();
      return true;
    } catch (e) {
      debugPrint('[OBD] Connection failed: $e');
      await disconnectOBD();
      return false;
    }
  }

  void _sendObdCommand(String command) {
    try {
      if (_obdConnection == null || !_obdConnection!.isConnected) return;
      _obdConnection!.output.add(utf8.encode(command));
    } catch (e) {
      debugPrint('[OBD] Send error: $e');
    }
  }

  Future<void> _spawnParserIsolate() async {
    await _parserRecvSub?.cancel();
    _parserRecv?.close();
    _parserIsolate?.kill(priority: Isolate.immediate);

    _parserRecv = ReceivePort();
    _parserIsolate = await Isolate.spawn(
      _obdParserEntry,
      _parserRecv!.sendPort,
    );

    final completer = Completer<SendPort>();
    _parserRecvSub = _parserRecv!.listen((msg) {
      if (msg is SendPort && !completer.isCompleted) {
        completer.complete(msg);
        return;
      }
      if (msg is Map) _handleParsedFromIsolate(msg);
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
        _parserSend?.send(data);
      },
      onError: (e) {
        debugPrint('[OBD] Stream error: $e');
        _handleDisconnect();
      },
      onDone: _handleDisconnect,
      cancelOnError: false,
    );
  }

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
          _activeTier = isGpsLocked ? 3 : 2;
          notifyListeners();
        }
      }

      _sendObdCommand('010C\r');
      Future.delayed(const Duration(milliseconds: 130), () {
        if (_isObdListening && _isObdConnected) _sendObdCommand('010D\r');
      });
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
      _sendObdCommand('ATRV\r');
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
    _activeTier = isGpsLocked ? 3 : 2;
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
    await _parserRecvSub?.cancel();
    _parserRecvSub = null;
    _parserRecv?.close();
    _parserRecv = null;
    _parserSend = null;

    _isObdConnected = false;
    _obdIsStale = false;
    _speedKmh = 0;
    _rpm = 0;
    _waitingForRttResponse = false;
    _activeTier = isGpsLocked ? 3 : 2;
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

  // ==========================================
  // GPS
  // ==========================================
  bool _isUsableGpsFix(Position position, {Duration? maxAge}) {
    if (position.latitude == 0.0 && position.longitude == 0.0) return false;
    if (position.latitude.abs() > 90 || position.longitude.abs() > 180)
      return false;
    if (position.accuracy.isNaN ||
        position.accuracy <= 0 ||
        position.accuracy > 60)
      return false;

    if (maxAge != null) {
      final age = DateTime.now().difference(position.timestamp);
      if (age > maxAge) return false;
    }
    return true;
  }

  LocationSettings _gpsSettings() {
    if (defaultTargetPlatform == TargetPlatform.android) {
      return AndroidSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0,
        intervalDuration: const Duration(seconds: 1),
        forceLocationManager: false,
      );
    }

    if (defaultTargetPlatform == TargetPlatform.iOS ||
        defaultTargetPlatform == TargetPlatform.macOS) {
      return AppleSettings(
        accuracy: LocationAccuracy.bestForNavigation,
        distanceFilter: 0,
        activityType: ActivityType.automotiveNavigation,
        pauseLocationUpdatesAutomatically: false,
        showBackgroundLocationIndicator: true,
      );
    }

    return const LocationSettings(
      accuracy: LocationAccuracy.bestForNavigation,
      distanceFilter: 0,
    );
  }

  void _applyGpsFix(Position position, {required bool fresh}) {
    _lat = position.latitude;
    _lon = position.longitude;
    _gpsAccuracyMeters = position.accuracy;
    _gpsSpeedKmh = max(0, position.speed * 3.6);

    if (fresh) {
      _hasGpsFix = true;
      _lastFixAt = DateTime.now();
      _gpsUpdateCount++;
    }
  }

  Future<bool> startGps() async {
    if (_gpsSubscription != null && isGpsLocked) return true;

    if (!await Geolocator.isLocationServiceEnabled()) {
      debugPrint('[GPS] Location service is disabled');
      return false;
    }

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
    }
    if (permission == LocationPermission.denied ||
        permission == LocationPermission.deniedForever) {
      debugPrint('[GPS] Permission denied: $permission');
      return false;
    }

    // Last-known is useful only for display. It must not unlock the system,
    // because it can be old and cause a full drive with fixed coordinates.
    try {
      final lastKnown = await Geolocator.getLastKnownPosition();
      if (lastKnown != null &&
          _isUsableGpsFix(lastKnown, maxAge: const Duration(seconds: 5))) {
        _applyGpsFix(lastKnown, fresh: false);
        notifyListeners();
      }
    } catch (e) {
      debugPrint('[GPS] lastKnown failed: $e');
    }

    await _startGpsStream();

    // Prime one fresh reading so the dashboard can become ready immediately.
    try {
      final current = await Geolocator.getCurrentPosition(
        locationSettings: _gpsSettings(),
      ).timeout(const Duration(seconds: 12));
      if (_isUsableGpsFix(current)) {
        _onGpsPosition(current);
      }
    } catch (e) {
      debugPrint('[GPS] getCurrentPosition failed: $e');
    }

    _startGpsWatchdog();
    notifyListeners();
    return isGpsLocked;
  }

  Future<void> _startGpsStream() async {
    await _gpsSubscription?.cancel();
    _gpsSubscription =
        Geolocator.getPositionStream(locationSettings: _gpsSettings()).listen(
          _onGpsPosition,
          onError: (e) async {
            debugPrint('[GPS] Stream error: $e');
            _hasGpsFix = false;
            notifyListeners();
            await _restartGpsStreamIfNeeded();
          },
          cancelOnError: false,
        );
    _lastGpsStreamRestartAt = DateTime.now();
  }

  void _onGpsPosition(Position position) {
    if (!_isUsableGpsFix(position)) return;

    _applyGpsFix(position, fresh: true);

    if (!_isObdConnected || _obdIsStale) {
      if (!_imuStopDetected) {
        _speedKmh = _gpsSpeedKmh ?? 0;
        _activeTier = 3;
      }
    }

    notifyListeners();
  }

  void _startGpsWatchdog() {
    _gpsWatchdogTimer?.cancel();
    _gpsWatchdogTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      if (_gpsSubscription == null) return;

      if (isGpsStale) {
        if (_hasGpsFix) {
          _hasGpsFix = false;
          if (_activeTier == 3) _speedKmh = 0;
          notifyListeners();
        }
        await _restartGpsStreamIfNeeded();
      }
    });
  }

  Future<void> _restartGpsStreamIfNeeded() async {
    final now = DateTime.now();
    if (_lastGpsStreamRestartAt != null &&
        now.difference(_lastGpsStreamRestartAt!) < _gpsRestartCooldown) {
      return;
    }

    debugPrint('[GPS] Restarting stale GPS stream');
    try {
      await _startGpsStream();
    } catch (e) {
      debugPrint('[GPS] Restart failed: $e');
    }
  }

  void stopGps() {
    _gpsWatchdogTimer?.cancel();
    _gpsWatchdogTimer = null;
    _gpsSubscription?.cancel();
    _gpsSubscription = null;
    _hasGpsFix = false;
    _lastFixAt = null;
    _lat = null;
    _lon = null;
    _gpsAccuracyMeters = null;
    _gpsSpeedKmh = null;
    _gpsUpdateCount = 0;
    if (_activeTier == 3) _speedKmh = 0;
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
    if (_gpsSubscription == null || !isGpsLocked) await startGps();
    if (!_isCameraReady) await activateCamera();

    if (!isGpsLocked) {
      throw StateError('Cannot start test without a fresh GPS fix');
    }

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

  double _getSyncedSpeedKmhForLog(DateTime logTime) {
    if (_activeTier != 1 || _sensorBuffer.isEmpty) return _speedKmh;

    _TimedSensorSample best = _sensorBuffer.last;
    int bestDiffMs = best.time.difference(logTime).inMilliseconds.abs();

    for (int i = _sensorBuffer.length - 1; i >= 0; i--) {
      final sample = _sensorBuffer[i];
      final diffMs = sample.time.difference(logTime).inMilliseconds.abs();
      if (diffMs < bestDiffMs) {
        best = sample;
        bestDiffMs = diffMs;
      }
      if (sample.time.isBefore(logTime) && diffMs > bestDiffMs) break;
    }

    if (bestDiffMs > 1500) return _speedKmh;
    return best.speedKmh;
  }

  void _logCurrentState() {
    if (_testStartTime == null) return;

    // This is the key GPS fix: never log stale GPS. If the stream stops,
    // logging pauses instead of creating a full CSV with one fixed coordinate.
    if (!isGpsLocked || _lat == null || _lon == null) return;

    final now = DateTime.now();
    final t = now.difference(_testStartTime!).inMilliseconds;
    final syncedSpeedKmh = _getSyncedSpeedKmhForLog(now);
    final gpsAge = gpsAgeMs ?? 0;

    _sensorLog.add({
      'time_ms': t,
      'speed_kmh': double.parse(syncedSpeedKmh.toStringAsFixed(2)),
      'rpm': double.parse(_rpm.toStringAsFixed(2)),
      'lat': double.parse(_lat!.toStringAsFixed(7)),
      'lon': double.parse(_lon!.toStringAsFixed(7)),
      'gps_accuracy': double.parse(
        (_gpsAccuracyMeters ?? 999).toStringAsFixed(2),
      ),
      'gps_age_ms': gpsAge,
      'gps_update_count': _gpsUpdateCount,
      'speed_source': _activeTier,
      'obd_latency_ms': double.parse(_obdLatencyMs.toStringAsFixed(1)),
      'g_force': double.parse(totalGForce.toStringAsFixed(3)),
      'accel': {
        'x': double.parse(_accelX.toStringAsFixed(4)),
        'y': double.parse(_accelY.toStringAsFixed(4)),
        'z': double.parse(_accelZ.toStringAsFixed(4)),
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
        'test_start_time': _testStartTime?.toIso8601String(),
        'video_offset_ms': videoOffsetMs,
        'total_duration_ms': _sensorLog.last['time_ms'],
        'gps_updates': _gpsUpdateCount,
        'data': _sensorLog,
      };
      await file.writeAsString(jsonEncode(out));
      return file.path;
    } catch (e) {
      debugPrint('[TEST] Export error: $e');
      return null;
    }
  }

  static Future<String> sha256OfFile(String path) async {
    final bytes = await File(path).readAsBytes();
    return sha256.convert(bytes).toString();
  }

  @override
  void dispose() {
    _loggingTimer?.cancel();
    _gpsWatchdogTimer?.cancel();
    _obdPollTimer?.cancel();
    _rttPingTimer?.cancel();
    _obdStreamSub?.cancel();
    _accelSubscription?.cancel();
    _gpsSubscription?.cancel();
    _obdConnection?.dispose();
    _parserIsolate?.kill(priority: Isolate.immediate);
    _parserRecvSub?.cancel();
    _parserRecv?.close();
    super.dispose();
  }
}

void _obdParserEntry(SendPort mainSend) {
  final port = ReceivePort();
  mainSend.send(port.sendPort);
  String buffer = '';

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

      mainSend.send({'type': 'rtt_response'});

      if (hex.contains('NODATA') ||
          hex.contains('STOPPED') ||
          hex.contains('?') ||
          hex.length < 6) {
        continue;
      }

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
