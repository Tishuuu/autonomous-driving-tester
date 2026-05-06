import 'dart:convert';
import 'dart:io';
import 'package:crypto/crypto.dart';
import 'package:http/http.dart' as http;
import 'api_config.dart';

class ApiService {
  // ==========================================
  // 🆕 JWT token management
  // ==========================================
  static String? _token;
  static void setToken(String? t) => _token = t;
  static String? get token => _token;
  static void clearToken() => _token = null;

  static Map<String, String> _authHeaders([Map<String, String>? extra]) {
    final h = <String, String>{
      if (_token != null) "Authorization": "Bearer $_token",
    };
    if (extra != null) h.addAll(extra);
    return h;
  }

  static Map<String, String> _jsonAuthHeaders() =>
      _authHeaders({"Content-Type": "application/json"});

  // ==========================================
  // 🆕 Auth endpoints
  // ==========================================
  static Future<Map<String, dynamic>?> login(
    String email,
    String password,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse('${ApiConfig.authBase}/login'),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({"email": email, "password": password}),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        final body = jsonDecode(response.body);
        _token = body["access_token"];
        return Map<String, dynamic>.from(body);
      }
      return null;
    } catch (e) {
      print("❌ login error: $e");
      return null;
    }
  }

  static Future<Map<String, dynamic>?> register(
    String name,
    String email,
    String password,
  ) async {
    try {
      final response = await http
          .post(
            Uri.parse('${ApiConfig.authBase}/register'),
            headers: {"Content-Type": "application/json"},
            body: jsonEncode({
              "name": name,
              "email": email,
              "password": password,
            }),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        final body = jsonDecode(response.body);
        _token = body["access_token"];
        return Map<String, dynamic>.from(body);
      }
      return null;
    } catch (e) {
      print("❌ register error: $e");
      return null;
    }
  }

  // ==========================================
  // 🆕 SHA-256 helper
  // ==========================================
  static Future<String> _sha256OfFile(String path) async {
    final bytes = await File(path).readAsBytes();
    return sha256.convert(bytes).toString();
  }

  // ==========================================
  // 1. Upload test files — with SHA256 integrity
  // ==========================================
  static Future<Map<String, dynamic>?> uploadTestFiles(
    String videoPath,
    String jsonPath,
    String studentId, {
    String? testId,
    int maxAttempts = 2,
  }) async {
    final actualTestId =
        testId ?? "TEST_${DateTime.now().millisecondsSinceEpoch}";

    // 🆕 Compute checksums once (even across retries)
    final videoHash = await _sha256OfFile(videoPath);
    final jsonHash = await _sha256OfFile(jsonPath);

    for (int attempt = 1; attempt <= maxAttempts; attempt++) {
      try {
        print("🚀 Upload attempt $attempt/$maxAttempts ID: $actualTestId");

        var request = http.MultipartRequest(
          'POST',
          Uri.parse('${ApiConfig.testsBase}/evaluate'),
        );

        // 🆕 Bearer token
        if (_token != null) {
          request.headers['Authorization'] = "Bearer $_token";
        }

        request.fields['student_id'] = studentId;
        request.fields['test_id'] = actualTestId;
        request.fields['video_sha256'] = videoHash;
        request.fields['sensors_sha256'] = jsonHash;

        request.files.add(
          await http.MultipartFile.fromPath('video', videoPath),
        );
        request.files.add(
          await http.MultipartFile.fromPath('sensors', jsonPath),
        );

        var response = await request.send().timeout(ApiConfig.longTimeout);
        final body = await response.stream.bytesToString();

        if (response.statusCode == 200) {
          print("✅ Analysis Complete!");
          final decoded = jsonDecode(body);
          if (decoded is Map<String, dynamic>) return decoded;
          return null;
        }

        // 422 = checksum mismatch — don't retry, file is bad
        if (response.statusCode == 422) {
          print("❌ Integrity failure (422): $body");
          return null;
        }

        // 401 = token expired
        if (response.statusCode == 401) {
          print("❌ Auth expired (401): $body");
          return null;
        }

        print("❌ Server Error: ${response.statusCode} - $body");
        if (response.statusCode >= 500 && attempt < maxAttempts) {
          await Future.delayed(const Duration(seconds: 3));
          continue;
        }
        return null;
      } catch (e) {
        print("❌ Connection Error (attempt $attempt): $e");
        if (attempt < maxAttempts) {
          await Future.delayed(const Duration(seconds: 3));
          continue;
        }
        return null;
      }
    }
    return null;
  }

  // ==========================================
  // 1b. Progress polling
  // ==========================================
  static Future<Map<String, dynamic>?> getProgress(String testId) async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/progress/$testId'),
            headers: _authHeaders(),
          )
          .timeout(const Duration(seconds: 5));
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (_) {}
    return null;
  }

  // ==========================================
  // 2. Save test result
  // ==========================================
  static Future<bool> saveTest({
    required String studentId,
    required int grade,
    String? result,
    bool? passed,
    int? mistakesCount,
    List<dynamic>? mistakeCodes,
    List<dynamic>? ignoredWarningCodes,
    int? ignoredWarningEventsCount,
    List<dynamic>? ignoredWarningEvents,
    required List<dynamic> violationsCodes,
    required Map<String, dynamic> xaiExplanations,
    required int violationEventsCount,
    int windowsAnalyzed = 0,
    String? testId,
    List<dynamic>? decisionLog,
    List<dynamic>? actionSequences,
    List<dynamic>? positiveActions,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('${ApiConfig.testsBase}/save'),
            headers: _jsonAuthHeaders(),
            body: jsonEncode({
              "student_id": studentId,
              // tester_email no longer sent — server uses token
              "tester_email": "from_token",
              "grade": grade,
              "result": result ?? (grade >= 80 ? "PASS" : "FAIL"),
              "passed": passed ?? grade >= 80,
              "mistakes_count": mistakesCount ?? violationEventsCount,
              "mistake_codes": mistakeCodes ?? violationsCodes,
              "violations_codes": violationsCodes,
              "xai_explanations": xaiExplanations,
              "violation_events_count": violationEventsCount,
              "ignored_warning_codes": ignoredWarningCodes ?? [],
              "ignored_warning_events_count": ignoredWarningEventsCount ?? 0,
              "ignored_warning_events": ignoredWarningEvents ?? [],
              "windows_analyzed": windowsAnalyzed,
              "test_id": testId,
              "test_date": DateTime.now().toIso8601String(),
              "decision_log": decisionLog ?? [],
              "action_sequences": actionSequences ?? [],
              "positive_actions": positiveActions ?? [],
            }),
          )
          .timeout(ApiConfig.mediumTimeout);
      if (response.statusCode == 200 || response.statusCode == 201) return true;
      print("❌ Save failed: ${response.statusCode} - ${response.body}");
      return false;
    } catch (e) {
      print("❌ Save error: $e");
      return false;
    }
  }

  // ==========================================
  // 3. Students
  // ==========================================
  static Future<List<Map<String, dynamic>>> getStudents() async {
    try {
      final response = await http
          .get(Uri.parse('${ApiConfig.studentsBase}/'), headers: _authHeaders())
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.map((e) => Map<String, dynamic>.from(e)).toList();
      }
      return [];
    } catch (e) {
      print("❌ getStudents error: $e");
      return [];
    }
  }

  static Future<Map<String, dynamic>> addStudent({
    required String studentId,
    required String name,
  }) async {
    try {
      final response = await http
          .post(
            Uri.parse('${ApiConfig.studentsBase}/'),
            headers: _jsonAuthHeaders(),
            body: jsonEncode({
              "student_id": studentId,
              "name": name,
              "tester_email": "from_token", // ignored by server
            }),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200 || response.statusCode == 201) {
        return {"success": true};
      } else if (response.statusCode == 409) {
        return {
          "success": false,
          "error": "Student with this ID already exists",
        };
      }
      return {"success": false, "error": "Server error: ${response.body}"};
    } catch (e) {
      return {"success": false, "error": "Connection error: $e"};
    }
  }

  // ==========================================
  // 4. History
  // ==========================================
  static Future<List<dynamic>> getStudentHistory(String studentId) async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/student/$studentId'),
            headers: _authHeaders(),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) return jsonDecode(response.body);
      return [];
    } catch (e) {
      print("❌ getStudentHistory error: $e");
      return [];
    }
  }

  /// 🆕 Migrated from /tester/{email} → /me/tests
  static Future<List<dynamic>> getTesterHistory() async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/me/tests'),
            headers: _authHeaders(),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) return jsonDecode(response.body);
      return [];
    } catch (e) {
      print("❌ getTesterHistory error: $e");
      return [];
    }
  }

  static Future<Map<String, dynamic>?> getTestDetail(
    String testObjectId,
  ) async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/detail/$testObjectId'),
            headers: _authHeaders(),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      return null;
    } catch (e) {
      print("❌ getTestDetail error: $e");
      return null;
    }
  }

  // ==========================================
  // 5. Predictions
  // ==========================================
  static Future<Map<String, dynamic>?> getStudentPrediction(
    String studentId,
  ) async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/prediction/$studentId'),
            headers: _authHeaders(),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      return null;
    } catch (e) {
      print("❌ getStudentPrediction error: $e");
      return null;
    }
  }

  /// 🆕 Migrated from /predictions/{email} → /predictions
  static Future<List<Map<String, dynamic>>> getAllPredictions() async {
    try {
      final response = await http
          .get(
            Uri.parse('${ApiConfig.testsBase}/predictions'),
            headers: _authHeaders(),
          )
          .timeout(ApiConfig.shortTimeout);
      if (response.statusCode == 200) {
        final List<dynamic> data = jsonDecode(response.body);
        return data.map((e) => Map<String, dynamic>.from(e)).toList();
      }
      return [];
    } catch (e) {
      print("❌ getAllPredictions error: $e");
      return [];
    }
  }
}
