import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://127.0.0.1:8000/api/tests";

  // =========================================================
  // 1. פונקציית סיום הטסט: שולחת וידאו, חיישנים ו-ID לעיבוד
  // =========================================================
  static Future<bool> uploadTestFiles(
    String videoPath,
    String jsonPath,
    String studentId,
  ) async {
    try {
      print("🚀 Uploading Test Data to Server...");

      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$baseUrl/process'), // פונה לנתיב ה-AI החדש שבנינו
      );

      // הוספת מזהה התלמיד (חובה, כי השרת מצפה לזה)
      request.fields['student_id'] = studentId;

      // הוספת הקבצים הכבדים
      request.files.add(await http.MultipartFile.fromPath('video', videoPath));
      request.files.add(
        await http.MultipartFile.fromPath('sensor_data', jsonPath),
      );

      var response = await request.send();

      if (response.statusCode == 200) {
        var responseData = await response.stream.bytesToString();
        print("✅ Analysis Complete! Server says: $responseData");
        return true;
      } else {
        var errorData = await response.stream.bytesToString();
        print("❌ Server Error: ${response.statusCode} - $errorData");
        return false;
      }
    } catch (e) {
      print("❌ Connection Error: $e");
      return false;
    }
  }

  // =========================================================
  // 2. פונקציית היסטוריה: שולפת את כל הטסטים של התלמיד מה-DB
  // =========================================================
  static Future<List<dynamic>> getStudentHistory(String studentId) async {
    try {
      print("🔍 Fetching history for student: $studentId...");

      final response = await http.get(Uri.parse('$baseUrl/user/$studentId'));

      if (response.statusCode == 200) {
        // השרת מחזיר מערך של טסטים ב-JSON, אנחנו מתרגמים אותו לרשימה
        List<dynamic> tests = jsonDecode(response.body);
        print("✅ Found ${tests.length} tests.");
        return tests;
      } else {
        print("❌ Server Error: ${response.statusCode} - ${response.body}");
        return [];
      }
    } catch (e) {
      print("❌ Connection Error: $e");
      return [];
    }
  }
}
