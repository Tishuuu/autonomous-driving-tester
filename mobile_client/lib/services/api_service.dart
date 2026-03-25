import 'dart:convert';
import 'package:http/http.dart' as http;

class ApiService {
  static const String baseUrl = "http://192.168.8.177:8000/api/tests";

  static Future<bool> sendTestResult(Map<String, dynamic> testData) async {
    try {
      print("Sending test data to server...");

      final response = await http.post(
        Uri.parse('$baseUrl/save'),
        headers: {"Content-Type": "application/json"},
        body: jsonEncode(testData),
      );

      if (response.statusCode == 200) {
        print("Success! Server response: ${response.body}");
        return true;
      } else {
        print(" Server Error: ${response.statusCode} - ${response.body}");
        return false;
      }
    } catch (e) {
      print(" Connection Error: $e");
      return false;
    }
  }
}
