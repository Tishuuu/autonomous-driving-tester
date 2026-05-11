/// נקודה מרכזית להגדרות שרת.
/// כל שינוי בכתובת ה-IP — רק כאן.
class ApiConfig {
  /// כתובת המחשב המארח (USB tethering)
  static const String host = "172.24.96.172";
  static const int port = 8000;

  static const String baseUrl = "http://$host:$port";

  // נתיבי API
  static const String authBase = "$baseUrl/api/auth";
  static const String testsBase = "$baseUrl/api/tests";
  static const String studentsBase = "$baseUrl/api/students";

  // טיים-אאוטים
  static const Duration shortTimeout = Duration(seconds: 10);
  static const Duration mediumTimeout = Duration(seconds: 30);
  static const Duration longTimeout = Duration(
    minutes: 40,
  ); // להעלאת וידאו וניתוח ארוך
}
