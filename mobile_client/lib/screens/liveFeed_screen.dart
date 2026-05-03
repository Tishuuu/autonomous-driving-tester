import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:camera/camera.dart';
import '../providers/sensor_provider.dart';
import 'processing_screen.dart';

class LivefeedScreen extends StatefulWidget {
  const LivefeedScreen({super.key});

  @override
  State<LivefeedScreen> createState() => _LivefeedScreenState();
}

class _LivefeedScreenState extends State<LivefeedScreen> {
  bool _isTesting = false;
  CameraController? _cameraController;
  bool _isCameraInitialized = false;

  @override
  void initState() {
    super.initState();
    // נעילת המסך לרוחב (Landscape) עבור ה-Dashcam
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.landscapeRight,
      DeviceOrientation.landscapeLeft,
    ]);

    _initCamera();
  }

  // אתחול מצלמת המכשיר (מצלמה אחורית)
  Future<void> _initCamera() async {
    try {
      final cameras = await availableCameras();
      if (cameras.isEmpty) return;

      // נחפש את המצלמה האחורית
      final backCamera = cameras.firstWhere(
        (c) => c.lensDirection == CameraLensDirection.back,
        orElse: () => cameras.first,
      );

      // נגדיר רזולוציה גבוהה, ללא סאונד (חוסך מקום ופרטיות)
      _cameraController = CameraController(
        backCamera,
        ResolutionPreset.high,
        enableAudio: false,
      );

      await _cameraController!.initialize();
      if (!mounted) return;

      setState(() {
        _isCameraInitialized = true;
      });
    } catch (e) {
      print("❌ Camera Init Error: $e");
    }
  }

  @override
  void dispose() {
    // ✅ עוצרים הקלטה אם עדיין רצה (למנוע קבצי וידאו corrupt)
    if (_cameraController != null &&
        _cameraController!.value.isInitialized &&
        _cameraController!.value.isRecordingVideo) {
      _cameraController!.stopVideoRecording().catchError((_) => XFile(''));
    }
    _cameraController?.dispose();

    // החזרת המסך למצב רגיל (Portrait)
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
    super.dispose();
  }

  // הפונקציה שמופעלת בלחיצה על "START / STOP"
  void _toggleTest() async {
    final sensorProvider = Provider.of<SensorProvider>(context, listen: false);

    // ✅ התיקון של קלאוד: חוסמים התחלת טסט עד שיש נעילת GPS תקינה (למניעת 0,0)
    if (!_isTesting && !sensorProvider.hasGpsFix) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("Waiting for GPS lock... please move outdoors."),
          backgroundColor: Colors.orange,
        ),
      );
      return; // יוצא מהפונקציה ולא מתחיל את הטסט
    }

    if (_isTesting) {
      // ===== עצירת הטסט =====
      setState(() => _isTesting = false);
      print("🔴 Stopping Test...");

      // 1. עצירת החיישנים וקבלת נתיב ה-JSON
      String? jsonPath = await sensorProvider.stopRealSensors();

      // 2. עצירת המצלמה וקבלת הוידאו
      if (_cameraController != null &&
          _cameraController!.value.isRecordingVideo) {
        final XFile videoFile = await _cameraController!.stopVideoRecording();
        print("✅ Video saved locally at: ${videoFile.path}");

        // ✅ החזרת ה-orientation ל-Portrait לפני המעבר למסך הבא
        await SystemChrome.setPreferredOrientations([
          DeviceOrientation.portraitUp,
          DeviceOrientation.portraitDown,
        ]);
        await Future.delayed(const Duration(milliseconds: 200));

        // 3. מעבר למסך העיבוד החדש
        if (jsonPath != null && mounted) {
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
              builder: (_) => ProcessingScreen(
                videoPath: videoFile.path,
                jsonPath: jsonPath,
              ),
            ),
          );
        } else if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('Failed to save sensor data ❌'),
              backgroundColor: Colors.red,
            ),
          );
        }
      }
    } else {
      // ===== התחלת הטסט =====
      setState(() => _isTesting = true);
      print("🟢 Starting Real World Test...");

      // 1. הפעלת איסוף נתוני החיישנים
      await sensorProvider.startRealSensors();

      // 2. התחלת צילום הוידאו
      if (_cameraController != null &&
          !_cameraController!.value.isRecordingVideo) {
        await _cameraController!.startVideoRecording();
        // ✅ סימון הזמן המדויק שהוידאו החל - לסנכרון עם החיישנים
        sensorProvider.markVideoStart();
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Row(
        children: [
          // צד שמאל: תצוגת המצלמה החיה (75% מהמסך)
          Expanded(
            flex: 3,
            child: _isCameraInitialized
                ? Stack(
                    alignment: Alignment.center,
                    children: [
                      // תצוגת המצלמה
                      SizedBox(
                        width: double.infinity,
                        height: double.infinity,
                        child: CameraPreview(_cameraController!),
                      ),
                      // נקודה אדומה מהבהבת שמראה הקלטה
                      if (_isTesting)
                        const Positioned(
                          top: 20,
                          right: 20,
                          child: Icon(
                            Icons.fiber_manual_record,
                            color: Colors.red,
                            size: 28,
                          ),
                        ),
                    ],
                  )
                : const Center(
                    child: CircularProgressIndicator(color: Colors.blueAccent),
                  ),
          ),

          // צד ימין: פאנל השליטה והחיישנים (25% מהמסך)
          Expanded(
            flex: 1,
            child: Container(
              color: const Color(0xFF1E1E1E),
              padding: const EdgeInsets.symmetric(vertical: 20, horizontal: 12),
              child: Consumer<SensorProvider>(
                builder: (context, sensorData, child) {
                  return Column(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      // כרטיסיות נתונים שמתעדכנות בזמן אמת מה-Provider
                      _buildDataCard(
                        "SPEED",
                        sensorData.speed.toStringAsFixed(0),
                        "KM/H",
                        Colors.blueAccent,
                      ),
                      _buildDataCard(
                        "RPM",
                        sensorData.rpm.toStringAsFixed(0),
                        "RPM",
                        Colors.orangeAccent,
                      ),
                      _buildDataCard(
                        "G-FORCE",
                        sensorData.totalGForce.toStringAsFixed(1),
                        "G",
                        Colors.purpleAccent,
                      ),

                      const Spacer(),

                      // כפתור התחלה / עצירה
                      GestureDetector(
                        onTap: _isCameraInitialized ? _toggleTest : null,
                        child: AnimatedContainer(
                          duration: const Duration(milliseconds: 300),
                          height: 60,
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: _isTesting
                                ? Colors.redAccent
                                : Colors.greenAccent.shade700,
                            borderRadius: BorderRadius.circular(15),
                            boxShadow: [
                              BoxShadow(
                                color:
                                    (_isTesting
                                            ? Colors.redAccent
                                            : Colors.greenAccent)
                                        .withOpacity(0.3),
                                blurRadius: 15,
                                spreadRadius: 2,
                              ),
                            ],
                          ),
                          child: Center(
                            child: Text(
                              _isTesting ? "STOP TEST" : "START TEST",
                              style: GoogleFonts.lexend(
                                color: Colors.white,
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.5,
                              ),
                            ),
                          ),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ),
        ],
      ),
    );
  }

  // עיצוב כרטיסיות הנתונים
  Widget _buildDataCard(String label, String value, String unit, Color color) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(vertical: 12),
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: const Color(0xFF2A2A2A),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withOpacity(0.5), width: 1.5),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.1),
            blurRadius: 10,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Text(
            label,
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
          const SizedBox(height: 4),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            crossAxisAlignment: CrossAxisAlignment.baseline,
            textBaseline: TextBaseline.alphabetic,
            children: [
              Text(
                value,
                style: GoogleFonts.lexend(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 4),
              Text(
                unit,
                style: GoogleFonts.lexend(color: Colors.white54, fontSize: 12),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
