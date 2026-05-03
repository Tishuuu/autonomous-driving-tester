import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../providers/user_provider.dart';

class TestDetailScreen extends StatefulWidget {
  final String testObjectId;
  const TestDetailScreen({super.key, required this.testObjectId});

  @override
  State<TestDetailScreen> createState() => _TestDetailScreenState();
}

class _TestDetailScreenState extends State<TestDetailScreen> {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);

  static const Map<int, String> _violationNames = {
    0: "Safe Driving",
    1: "Tailgating",
    2: "Running a Stop Sign",
    3: "Failure to Yield",
    4: "No Entry Violation",
  };

  static const Map<int, IconData> _violationIcons = {
    0: Icons.check_circle_outline,
    1: Icons.directions_car,
    2: Icons.stop_circle_outlined,
    3: Icons.change_history,
    4: Icons.do_not_disturb_on,
  };

  Future<Map<String, dynamic>?>? _future;

  @override
  void initState() {
    super.initState();
    // ✅ הכרחה ל-portrait למנוע שאריות מ-LiveFeed (landscape)
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _future ??= _load();
  }

  Future<Map<String, dynamic>?> _load() async {
    final email = Provider.of<UserProvider>(context, listen: false).user?.email;
    if (email == null) return null;
    return ApiService.getTestDetail(widget.testObjectId);
  }

  /// בונה ציר זמן שלם מ-decision_log: גם הפרות, גם פעולות תקינות.
  /// אם אין decision_log (טסטים ישנים), נופל חזרה ל-xai_explanations בלבד.
  List<_TimelineEvent> _buildTimeline(Map<String, dynamic> data) {
    final List<dynamic> decisionLog = data['decision_log'] ?? [];

    // אם יש decision_log חדש - נשתמש בו (כולל פעולות תקינות בעצור וכו')
    if (decisionLog.isNotEmpty) {
      return _buildTimelineFromLog(decisionLog, data);
    }

    // נפילה לאחור: רק הפרות מ-XAI
    return _buildTimelineFromXAI(data);
  }

  /// בונה ציר זמן מלא מלוג ההחלטות (כל החלטה כולל "Safe Driving").
  List<_TimelineEvent> _buildTimelineFromLog(
    List<dynamic> log,
    Map<String, dynamic> data,
  ) {
    // ממזגים החלטות עוקבות של אותה קלסה לאירוע אחד
    final List<_TimelineEvent> events = [];
    if (log.isEmpty) return events;

    int currentCode = (log.first['predicted_class'] ?? 0) as int;
    double startSec = ((log.first['timestamp_sec'] ?? 0) as num).toDouble();
    double endSec = startSec;
    double minConfidence = ((log.first['confidence'] ?? 0) as num).toDouble();

    for (int i = 1; i < log.length; i++) {
      final entry = log[i];
      final int c = (entry['predicted_class'] ?? 0) as int;
      final double t = ((entry['timestamp_sec'] ?? 0) as num).toDouble();
      final double conf = ((entry['confidence'] ?? 0) as num).toDouble();

      if (c == currentCode) {
        endSec = t;
        if (conf < minConfidence) minConfidence = conf;
      } else {
        // סוגרים את האירוע הנוכחי
        final double duration = endSec - startSec;
        // ✅ אירוע תקין - דורש לפחות 3 שניות (אחרת רעש)
        // ✅ הפרה - דורש לפחות 0.5 שניות
        final double minDuration = currentCode == 0 ? 3.0 : 0.5;
        if (duration >= minDuration) {
          events.add(
            _TimelineEvent(
              timeSec: startSec,
              durationSec: duration,
              violationCode: currentCode,
              isViolation: currentCode != 0,
              confidence: minConfidence,
            ),
          );
        }
        currentCode = c;
        startSec = t;
        endSec = t;
        minConfidence = conf;
      }
    }

    // אירוע אחרון
    final double duration = endSec - startSec;
    final double minDuration = currentCode == 0 ? 3.0 : 0.5;
    if (duration >= minDuration) {
      events.add(
        _TimelineEvent(
          timeSec: startSec,
          durationSec: duration,
          violationCode: currentCode,
          isViolation: currentCode != 0,
          confidence: minConfidence,
        ),
      );
    }

    return events;
  }

  /// נפילה לאחור - רק אירועי הפרה מ-XAI (טסטים ישנים).
  List<_TimelineEvent> _buildTimelineFromXAI(Map<String, dynamic> data) {
    final Map<String, dynamic> explanations = data['xai_explanations'] ?? {};

    final List<Map<String, dynamic>> entries =
        explanations.values
            .map((e) => Map<String, dynamic>.from(e as Map))
            .where((e) => (e['violation_code'] ?? 0) != 0)
            .toList()
          ..sort(
            (a, b) => (a['timestamp_sec'] as num).compareTo(
              b['timestamp_sec'] as num,
            ),
          );

    final List<_TimelineEvent> events = [];
    if (entries.isEmpty) return events;

    double startSec = (entries.first['timestamp_sec'] as num).toDouble();
    double endSec = startSec;
    int code = entries.first['violation_code'] as int;

    for (int i = 1; i < entries.length; i++) {
      final entry = entries[i];
      final double t = (entry['timestamp_sec'] as num).toDouble();
      final int c = entry['violation_code'] as int;

      if (c == code && (t - endSec) < 3.0) {
        endSec = t;
      } else {
        events.add(
          _TimelineEvent(
            timeSec: startSec,
            durationSec: endSec - startSec,
            violationCode: code,
            isViolation: true,
          ),
        );
        startSec = t;
        endSec = t;
        code = c;
      }
    }
    events.add(
      _TimelineEvent(
        timeSec: startSec,
        durationSec: endSec - startSec,
        violationCode: code,
        isViolation: true,
      ),
    );

    return events;
  }

  /// אם אין אירועים בכלל (טסט נקי לגמרי) - מציג נקודה אחת ירוקה במרכז
  List<_TimelineEvent> _addSafePoints(
    List<_TimelineEvent> events,
    double totalDuration,
  ) {
    // עם decision_log אנחנו כבר מקבלים גם פעולות תקינות, לכן רק נטפל במצב ריק
    if (events.isEmpty) {
      return [
        _TimelineEvent(
          timeSec: totalDuration / 2,
          durationSec: totalDuration,
          violationCode: 0,
          isViolation: false,
        ),
      ];
    }
    return events;
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          "Test Details",
          style: GoogleFonts.lexend(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF314972), Color(0xFF233452)],
          ),
        ),
        child: SafeArea(
          child: FutureBuilder<Map<String, dynamic>?>(
            future: _future,
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(
                  child: CircularProgressIndicator(color: _primaryBlue),
                );
              }
              final data = snapshot.data;
              if (data == null) {
                return Center(
                  child: Text(
                    "Could not load test details",
                    style: GoogleFonts.lexend(color: Colors.white54),
                  ),
                );
              }
              return _buildContent(data);
            },
          ),
        ),
      ),
    );
  }

  Widget _buildContent(Map<String, dynamic> data) {
    final int grade = data['grade'] ?? 0;
    final bool passed = grade >= 80;
    final Color gradeColor = passed ? _activeGreen : _errorRed;
    final String studentName = data['student_name']?.toString() ?? 'Unknown';
    final String studentId = data['student_id']?.toString() ?? '';
    final String savedAt = data['saved_at']?.toString() ?? '';

    final violationEvents = _buildTimeline(data);
    final int violationsCount = violationEvents.length;

    // משך כולל - אם אין, נחשב מהאירוע האחרון
    double totalDuration = 60;
    if (violationEvents.isNotEmpty) {
      totalDuration = max(
        60,
        violationEvents.last.timeSec + violationEvents.last.durationSec + 10,
      );
    }

    final allEvents = _addSafePoints(violationEvents, totalDuration);

    String dateLabel = "";
    try {
      final dt = DateTime.parse(savedAt).toLocal();
      dateLabel =
          "${dt.day}/${dt.month}/${dt.year}  ${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
    } catch (_) {}

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 60, 20, 30),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ===== Header: שם, ID, ציון =====
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.05),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: gradeColor.withOpacity(0.3)),
            ),
            child: Row(
              children: [
                Container(
                  width: 70,
                  height: 70,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: gradeColor.withOpacity(0.15),
                    border: Border.all(color: gradeColor, width: 3),
                    boxShadow: [
                      BoxShadow(
                        color: gradeColor.withOpacity(0.25),
                        blurRadius: 20,
                        spreadRadius: 2,
                      ),
                    ],
                  ),
                  alignment: Alignment.center,
                  child: Text(
                    "$grade",
                    style: GoogleFonts.lexend(
                      color: gradeColor,
                      fontWeight: FontWeight.bold,
                      fontSize: 22,
                    ),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        studentName,
                        style: GoogleFonts.lexend(
                          color: Colors.white,
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                      const SizedBox(height: 2),
                      Text(
                        "ID: $studentId",
                        style: GoogleFonts.lexend(
                          color: Colors.white54,
                          fontSize: 12,
                        ),
                      ),
                      const SizedBox(height: 6),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 3,
                        ),
                        decoration: BoxDecoration(
                          color: gradeColor.withOpacity(0.15),
                          borderRadius: BorderRadius.circular(8),
                        ),
                        child: Text(
                          passed ? "PASSED" : "FAILED",
                          style: GoogleFonts.lexend(
                            color: gradeColor,
                            fontSize: 10,
                            fontWeight: FontWeight.bold,
                            letterSpacing: 1,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 12),
          if (dateLabel.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 4),
              child: Row(
                children: [
                  Icon(Icons.access_time, color: Colors.white38, size: 12),
                  const SizedBox(width: 6),
                  Text(
                    dateLabel,
                    style: GoogleFonts.lexend(
                      color: Colors.white38,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),

          const SizedBox(height: 30),

          // ===== כותרת ציר הזמן =====
          Row(
            children: [
              Text(
                "DRIVE TIMELINE",
                style: GoogleFonts.lexend(
                  color: Colors.white38,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.5,
                  fontSize: 12,
                ),
              ),
              const SizedBox(width: 10),
              Expanded(child: Container(height: 1, color: Colors.white12)),
              const SizedBox(width: 10),
              if (violationsCount > 0)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: _errorRed.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: _errorRed.withOpacity(0.5)),
                  ),
                  child: Text(
                    "$violationsCount violation${violationsCount == 1 ? '' : 's'}",
                    style: GoogleFonts.lexend(
                      color: _errorRed,
                      fontWeight: FontWeight.bold,
                      fontSize: 11,
                    ),
                  ),
                )
              else
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 10,
                    vertical: 4,
                  ),
                  decoration: BoxDecoration(
                    color: _activeGreen.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(20),
                    border: Border.all(color: _activeGreen.withOpacity(0.5)),
                  ),
                  child: Text(
                    "Clean drive",
                    style: GoogleFonts.lexend(
                      color: _activeGreen,
                      fontWeight: FontWeight.bold,
                      fontSize: 11,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: 20),

          // ===== ציר זמן =====
          ...allEvents.asMap().entries.map((entry) {
            final i = entry.key;
            final ev = entry.value;
            final isLast = i == allEvents.length - 1;
            return _buildTimelineNode(ev, isLast);
          }),
        ],
      ),
    );
  }

  Widget _buildTimelineNode(_TimelineEvent ev, bool isLast) {
    final Color color = ev.isViolation ? _errorRed : _activeGreen;
    final IconData icon = _violationIcons[ev.violationCode] ?? Icons.help;
    final String name = _violationNames[ev.violationCode] ?? "Unknown";
    final String timeLabel = "${ev.timeSec.toStringAsFixed(1)}s";

    return IntrinsicHeight(
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          // ===== עמודת הציר =====
          Column(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  shape: BoxShape.circle,
                  border: Border.all(color: color, width: 2),
                  boxShadow: [
                    BoxShadow(
                      color: color.withOpacity(0.3),
                      blurRadius: 8,
                      spreadRadius: 1,
                    ),
                  ],
                ),
                child: Icon(icon, color: color, size: 20),
              ),
              if (!isLast)
                Expanded(
                  child: Container(
                    width: 2,
                    margin: const EdgeInsets.symmetric(vertical: 4),
                    color: Colors.white12,
                  ),
                ),
            ],
          ),
          const SizedBox(width: 14),
          // ===== כרטיס =====
          Expanded(
            child: Padding(
              padding: const EdgeInsets.only(bottom: 14),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: Colors.white.withOpacity(0.04),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(
                    color: ev.isViolation
                        ? _errorRed.withOpacity(0.3)
                        : Colors.white12,
                  ),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Icon(Icons.access_time, color: _primaryBlue, size: 13),
                        const SizedBox(width: 5),
                        Text(
                          timeLabel,
                          style: GoogleFonts.lexend(
                            color: _primaryBlue,
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                        ),
                        if (ev.durationSec > 0.2) ...[
                          const SizedBox(width: 6),
                          Container(
                            padding: const EdgeInsets.symmetric(
                              horizontal: 7,
                              vertical: 2,
                            ),
                            decoration: BoxDecoration(
                              color: _primaryBlue.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(10),
                            ),
                            child: Text(
                              "${ev.durationSec.toStringAsFixed(1)}s",
                              style: GoogleFonts.lexend(
                                color: _primaryBlue,
                                fontSize: 9,
                                fontWeight: FontWeight.w500,
                              ),
                            ),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 6),
                    Text(
                      name,
                      style: GoogleFonts.lexend(
                        color: Colors.white,
                        fontSize: 15,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    if (!ev.isViolation) ...[
                      const SizedBox(height: 4),
                      Text(
                        "Driving within rules",
                        style: GoogleFonts.lexend(
                          color: Colors.white54,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _TimelineEvent {
  final double timeSec;
  final double durationSec;
  final int violationCode;
  final bool isViolation;
  final double confidence;

  _TimelineEvent({
    required this.timeSec,
    required this.durationSec,
    required this.violationCode,
    required this.isViolation,
    this.confidence = 0,
  });
}
