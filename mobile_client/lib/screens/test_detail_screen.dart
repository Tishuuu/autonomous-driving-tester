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

  static const Map<int, String> _positiveActionNames = {
    5: "Correct Stop",
    6: "Correct Yield",
  };

  static const Map<int, IconData> _positiveActionIcons = {
    5: Icons.stop_circle_outlined,
    6: Icons.check_circle_outline,
  };

  int _intFrom(dynamic value, {int fallback = 0}) {
    if (value is num) return value.toInt();
    return int.tryParse(value?.toString() ?? '') ?? fallback;
  }

  double _doubleFrom(dynamic value, {double fallback = 0.0}) {
    if (value is num) return value.toDouble();
    return double.tryParse(value?.toString() ?? '') ?? fallback;
  }

  int _xaiClassId(Map<String, dynamic> e) {
    return _intFrom(
      e['violation_code'] ?? e['class_id'] ?? e['predicted_class'],
      fallback: 0,
    );
  }

  bool _isFailClass(int code) => code == 2 || code == 3 || code == 4;


  bool _passedFrom(Map<String, dynamic> data) {
    if (data['passed'] is bool) return data['passed'] as bool;
    final result = data['result']?.toString().toUpperCase();
    if (result == 'PASS') return true;
    if (result == 'FAIL') return false;
    return (data['grade'] ?? 0) >= 80;
  }

  int _mistakesCountFrom(Map<String, dynamic> data) {
    final value = data['mistakes_count'] ?? data['violation_events_count'] ?? 0;
    return value is num ? value.toInt() : int.tryParse(value.toString()) ?? 0;
  }

  int _ignoredWarningsCountFrom(Map<String, dynamic> data) {
    final value = data['ignored_warning_events_count'] ?? 0;
    return value is num ? value.toInt() : int.tryParse(value.toString()) ?? 0;
  }

  List<dynamic> _mistakeCodesFrom(Map<String, dynamic> data) {
    return List<dynamic>.from(data['mistake_codes'] ?? data['violations_codes'] ?? []);
  }

  List<Map<String, dynamic>> _positiveActionsFrom(Map<String, dynamic> data) {
    final raw = data['positive_actions'];
    if (raw is! List) return [];

    final actions = raw
        .whereType<Map>()
        .map((e) {
          final action = Map<String, dynamic>.from(e);
          int classId = _intFrom(action['class_id'], fallback: -1);
          final type = action['type']?.toString() ?? '';
          if (classId < 0) {
            if (type == 'CorrectStop') classId = 5;
            if (type == 'CorrectYield') classId = 6;
          }
          action['class_id'] = classId;
          action['timestamp_sec'] = _doubleFrom(action['timestamp_sec']);
          action['confidence'] = _doubleFrom(action['confidence']);
          return action;
        })
        .where((a) => _positiveActionNames.containsKey(a['class_id']))
        .toList()
      ..sort(
        (a, b) => _doubleFrom(a['timestamp_sec']).compareTo(
          _doubleFrom(b['timestamp_sec']),
        ),
      );

    return actions;
  }

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

  /// Timeline shows only major mistakes.
  /// decision_log contains raw model states and ignored warnings, so we do not use it here.
  List<_TimelineEvent> _buildTimeline(Map<String, dynamic> data) {
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

  /// נפילה לאחור - רק אירועי הפרה מ-XAI.
  /// תומך גם בפורמט M11 הישן (violation_code) וגם ב-M12 החדש (class_id).
  List<_TimelineEvent> _buildTimelineFromXAI(Map<String, dynamic> data) {
    final Map<String, dynamic> explanations = data['xai_explanations'] ?? {};

    final List<Map<String, dynamic>> entries = explanations.values
        .whereType<Map>()
        .map((e) => Map<String, dynamic>.from(e))
        .where((e) => _isFailClass(_xaiClassId(e)))
        .map((e) {
          final copy = Map<String, dynamic>.from(e);
          copy['violation_code'] = _xaiClassId(copy);
          copy['timestamp_sec'] = _doubleFrom(copy['timestamp_sec']);
          return copy;
        })
        .toList()
      ..sort(
        (a, b) => _doubleFrom(a['timestamp_sec']).compareTo(
          _doubleFrom(b['timestamp_sec']),
        ),
      );

    final List<_TimelineEvent> events = [];
    if (entries.isEmpty) return events;

    double startSec = _doubleFrom(entries.first['timestamp_sec']);
    double endSec = startSec;
    int code = _xaiClassId(entries.first);

    for (int i = 1; i < entries.length; i++) {
      final entry = entries[i];
      final double t = _doubleFrom(entry['timestamp_sec']);
      final int c = _xaiClassId(entry);

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

  List<_TimelineEvent> _fallbackMistakeEvents(List<dynamic> mistakeCodes) {
    return mistakeCodes
        .map((c) => _intFrom(c, fallback: -1))
        .where(_isFailClass)
        .map(
          (code) => _TimelineEvent(
            timeSec: 0.0,
            durationSec: 0.0,
            violationCode: code,
            isViolation: true,
          ),
        )
        .toList();
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
    final bool passed = _passedFrom(data);
    final int mistakesCount = _mistakesCountFrom(data);
    final int ignoredWarningsCount = _ignoredWarningsCountFrom(data);
    final List<dynamic> mistakeCodes = _mistakeCodesFrom(data);
    final List<Map<String, dynamic>> positiveActions = _positiveActionsFrom(data);
    final Color gradeColor = passed ? _activeGreen : _errorRed;
    final String studentName = data['student_name']?.toString() ?? 'Unknown';
    final String studentId = data['student_id']?.toString() ?? '';
    final String savedAt = data['saved_at']?.toString() ?? '';

    List<_TimelineEvent> violationEvents = _buildTimeline(data);
    if (!passed && violationEvents.isEmpty) {
      violationEvents = _fallbackMistakeEvents(mistakeCodes);
    }
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
                  child: Icon(
                    passed ? Icons.verified_rounded : Icons.cancel_rounded,
                    color: gradeColor,
                    size: 34,
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
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                            decoration: BoxDecoration(
                              color: gradeColor.withOpacity(0.15),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Text(
                              passed ? "PASS" : "FAIL",
                              style: GoogleFonts.lexend(
                                color: gradeColor,
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1,
                              ),
                            ),
                          ),
                          const SizedBox(width: 8),
                          Flexible(
                            child: Text(
                              passed
                                  ? "No major mistakes"
                                  : "$mistakesCount major mistake${mistakesCount == 1 ? '' : 's'}",
                              overflow: TextOverflow.ellipsis,
                              style: GoogleFonts.lexend(
                                color: Colors.white70,
                                fontSize: 12,
                              ),
                            ),
                          ),
                        ],
                      ),
                      if (ignoredWarningsCount > 0) ...[
                        const SizedBox(height: 4),
                        Text(
                          "Ignored warnings: $ignoredWarningsCount",
                          style: GoogleFonts.lexend(
                            color: Colors.white38,
                            fontSize: 11,
                          ),
                        ),
                      ],
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

          if (positiveActions.isNotEmpty) ...[
            const SizedBox(height: 24),
            _buildPositiveActionsSection(positiveActions),
          ],

          const SizedBox(height: 30),

          // ===== כותרת ציר הזמן =====
          Row(
            children: [
              Text(
                "MAJOR MISTAKES",
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
                    "$mistakesCount mistake${mistakesCount == 1 ? '' : 's'}",
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
                    "No mistakes",
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

  Widget _buildPositiveActionsSection(List<Map<String, dynamic>> actions) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              "GOOD DRIVING",
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
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              decoration: BoxDecoration(
                color: _activeGreen.withOpacity(0.15),
                borderRadius: BorderRadius.circular(20),
                border: Border.all(color: _activeGreen.withOpacity(0.5)),
              ),
              child: Text(
                "${actions.length}",
                style: GoogleFonts.lexend(
                  color: _activeGreen,
                  fontWeight: FontWeight.bold,
                  fontSize: 11,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        ...actions.map(_buildPositiveActionCard),
      ],
    );
  }

  Widget _buildPositiveActionCard(Map<String, dynamic> action) {
    final int classId = _intFrom(action['class_id']);
    final String name = _positiveActionNames[classId] ??
        action['class_label']?.toString() ??
        action['type']?.toString() ??
        "Correct Action";
    final IconData icon = _positiveActionIcons[classId] ?? Icons.check_circle_outline;
    final double timestamp = _doubleFrom(action['timestamp_sec']);
    final double confidence = _doubleFrom(action['confidence']);
    final String confidenceText = confidence > 0
        ? " • ${(confidence * 100).clamp(0, 100).toStringAsFixed(0)}%"
        : "";

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _activeGreen.withOpacity(0.08),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: _activeGreen.withOpacity(0.35)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(9),
            decoration: BoxDecoration(
              color: _activeGreen.withOpacity(0.16),
              shape: BoxShape.circle,
            ),
            child: Icon(icon, color: _activeGreen, size: 20),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  name,
                  style: GoogleFonts.lexend(
                    color: Colors.white,
                    fontSize: 15,
                    fontWeight: FontWeight.w700,
                  ),
                ),
                const SizedBox(height: 3),
                Text(
                  "${timestamp.toStringAsFixed(1)}s$confidenceText",
                  style: GoogleFonts.lexend(color: Colors.white54, fontSize: 11),
                ),
              ],
            ),
          ),
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
