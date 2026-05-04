import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import '../services/api_service.dart';

class TestSummaryScreen extends StatefulWidget {
  final Map<String, dynamic> result;

  const TestSummaryScreen({super.key, required this.result});

  @override
  State<TestSummaryScreen> createState() => _TestSummaryScreenState();
}

class _TestSummaryScreenState extends State<TestSummaryScreen> {
  // ===== צבעי הפלטה (תואמים לדאשבורד) =====
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);

  // מפת העבירות לפי מודל 8
  static const Map<int, String> _violationNames = {
    0: "Safe Driving",
    1: "Tailgating (Following Too Close)",
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

  bool _isSaving = false;
  bool _saved = false;
  String? _savedStudentName;

  @override
  void initState() {
    super.initState();
    // ✅ הכרחה ל-portrait מיד, להגנה במקרה שהמסך הקודם השאיר landscape
    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);
  }

  // ==========================================
  // מיזוג אירועים עוקבים מאותו סוג לאירוע אחד
  // ==========================================
  List<Map<String, dynamic>> _groupEvents(Map<String, dynamic> explanations) {
    if (explanations.isEmpty) return [];

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

    if (entries.isEmpty) return [];

    final List<Map<String, dynamic>> grouped = [];
    Map<String, dynamic> current = Map<String, dynamic>.from(entries.first);
    current['start_sec'] = current['timestamp_sec'];
    current['end_sec'] = current['timestamp_sec'];

    for (int i = 1; i < entries.length; i++) {
      final entry = entries[i];
      final double t = (entry['timestamp_sec'] as num).toDouble();
      final int code = entry['violation_code'] as int;
      final double currentEnd = (current['end_sec'] as num).toDouble();
      final int currentCode = current['violation_code'] as int;

      if (code == currentCode && (t - currentEnd) < 3.0) {
        current['end_sec'] = t;
        final double currentScore = (current['attention_score'] as num)
            .toDouble();
        final double newScore = (entry['attention_score'] as num).toDouble();
        if (newScore > currentScore) {
          current['attention_array'] = entry['attention_array'];
          current['decisive_frame_in_window'] =
              entry['decisive_frame_in_window'];
          current['attention_score'] = entry['attention_score'];
          current['timestamp_sec'] = entry['timestamp_sec'];
        }
      } else {
        grouped.add(current);
        current = Map<String, dynamic>.from(entry);
        current['start_sec'] = current['timestamp_sec'];
        current['end_sec'] = current['timestamp_sec'];
      }
    }
    grouped.add(current);
    return grouped;
  }

  // ==========================================
  // 💾 פתיחת Bottom Sheet לבחירת/הוספת תלמיד
  // ==========================================
  Future<void> _openSaveSheet() async {
    final userProvider = Provider.of<UserProvider>(context, listen: false);
    final testerEmail = userProvider.user?.email;

    if (testerEmail == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(
          content: Text("You must be logged in to save tests"),
          backgroundColor: _errorRed,
        ),
      );
      return;
    }

    // 🔒 Guard: מונע פתיחה כפולה של ה-bottom sheet
    if (_isSaving || _saved) return;

    final selected = await showModalBottomSheet<Map<String, String>>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (ctx) => _StudentPickerSheet(testerEmail: testerEmail),
    );

    if (selected != null && mounted) {
      await _saveTest(
        studentId: selected['student_id']!,
        studentName: selected['name']!,
        testerEmail: testerEmail,
      );
    }
  }

  Future<void> _saveTest({
    required String studentId,
    required String studentName,
    required String testerEmail,
  }) async {
    // 🔒 Strict debounce: מונע ריצה כפולה (לחיצה מהירה / submit כפול / race)
    if (_isSaving || _saved) {
      print("⚠️ _saveTest call ignored (already saving or saved)");
      return;
    }
    setState(() => _isSaving = true);

    final success = await ApiService.saveTest(
      studentId: studentId,
      grade: widget.result['grade'] ?? 0,
      violationsCodes: widget.result['violations_codes'] ?? [],
      xaiExplanations: widget.result['xai_explanations'] ?? {},
      violationEventsCount: widget.result['violation_events_count'] ?? 0,
      windowsAnalyzed: widget.result['windows_analyzed'] ?? 0,
      testId: widget.result['test_id'],
      decisionLog: widget.result['decision_log'] ?? [],
      actionSequences: widget.result['action_sequences'] ?? [],
      positiveActions: widget.result['positive_actions'] ?? [],
    );

    if (!mounted) return;

    setState(() {
      _isSaving = false;
      _saved = success;
      if (success) _savedStudentName = studentName;
    });

    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(
          success
              ? "✅ Test saved for $studentName"
              : "❌ Failed to save test. Please try again.",
        ),
        backgroundColor: success ? _activeGreen : _errorRed,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final int grade = widget.result['grade'] ?? 0;
    final Map<String, dynamic> explanations =
        widget.result['xai_explanations'] ?? {};
    final List<Map<String, dynamic>> events = _groupEvents(explanations);

    return Scaffold(
      extendBodyBehindAppBar: true,
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF314972), Color(0xFF233452)],
          ),
        ),
        child: SafeArea(
          child: Stack(
            children: [
              CustomScrollView(
                slivers: [
                  SliverAppBar(
                    pinned: true,
                    backgroundColor: Colors.transparent,
                    elevation: 0,
                    title: Text(
                      "Test Summary",
                      style: GoogleFonts.lexend(
                        color: Colors.white,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    iconTheme: const IconThemeData(color: Colors.white),
                  ),

                  SliverToBoxAdapter(child: _buildGradeHeader(grade)),

                  if (events.isNotEmpty)
                    SliverToBoxAdapter(
                      child: Padding(
                        padding: const EdgeInsets.fromLTRB(24, 10, 24, 16),
                        child: Row(
                          children: [
                            Text(
                              "EVENTS TIMELINE",
                              style: GoogleFonts.lexend(
                                color: Colors.white38,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.5,
                                fontSize: 12,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Expanded(
                              child: Container(
                                height: 1,
                                color: Colors.white12,
                              ),
                            ),
                            const SizedBox(width: 10),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 10,
                                vertical: 4,
                              ),
                              decoration: BoxDecoration(
                                color: _errorRed.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(20),
                                border: Border.all(
                                  color: _errorRed.withOpacity(0.5),
                                ),
                              ),
                              child: Text(
                                "${events.length}",
                                style: GoogleFonts.lexend(
                                  color: _errorRed,
                                  fontWeight: FontWeight.bold,
                                  fontSize: 12,
                                ),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),

                  if (events.isEmpty)
                    SliverToBoxAdapter(child: _buildPerfectDriveCard())
                  else
                    SliverList(
                      delegate: SliverChildBuilderDelegate((context, index) {
                        return _buildTimelineCard(
                          events[index],
                          index == events.length - 1,
                        );
                      }, childCount: events.length),
                    ),

                  // ריווח כדי שהכפתור הצף לא יסתיר את התוכן
                  const SliverToBoxAdapter(child: SizedBox(height: 110)),
                ],
              ),

              // ===== כפתור Save צף בתחתית =====
              Positioned(
                left: 20,
                right: 20,
                bottom: 20,
                child: _buildSaveButton(),
              ),
            ],
          ),
        ),
      ),
    );
  }

  // ==========================================
  // ✅ כפתור Save (משתנה לפי מצב)
  // ==========================================
  Widget _buildSaveButton() {
    final Color color = _saved ? _activeGreen : _primaryBlue;
    final IconData icon = _saved ? Icons.check_circle : Icons.save_rounded;
    final String label = _saved ? "Saved for $_savedStudentName" : "Save Test";

    return Container(
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        boxShadow: [
          BoxShadow(
            color: color.withOpacity(0.4),
            blurRadius: 20,
            spreadRadius: 1,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: ElevatedButton(
        onPressed: (_saved || _isSaving) ? null : _openSaveSheet,
        style: ElevatedButton.styleFrom(
          backgroundColor: color,
          disabledBackgroundColor: color,
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 16),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(18),
          ),
          elevation: 0,
        ),
        child: _isSaving
            ? const SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  color: Colors.white,
                  strokeWidth: 2.5,
                ),
              )
            : Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(icon, color: Colors.white),
                  const SizedBox(width: 10),
                  Flexible(
                    child: Text(
                      label,
                      overflow: TextOverflow.ellipsis,
                      style: GoogleFonts.lexend(
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                        letterSpacing: 1,
                      ),
                    ),
                  ),
                ],
              ),
      ),
    );
  }

  // ==========================================
  // ✅ כותרת הציון
  // ==========================================
  Widget _buildGradeHeader(int grade) {
    final bool passed = grade >= 80;
    final Color gradeColor = passed ? _activeGreen : _errorRed;

    return Container(
      padding: const EdgeInsets.symmetric(vertical: 30, horizontal: 20),
      child: Column(
        children: [
          Stack(
            alignment: Alignment.center,
            children: [
              Container(
                width: 180,
                height: 180,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: gradeColor.withOpacity(0.25),
                      blurRadius: 30,
                      spreadRadius: 4,
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: 160,
                height: 160,
                child: CircularProgressIndicator(
                  value: grade / 100,
                  strokeWidth: 12,
                  backgroundColor: Colors.white10,
                  valueColor: AlwaysStoppedAnimation<Color>(gradeColor),
                ),
              ),
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    "$grade",
                    style: GoogleFonts.lexend(
                      fontSize: 48,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                      height: 1,
                    ),
                  ),
                  Text(
                    "/ 100",
                    style: GoogleFonts.lexend(
                      fontSize: 12,
                      color: Colors.white54,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 24),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
            decoration: BoxDecoration(
              color: gradeColor.withOpacity(0.15),
              borderRadius: BorderRadius.circular(30),
              border: Border.all(color: gradeColor.withOpacity(0.5)),
            ),
            child: Text(
              passed ? "PASSED" : "FAILED",
              style: GoogleFonts.lexend(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                color: gradeColor,
                letterSpacing: 3,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPerfectDriveCard() {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Container(
        padding: const EdgeInsets.all(30),
        decoration: BoxDecoration(
          color: _activeGreen.withOpacity(0.08),
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: _activeGreen.withOpacity(0.4)),
        ),
        child: Column(
          children: [
            Icon(Icons.verified_rounded, color: _activeGreen, size: 60),
            const SizedBox(height: 16),
            Text(
              "Perfect Drive!",
              style: GoogleFonts.lexend(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              "No violations detected during this test.",
              textAlign: TextAlign.center,
              style: GoogleFonts.lexend(color: Colors.white70, fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTimelineCard(Map<String, dynamic> data, bool isLast) {
    final int code = data['violation_code'] ?? 0;
    final double startSec = (data['start_sec'] as num).toDouble();
    final double endSec = (data['end_sec'] as num).toDouble();
    final List<double> attention = List<double>.from(
      data['attention_array'] ?? [],
    );
    final int peakFrame = data['decisive_frame_in_window'] ?? 0;
    final double duration = endSec - startSec;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: IntrinsicHeight(
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Column(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: _errorRed.withOpacity(0.2),
                    shape: BoxShape.circle,
                    border: Border.all(color: _errorRed, width: 2),
                    boxShadow: [
                      BoxShadow(
                        color: _errorRed.withOpacity(0.3),
                        blurRadius: 8,
                        spreadRadius: 1,
                      ),
                    ],
                  ),
                  child: Icon(
                    _violationIcons[code],
                    color: _errorRed,
                    size: 22,
                  ),
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
            const SizedBox(width: 16),
            Expanded(
              child: Padding(
                padding: const EdgeInsets.only(bottom: 16),
                child: Container(
                  padding: const EdgeInsets.all(16),
                  decoration: BoxDecoration(
                    color: Colors.white.withOpacity(0.05),
                    borderRadius: BorderRadius.circular(16),
                    border: Border.all(color: Colors.white12),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Row(
                        children: [
                          Icon(
                            Icons.access_time,
                            color: _primaryBlue,
                            size: 14,
                          ),
                          const SizedBox(width: 6),
                          Text(
                            "${startSec.toStringAsFixed(1)}s",
                            style: GoogleFonts.lexend(
                              color: _primaryBlue,
                              fontWeight: FontWeight.bold,
                              fontSize: 13,
                            ),
                          ),
                          if (duration > 0.2) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(
                                horizontal: 8,
                                vertical: 2,
                              ),
                              decoration: BoxDecoration(
                                color: _primaryBlue.withOpacity(0.15),
                                borderRadius: BorderRadius.circular(10),
                              ),
                              child: Text(
                                "${duration.toStringAsFixed(1)}s duration",
                                style: GoogleFonts.lexend(
                                  color: _primaryBlue,
                                  fontSize: 10,
                                  fontWeight: FontWeight.w500,
                                ),
                              ),
                            ),
                          ],
                        ],
                      ),
                      const SizedBox(height: 8),
                      Text(
                        _violationNames[code] ?? "Unknown Violation",
                        style: GoogleFonts.lexend(
                          color: Colors.white,
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 14),
                      Row(
                        children: [
                          Icon(
                            Icons.psychology_outlined,
                            color: Colors.white38,
                            size: 12,
                          ),
                          const SizedBox(width: 4),
                          Text(
                            "AI ATTENTION HEATMAP",
                            style: GoogleFonts.lexend(
                              color: Colors.white38,
                              fontSize: 10,
                              letterSpacing: 1,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                        ],
                      ),
                      const SizedBox(height: 8),
                      _AttentionChart(
                        values: attention,
                        decisiveIndex: peakFrame,
                        color: _errorRed,
                      ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

// ==========================================
// 📊 גרף ה-Attention (XAI)
// ==========================================
class _AttentionChart extends StatelessWidget {
  final List<double> values;
  final int decisiveIndex;
  final Color color;

  const _AttentionChart({
    required this.values,
    required this.decisiveIndex,
    required this.color,
  });

  @override
  Widget build(BuildContext context) {
    if (values.isEmpty) return const SizedBox.shrink();
    final double maxVal = values.reduce(max);
    return SizedBox(
      height: 45,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: List.generate(values.length, (i) {
          double hFactor = values[i] / (maxVal == 0 ? 1 : maxVal);
          final bool isPeak = i == decisiveIndex;
          return Expanded(
            child: Container(
              margin: const EdgeInsets.symmetric(horizontal: 0.8),
              height: max(3.0, hFactor * 45),
              decoration: BoxDecoration(
                color: isPeak ? color : color.withOpacity(0.25),
                borderRadius: BorderRadius.circular(2),
                boxShadow: isPeak
                    ? [
                        BoxShadow(
                          color: color.withOpacity(0.6),
                          blurRadius: 6,
                          spreadRadius: 0.5,
                        ),
                      ]
                    : [],
              ),
            ),
          );
        }),
      ),
    );
  }
}

// ==========================================================================
// 🎯 Bottom Sheet לבחירה/הוספה של תלמיד
// ==========================================================================
class _StudentPickerSheet extends StatefulWidget {
  final String testerEmail;
  const _StudentPickerSheet({required this.testerEmail});

  @override
  State<_StudentPickerSheet> createState() => _StudentPickerSheetState();
}

class _StudentPickerSheetState extends State<_StudentPickerSheet> {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);

  bool _showAddForm = false;
  bool _isLoading = true;
  bool _isSubmitting = false;

  List<Map<String, dynamic>> _students = [];
  List<Map<String, dynamic>> _filteredStudents = [];

  final TextEditingController _searchController = TextEditingController();
  final TextEditingController _newIdController = TextEditingController();
  final TextEditingController _newNameController = TextEditingController();
  String? _addError;

  @override
  void initState() {
    super.initState();
    _loadStudents();
    _searchController.addListener(_filter);
  }

  @override
  void dispose() {
    _searchController.dispose();
    _newIdController.dispose();
    _newNameController.dispose();
    super.dispose();
  }

  Future<void> _loadStudents() async {
    setState(() => _isLoading = true);
    final list = await ApiService.getStudents();
    if (!mounted) return;
    setState(() {
      _students = list;
      _filteredStudents = list;
      _isLoading = false;
    });
  }

  void _filter() {
    final q = _searchController.text.trim().toLowerCase();
    setState(() {
      if (q.isEmpty) {
        _filteredStudents = _students;
      } else {
        _filteredStudents = _students.where((s) {
          final name = (s['name'] ?? '').toString().toLowerCase();
          final id = (s['student_id'] ?? '').toString().toLowerCase();
          return name.contains(q) || id.contains(q);
        }).toList();
      }
    });
  }

  void _selectStudent(Map<String, dynamic> student) {
    // 🔒 Guard: מונע double-tap מהיר על תלמיד
    if (_isSubmitting) return;
    _isSubmitting = true;
    Navigator.pop(context, {
      'student_id': (student['student_id'] ?? '').toString(),
      'name': (student['name'] ?? '').toString(),
    });
  }

  Future<void> _submitNewStudent() async {
    // 🔒 Strict debounce על submit
    if (_isSubmitting) return;

    final id = _newIdController.text.trim();
    final name = _newNameController.text.trim();

    if (id.length < 5) {
      setState(() => _addError = "Student ID must be at least 5 digits");
      return;
    }
    if (name.length < 2) {
      setState(() => _addError = "Name must be at least 2 characters");
      return;
    }

    setState(() {
      _addError = null;
      _isSubmitting = true;
    });

    final result = await ApiService.addStudent(studentId: id, name: name);

    if (!mounted) return;

    if (result['success'] == true) {
      // סגירה ושליחת התלמיד החדש כתוצאת הסדר
      // (לא מאפסים _isSubmitting כי הסשן נסגר)
      Navigator.pop(context, {'student_id': id, 'name': name});
    } else {
      // נכשל - מאפסים כדי לאפשר ניסיון נוסף
      setState(() {
        _isSubmitting = false;
        _addError = result['error'] ?? "Unknown error";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final viewInsets = MediaQuery.of(context).viewInsets.bottom;

    return Padding(
      padding: EdgeInsets.only(bottom: viewInsets),
      child: Container(
        height: MediaQuery.of(context).size.height * 0.75,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF314972), Color(0xFF233452)],
          ),
          borderRadius: BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          children: [
            // ידית
            Container(
              margin: const EdgeInsets.only(top: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: Colors.white24,
                borderRadius: BorderRadius.circular(4),
              ),
            ),
            // כותרת
            Padding(
              padding: const EdgeInsets.fromLTRB(20, 16, 20, 8),
              child: Row(
                children: [
                  if (_showAddForm)
                    IconButton(
                      icon: const Icon(Icons.arrow_back, color: Colors.white),
                      onPressed: () => setState(() {
                        _showAddForm = false;
                        _addError = null;
                      }),
                    ),
                  Expanded(
                    child: Text(
                      _showAddForm ? "Add New Student" : "Select Student",
                      style: GoogleFonts.lexend(
                        color: Colors.white,
                        fontSize: 20,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            Expanded(child: _showAddForm ? _buildAddForm() : _buildList()),
          ],
        ),
      ),
    );
  }

  Widget _buildList() {
    return Column(
      children: [
        // שורת חיפוש
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 8, 20, 12),
          child: TextField(
            controller: _searchController,
            style: const TextStyle(color: Colors.white),
            decoration: InputDecoration(
              hintText: "Search by name or ID",
              hintStyle: GoogleFonts.lexend(color: Colors.white38),
              prefixIcon: const Icon(Icons.search, color: Colors.white54),
              filled: true,
              fillColor: Colors.white.withOpacity(0.05),
              border: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide(color: Colors.white12),
              ),
              enabledBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide(color: Colors.white12),
              ),
              focusedBorder: OutlineInputBorder(
                borderRadius: BorderRadius.circular(14),
                borderSide: BorderSide(color: _primaryBlue, width: 2),
              ),
            ),
          ),
        ),
        // רשימה
        Expanded(
          child: _isLoading
              ? const Center(
                  child: CircularProgressIndicator(color: _primaryBlue),
                )
              : _filteredStudents.isEmpty
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        Icons.people_outline,
                        color: Colors.white24,
                        size: 60,
                      ),
                      const SizedBox(height: 12),
                      Text(
                        _students.isEmpty
                            ? "No students yet.\nAdd your first one below."
                            : "No matches.",
                        textAlign: TextAlign.center,
                        style: GoogleFonts.lexend(
                          color: Colors.white54,
                          fontSize: 14,
                        ),
                      ),
                    ],
                  ),
                )
              : ListView.builder(
                  padding: const EdgeInsets.symmetric(horizontal: 20),
                  itemCount: _filteredStudents.length,
                  itemBuilder: (ctx, i) {
                    final student = _filteredStudents[i];
                    return _buildStudentTile(student);
                  },
                ),
        ),
        // כפתור הוספה
        Padding(
          padding: const EdgeInsets.fromLTRB(20, 12, 20, 24),
          child: ElevatedButton.icon(
            onPressed: () => setState(() => _showAddForm = true),
            icon: const Icon(Icons.add, color: _primaryBlue),
            label: Text(
              "Add New Student",
              style: GoogleFonts.lexend(
                color: _primaryBlue,
                fontSize: 15,
                fontWeight: FontWeight.bold,
              ),
            ),
            style: ElevatedButton.styleFrom(
              backgroundColor: _primaryBlue.withOpacity(0.15),
              minimumSize: const Size.fromHeight(50),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(14),
                side: BorderSide(color: _primaryBlue.withOpacity(0.5)),
              ),
              elevation: 0,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildStudentTile(Map<String, dynamic> student) {
    final name = (student['name'] ?? '').toString();
    final id = (student['student_id'] ?? '').toString();
    final initial = name.isNotEmpty ? name[0].toUpperCase() : '?';

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white12),
      ),
      child: ListTile(
        contentPadding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
        leading: CircleAvatar(
          backgroundColor: _primaryBlue.withOpacity(0.2),
          radius: 22,
          child: Text(
            initial,
            style: GoogleFonts.lexend(
              color: _primaryBlue,
              fontWeight: FontWeight.bold,
              fontSize: 18,
            ),
          ),
        ),
        title: Text(
          name,
          style: GoogleFonts.lexend(
            color: Colors.white,
            fontWeight: FontWeight.w600,
          ),
        ),
        subtitle: Text(
          "ID: $id",
          style: GoogleFonts.lexend(color: Colors.white54, fontSize: 12),
        ),
        trailing: const Icon(
          Icons.arrow_forward_ios,
          color: Colors.white38,
          size: 16,
        ),
        onTap: () => _selectStudent(student),
      ),
    );
  }

  Widget _buildAddForm() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(20, 8, 20, 24),
      child: Column(
        children: [
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white.withOpacity(0.04),
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.white12),
            ),
            child: Column(
              children: [
                _buildAddField(
                  controller: _newNameController,
                  label: "Full Name",
                  icon: Icons.person_outline,
                ),
                const SizedBox(height: 16),
                _buildAddField(
                  controller: _newIdController,
                  label: "Student ID",
                  icon: Icons.badge_outlined,
                  keyboardType: TextInputType.number,
                ),
                if (_addError != null) ...[
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Icon(Icons.error_outline, color: _errorRed, size: 18),
                      const SizedBox(width: 6),
                      Expanded(
                        child: Text(
                          _addError!,
                          style: GoogleFonts.lexend(
                            color: _errorRed,
                            fontSize: 13,
                          ),
                        ),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
          const Spacer(),
          ElevatedButton(
            onPressed: _isSubmitting ? null : _submitNewStudent,
            style: ElevatedButton.styleFrom(
              backgroundColor: _activeGreen,
              foregroundColor: Colors.black,
              minimumSize: const Size.fromHeight(54),
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(16),
              ),
              elevation: 0,
            ),
            child: _isSubmitting
                ? const SizedBox(
                    width: 22,
                    height: 22,
                    child: CircularProgressIndicator(
                      color: Colors.black,
                      strokeWidth: 2.5,
                    ),
                  )
                : Text(
                    "Add & Save Test",
                    style: GoogleFonts.lexend(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1,
                    ),
                  ),
          ),
        ],
      ),
    );
  }

  Widget _buildAddField({
    required TextEditingController controller,
    required String label,
    required IconData icon,
    TextInputType keyboardType = TextInputType.text,
  }) {
    return TextField(
      controller: controller,
      keyboardType: keyboardType,
      style: const TextStyle(color: Colors.white),
      cursorColor: _primaryBlue,
      decoration: InputDecoration(
        labelText: label,
        labelStyle: GoogleFonts.lexend(color: Colors.white54),
        prefixIcon: Icon(icon, color: _primaryBlue),
        filled: true,
        fillColor: Colors.white.withOpacity(0.04),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: Colors.white12),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: _primaryBlue, width: 2),
        ),
      ),
    );
  }
}
