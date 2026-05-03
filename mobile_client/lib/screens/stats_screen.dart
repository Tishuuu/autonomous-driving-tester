import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../providers/user_provider.dart';
import 'test_detail_screen.dart';

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);

  Future<List<dynamic>>? _historyFuture;
  String? _selectedStudentId; // null = כל התלמידים

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _historyFuture ??= _loadHistory();
  }

  Future<List<dynamic>> _loadHistory() async {
    final user = Provider.of<UserProvider>(context, listen: false).user;
    if (user == null) return [];
    return ApiService.getTesterHistory();
  }

  Future<void> _refresh() async {
    final f = _loadHistory();
    if (!mounted) return;
    setState(() {
      _historyFuture = f;
    });
    try {
      await f;
    } catch (_) {
      // FutureBuilder surfaces error
    }
  }

  /// מחזיר רשימה של תלמידים ייחודיים (מבוסס על הטסטים)
  List<Map<String, String>> _extractStudents(List<dynamic> tests) {
    final Map<String, String> studentsMap = {};
    for (final t in tests) {
      final id = (t['student_id'] ?? '').toString();
      final name = (t['student_name'] ?? '').toString();
      if (id.isNotEmpty && name.isNotEmpty) {
        studentsMap[id] = name;
      }
    }
    final list = studentsMap.entries
        .map((e) => {'id': e.key, 'name': e.value})
        .toList();
    list.sort((a, b) => a['name']!.compareTo(b['name']!));
    return list;
  }

  /// מסנן טסטים לפי תלמיד נבחר
  List<dynamic> _filterTests(List<dynamic> tests) {
    if (_selectedStudentId == null) return tests;
    return tests
        .where((t) => (t['student_id']?.toString() ?? '') == _selectedStudentId)
        .toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [Color(0xFF314972), Color(0xFF233452)],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 20, 20, 0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Image.asset('assets/images/logo.webp', height: 50),
                    IconButton(
                      icon: const Icon(Icons.refresh, color: Colors.white),
                      onPressed: _refresh,
                    ),
                  ],
                ),
                const SizedBox(height: 6),
                Text(
                  "Statistics",
                  style: GoogleFonts.lexend(
                    fontSize: 26,
                    fontWeight: FontWeight.bold,
                    color: _primaryBlue,
                    shadows: [Shadow(color: _primaryBlue, blurRadius: 10)],
                  ),
                ),
                const SizedBox(height: 16),
                Expanded(
                  child: FutureBuilder<List<dynamic>>(
                    future: _historyFuture,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const Center(
                          child: CircularProgressIndicator(color: _primaryBlue),
                        );
                      }
                      if (snapshot.hasError) {
                        return Center(
                          child: Text(
                            "Error loading data",
                            style: GoogleFonts.lexend(color: _errorRed),
                          ),
                        );
                      }

                      final allTests = snapshot.data ?? [];
                      if (allTests.isEmpty) {
                        return _buildEmptyState();
                      }

                      final students = _extractStudents(allTests);
                      final filtered = _filterTests(allTests);

                      return RefreshIndicator(
                        onRefresh: _refresh,
                        color: _primaryBlue,
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            // הסטטיסטיקה תמיד מבוססת על הסינון הנוכחי
                            _buildSummaryCards(filtered),
                            const SizedBox(height: 16),

                            // ===== Filter Chips: כל התלמידים בשורה גוללת =====
                            Text(
                              "FILTER BY STUDENT",
                              style: GoogleFonts.lexend(
                                color: Colors.white38,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.5,
                                fontSize: 11,
                              ),
                            ),
                            const SizedBox(height: 8),
                            SizedBox(
                              height: 38,
                              child: ListView(
                                scrollDirection: Axis.horizontal,
                                children: [
                                  _buildFilterChip(
                                    label: "All",
                                    icon: Icons.groups_outlined,
                                    isSelected: _selectedStudentId == null,
                                    onTap: () => setState(
                                      () => _selectedStudentId = null,
                                    ),
                                    count: allTests.length,
                                  ),
                                  ...students.map((s) {
                                    final id = s['id']!;
                                    final name = s['name']!;
                                    final count = allTests
                                        .where(
                                          (t) =>
                                              (t['student_id']?.toString() ??
                                                  '') ==
                                              id,
                                        )
                                        .length;
                                    return _buildFilterChip(
                                      label: name,
                                      icon: null,
                                      isSelected: _selectedStudentId == id,
                                      onTap: () => setState(
                                        () => _selectedStudentId = id,
                                      ),
                                      count: count,
                                    );
                                  }),
                                ],
                              ),
                            ),
                            const SizedBox(height: 16),

                            // ===== כותרת רשימה =====
                            Row(
                              children: [
                                Text(
                                  _selectedStudentId == null
                                      ? "ALL TESTS"
                                      : "STUDENT TESTS",
                                  style: GoogleFonts.lexend(
                                    color: Colors.white38,
                                    fontWeight: FontWeight.bold,
                                    letterSpacing: 1.5,
                                    fontSize: 12,
                                  ),
                                ),
                                const Spacer(),
                                Container(
                                  padding: const EdgeInsets.symmetric(
                                    horizontal: 8,
                                    vertical: 2,
                                  ),
                                  decoration: BoxDecoration(
                                    color: _primaryBlue.withOpacity(0.15),
                                    borderRadius: BorderRadius.circular(8),
                                  ),
                                  child: Text(
                                    "${filtered.length}",
                                    style: GoogleFonts.lexend(
                                      color: _primaryBlue,
                                      fontSize: 11,
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                              ],
                            ),
                            const SizedBox(height: 8),
                            Expanded(
                              child: filtered.isEmpty
                                  ? _buildNoMatch()
                                  : ListView.builder(
                                      itemCount: filtered.length,
                                      itemBuilder: (ctx, i) =>
                                          _buildTestCard(filtered[i]),
                                    ),
                            ),
                          ],
                        ),
                      );
                    },
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // ===== Filter Chip =====
  Widget _buildFilterChip({
    required String label,
    IconData? icon,
    required bool isSelected,
    required VoidCallback onTap,
    required int count,
  }) {
    final Color color = isSelected ? _primaryBlue : Colors.white24;
    final Color textColor = isSelected ? Colors.white : Colors.white70;

    return Padding(
      padding: const EdgeInsets.only(right: 8),
      child: GestureDetector(
        onTap: onTap,
        child: AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
          decoration: BoxDecoration(
            color: isSelected
                ? _primaryBlue.withOpacity(0.2)
                : Colors.white.withOpacity(0.04),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: color.withOpacity(0.5), width: 1.2),
            boxShadow: isSelected
                ? [
                    BoxShadow(
                      color: _primaryBlue.withOpacity(0.3),
                      blurRadius: 8,
                    ),
                  ]
                : [],
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              if (icon != null) ...[
                Icon(icon, color: textColor, size: 14),
                const SizedBox(width: 6),
              ],
              Text(
                label,
                style: GoogleFonts.lexend(
                  color: textColor,
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                ),
              ),
              const SizedBox(width: 6),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                decoration: BoxDecoration(
                  color: isSelected
                      ? _primaryBlue.withOpacity(0.3)
                      : Colors.white12,
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  "$count",
                  style: GoogleFonts.lexend(
                    color: textColor,
                    fontSize: 10,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSummaryCards(List<dynamic> tests) {
    final int total = tests.length;
    final int passed = tests.where((t) => (t['grade'] ?? 0) >= 80).length;
    final double avg = total == 0
        ? 0
        : tests.map((t) => (t['grade'] ?? 0) as num).reduce((a, b) => a + b) /
              total;
    final int passRate = total == 0 ? 0 : ((passed / total) * 100).round();

    return Row(
      children: [
        Expanded(
          child: _statCard(
            "Tests",
            "$total",
            Icons.assignment_outlined,
            _primaryBlue,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _statCard(
            "Avg Grade",
            avg.toStringAsFixed(0),
            Icons.bar_chart,
            avg >= 80 ? _activeGreen : _errorRed,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: _statCard(
            "Pass Rate",
            "$passRate%",
            Icons.verified_outlined,
            passRate >= 80 ? _activeGreen : _errorRed,
          ),
        ),
      ],
    );
  }

  Widget _statCard(String label, String value, IconData icon, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 14, horizontal: 8),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(15),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Column(
        children: [
          Icon(icon, color: color, size: 22),
          const SizedBox(height: 6),
          Text(
            value,
            style: GoogleFonts.lexend(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: GoogleFonts.lexend(color: Colors.white54, fontSize: 10),
          ),
        ],
      ),
    );
  }

  Widget _buildTestCard(dynamic test) {
    final int grade = test['grade'] ?? 0;
    final bool passed = grade >= 80;
    final String studentName = test['student_name']?.toString() ?? "Unknown";
    final String studentId = test['student_id']?.toString() ?? "";
    final String savedAt = test['saved_at']?.toString() ?? "";
    final String testId = test['_id']?.toString() ?? "";
    final Color color = passed ? _activeGreen : _errorRed;

    String dateLabel = "";
    try {
      final dt = DateTime.parse(savedAt).toLocal();
      dateLabel =
          "${dt.day.toString().padLeft(2, '0')}/${dt.month.toString().padLeft(2, '0')} "
          "${dt.hour.toString().padLeft(2, '0')}:${dt.minute.toString().padLeft(2, '0')}";
    } catch (_) {}

    return GestureDetector(
      onTap: testId.isEmpty
          ? null
          : () {
              Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (_) => TestDetailScreen(testObjectId: testId),
                ),
              );
            },
      child: Container(
        margin: const EdgeInsets.only(bottom: 10),
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(14),
          border: Border.all(color: Colors.white12),
        ),
        child: Row(
          children: [
            Container(
              width: 50,
              height: 50,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color.withOpacity(0.2),
                border: Border.all(color: color, width: 2),
              ),
              alignment: Alignment.center,
              child: Text(
                "$grade",
                style: GoogleFonts.lexend(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 16,
                ),
              ),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    studentName,
                    style: GoogleFonts.lexend(
                      color: Colors.white,
                      fontWeight: FontWeight.w600,
                      fontSize: 15,
                    ),
                  ),
                  const SizedBox(height: 2),
                  Text(
                    "ID: $studentId",
                    style: GoogleFonts.lexend(
                      color: Colors.white54,
                      fontSize: 11,
                    ),
                  ),
                ],
              ),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: 8,
                    vertical: 3,
                  ),
                  decoration: BoxDecoration(
                    color: color.withOpacity(0.15),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Text(
                    passed ? "PASSED" : "FAILED",
                    style: GoogleFonts.lexend(
                      color: color,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                      letterSpacing: 1,
                    ),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  dateLabel,
                  style: GoogleFonts.lexend(
                    color: Colors.white38,
                    fontSize: 10,
                  ),
                ),
              ],
            ),
            const SizedBox(width: 6),
            Icon(Icons.arrow_forward_ios, color: Colors.white24, size: 14),
          ],
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.bar_chart_rounded, color: Colors.white24, size: 70),
          const SizedBox(height: 14),
          Text(
            "No tests yet",
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            "Run a test from the dashboard to see stats here",
            textAlign: TextAlign.center,
            style: GoogleFonts.lexend(color: Colors.white38, fontSize: 13),
          ),
        ],
      ),
    );
  }

  Widget _buildNoMatch() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.search_off, color: Colors.white24, size: 50),
          const SizedBox(height: 10),
          Text(
            "No tests for this student yet",
            style: GoogleFonts.lexend(color: Colors.white54, fontSize: 14),
          ),
        ],
      ),
    );
  }
}
