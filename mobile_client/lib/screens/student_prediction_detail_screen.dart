import 'dart:math';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../providers/user_provider.dart';

class StudentPredictionDetailScreen extends StatefulWidget {
  final String studentId;
  final String studentName;

  const StudentPredictionDetailScreen({
    super.key,
    required this.studentId,
    required this.studentName,
  });

  @override
  State<StudentPredictionDetailScreen> createState() =>
      _StudentPredictionDetailScreenState();
}

class _StudentPredictionDetailScreenState
    extends State<StudentPredictionDetailScreen> {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);
  static const Color _warningOrange = Color(0xFFFFA94C);

  static const Map<int, String> _violationNames = {
    1: "Tailgating",
    2: "Running Stop Signs",
    3: "Failure to Yield",
    4: "No Entry Violations",
  };

  static const Map<int, IconData> _violationIcons = {
    1: Icons.directions_car,
    2: Icons.stop_circle_outlined,
    3: Icons.change_history,
    4: Icons.do_not_disturb_on,
  };

  Future<Map<String, dynamic>?>? _future;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _future ??= _load();
  }

  Future<Map<String, dynamic>?> _load() async {
    final email = Provider.of<UserProvider>(context, listen: false).user?.email;
    if (email == null) return null;
    return ApiService.getStudentPrediction(widget.studentId);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      extendBodyBehindAppBar: true,
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        title: Text(
          "Prediction Analysis",
          style: GoogleFonts.lexend(
            color: Colors.white,
            fontWeight: FontWeight.w600,
            fontSize: 16,
          ),
        ),
        iconTheme: const IconThemeData(color: Colors.white),
      ),
      body: Container(
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
                    "Could not load prediction",
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
    final int testsCount = data['tests_count'] ?? 0;
    final int? rate = data['predicted_success_rate'];
    final String trend = data['trend']?.toString() ?? 'unknown';
    final String confidence = data['confidence']?.toString() ?? 'no_data';
    final num avg = data['average_grade'] ?? 0;
    final List<dynamic> lastGrades = data['last_grades'] ?? [];
    final List<dynamic> weakest = data['weakest_violations'] ?? [];
    final String recommendation = data['recommendation']?.toString() ?? '';

    if (testsCount == 0) {
      return _buildNoData();
    }

    final Color color = rate == null
        ? Colors.white24
        : rate >= 80
        ? _activeGreen
        : rate >= 60
        ? _warningOrange
        : _errorRed;

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 60, 20, 30),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // ===== Section 1: ה-Headline (תחזית) =====
          _buildPredictionHero(rate, color, confidence),

          const SizedBox(height: 20),

          // ===== Section 2: המלצת ה-AI =====
          _buildRecommendationCard(recommendation, color),

          const SizedBox(height: 28),

          // ===== Section 3: ניתוח מגמה =====
          _buildSectionTitle("PERFORMANCE TREND", Icons.timeline),
          const SizedBox(height: 12),
          _buildTrendCard(trend, lastGrades, avg),

          const SizedBox(height: 28),

          // ===== Section 4: זיהוי תחומי חולשה =====
          _buildSectionTitle("WEAKNESS DETECTION", Icons.warning_amber_rounded),
          const SizedBox(height: 12),
          if (weakest.isNotEmpty)
            ...weakest.map(
              (w) => _buildWeaknessCard(
                w['code'] ?? 0,
                w['count'] ?? 0,
                testsCount,
              ),
            )
          else
            _buildNoWeaknessCard(),

          const SizedBox(height: 28),

          // ===== Section 5: מידע על האנליזה =====
          _buildSectionTitle("ABOUT THIS PREDICTION", Icons.info_outline),
          const SizedBox(height: 12),
          _buildInfoCard(testsCount, confidence),
        ],
      ),
    );
  }

  Widget _buildSectionTitle(String label, IconData icon) {
    return Row(
      children: [
        Icon(icon, color: Colors.white38, size: 14),
        const SizedBox(width: 6),
        Text(
          label,
          style: GoogleFonts.lexend(
            color: Colors.white38,
            fontWeight: FontWeight.bold,
            letterSpacing: 1.5,
            fontSize: 11,
          ),
        ),
        const SizedBox(width: 10),
        Expanded(child: Container(height: 1, color: Colors.white12)),
      ],
    );
  }

  // ===== Hero: תחזית מרכזית =====
  Widget _buildPredictionHero(int? rate, Color color, String confidence) {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.05),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withOpacity(0.4)),
      ),
      child: Column(
        children: [
          Text(
            widget.studentName,
            style: GoogleFonts.lexend(
              color: Colors.white,
              fontSize: 20,
              fontWeight: FontWeight.bold,
            ),
          ),
          Text(
            "ID: ${widget.studentId}",
            style: GoogleFonts.lexend(color: Colors.white54, fontSize: 11),
          ),
          const SizedBox(height: 18),
          Stack(
            alignment: Alignment.center,
            children: [
              Container(
                width: 150,
                height: 150,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  boxShadow: [
                    BoxShadow(
                      color: color.withOpacity(0.25),
                      blurRadius: 25,
                      spreadRadius: 3,
                    ),
                  ],
                ),
              ),
              SizedBox(
                width: 130,
                height: 130,
                child: CircularProgressIndicator(
                  value: (rate ?? 0) / 100,
                  strokeWidth: 10,
                  backgroundColor: Colors.white10,
                  valueColor: AlwaysStoppedAnimation<Color>(color),
                ),
              ),
              Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    rate != null ? "$rate" : "—",
                    style: GoogleFonts.lexend(
                      color: Colors.white,
                      fontSize: 38,
                      fontWeight: FontWeight.bold,
                      height: 1,
                    ),
                  ),
                  Text(
                    "% chance",
                    style: GoogleFonts.lexend(
                      color: Colors.white54,
                      fontSize: 10,
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 14),
          Text(
            "of passing the next driving test",
            style: GoogleFonts.lexend(color: Colors.white70, fontSize: 13),
          ),
        ],
      ),
    );
  }

  // ===== המלצת AI =====
  Widget _buildRecommendationCard(String text, Color color) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.15),
              shape: BoxShape.circle,
            ),
            child: Icon(Icons.lightbulb_outline, color: color, size: 18),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "AI Recommendation",
                  style: GoogleFonts.lexend(
                    color: color,
                    fontSize: 11,
                    fontWeight: FontWeight.bold,
                    letterSpacing: 1,
                  ),
                ),
                const SizedBox(height: 6),
                Text(
                  text,
                  style: GoogleFonts.lexend(
                    color: Colors.white,
                    fontSize: 14,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===== ניתוח מגמה =====
  Widget _buildTrendCard(String trend, List<dynamic> lastGrades, num avg) {
    final IconData trendIcon =
        {
          'improving': Icons.trending_up,
          'declining': Icons.trending_down,
          'stable': Icons.trending_flat,
        }[trend] ??
        Icons.help_outline;

    final Color trendColor =
        {
          'improving': _activeGreen,
          'declining': _errorRed,
          'stable': _primaryBlue,
        }[trend] ??
        Colors.white54;

    final String trendLabel =
        {
          'improving': "Improving",
          'declining': "Declining",
          'stable': "Stable",
          'insufficient_data': "Insufficient Data",
        }[trend] ??
        "Unknown";

    final String trendDescription =
        {
          'improving':
              "Performance is rising. Recent grades higher than earlier ones.",
          'declining':
              "Recent performance has dropped. Focus on consistent practice.",
          'stable': "Performance is consistent across recent tests.",
          'insufficient_data': "Need at least 3 tests for trend analysis.",
        }[trend] ??
        "Trend not available.";

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: trendColor.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: Icon(trendIcon, color: trendColor, size: 22),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      trendLabel,
                      style: GoogleFonts.lexend(
                        color: trendColor,
                        fontSize: 16,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 2),
                    Text(
                      "Average: ${avg.toStringAsFixed(0)}",
                      style: GoogleFonts.lexend(
                        color: Colors.white54,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            trendDescription,
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 12,
              height: 1.4,
            ),
          ),
          if (lastGrades.isNotEmpty) ...[
            const SizedBox(height: 16),
            Text(
              "Recent grades (oldest to newest):",
              style: GoogleFonts.lexend(color: Colors.white54, fontSize: 10),
            ),
            const SizedBox(height: 8),
            _buildGradesChart(lastGrades),
          ],
        ],
      ),
    );
  }

  Widget _buildGradesChart(List<dynamic> grades) {
    final List<num> nums = grades.map((g) => g as num).toList();
    const num maxGrade = 100;
    const double chartHeight = 110; // ✅ גובה כולל מספיק לכל התוכן
    const double maxBarHeight = 70; // ✅ עמודה מקסימלית - משאיר מקום לטקסט

    return SizedBox(
      height: chartHeight,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          // ציר Y
          SizedBox(
            width: 24,
            height: chartHeight,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                const SizedBox(height: 14), // align עם הטקסט של המספר
                Text(
                  "100",
                  style: GoogleFonts.lexend(color: Colors.white24, fontSize: 9),
                ),
                const Spacer(),
                Text(
                  "0",
                  style: GoogleFonts.lexend(color: Colors.white24, fontSize: 9),
                ),
              ],
            ),
          ),
          const SizedBox(width: 6),
          ...nums.asMap().entries.map((e) {
            final i = e.key;
            final g = e.value;
            final h = (g / maxGrade) * maxBarHeight;
            final passed = g >= 80;
            final isLatest = i == nums.length - 1;
            return Expanded(
              child: Padding(
                padding: const EdgeInsets.symmetric(horizontal: 3),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  mainAxisAlignment: MainAxisAlignment.end,
                  children: [
                    Text(
                      "$g",
                      style: GoogleFonts.lexend(
                        color: passed ? _activeGreen : _errorRed,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Container(
                      height: max(4, h),
                      decoration: BoxDecoration(
                        color: passed
                            ? _activeGreen.withOpacity(isLatest ? 1 : 0.5)
                            : _errorRed.withOpacity(isLatest ? 1 : 0.5),
                        borderRadius: const BorderRadius.vertical(
                          top: Radius.circular(4),
                        ),
                        boxShadow: isLatest
                            ? [
                                BoxShadow(
                                  color: (passed ? _activeGreen : _errorRed)
                                      .withOpacity(0.5),
                                  blurRadius: 6,
                                ),
                              ]
                            : [],
                      ),
                    ),
                  ],
                ),
              ),
            );
          }),
        ],
      ),
    );
  }

  // ===== כרטיס חולשה =====
  Widget _buildWeaknessCard(int code, int count, int totalTests) {
    final String name = _violationNames[code] ?? "Unknown Violation";
    final IconData icon = _violationIcons[code] ?? Icons.warning_amber;
    final double frequency = totalTests == 0 ? 0 : count / totalTests;
    final int frequencyPct = (frequency * 100).round();

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: _errorRed.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _errorRed.withOpacity(0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: _errorRed.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: Icon(icon, color: _errorRed, size: 18),
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
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      "Occurred in $count of $totalTests tests ($frequencyPct%)",
                      style: GoogleFonts.lexend(
                        color: Colors.white54,
                        fontSize: 11,
                      ),
                    ),
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: _errorRed.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  "$count×",
                  style: GoogleFonts.lexend(
                    color: _errorRed,
                    fontSize: 12,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: frequency,
              minHeight: 6,
              backgroundColor: Colors.white10,
              valueColor: const AlwaysStoppedAnimation<Color>(_errorRed),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoWeaknessCard() {
    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: _activeGreen.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: _activeGreen.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Icon(Icons.verified, color: _activeGreen, size: 24),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  "No recurring weaknesses",
                  style: GoogleFonts.lexend(
                    color: _activeGreen,
                    fontWeight: FontWeight.bold,
                    fontSize: 14,
                  ),
                ),
                const SizedBox(height: 2),
                Text(
                  "No specific violation pattern detected",
                  style: GoogleFonts.lexend(
                    color: Colors.white54,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  // ===== מידע על האנליזה =====
  Widget _buildInfoCard(int testsCount, String confidence) {
    final Color confColor = confidence == 'high'
        ? _activeGreen
        : confidence == 'medium'
        ? _primaryBlue
        : Colors.white38;

    final String confDescription =
        {
          'high': "Based on 5+ tests. High accuracy.",
          'medium': "Based on 3-4 tests. Moderate accuracy.",
          'low': "Based on 1-2 tests. Limited accuracy.",
        }[confidence] ??
        "Insufficient data for prediction.";

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.04),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: Colors.white12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.shield_outlined, color: confColor, size: 16),
              const SizedBox(width: 8),
              Text(
                "${confidence[0].toUpperCase()}${confidence.substring(1)} Confidence",
                style: GoogleFonts.lexend(
                  color: confColor,
                  fontSize: 13,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const Spacer(),
              Text(
                "$testsCount test${testsCount == 1 ? '' : 's'}",
                style: GoogleFonts.lexend(color: Colors.white54, fontSize: 11),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            confDescription,
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 12,
              height: 1.4,
            ),
          ),
          const SizedBox(height: 12),
          const Divider(color: Colors.white12, height: 1),
          const SizedBox(height: 10),
          Text(
            "How predictions work",
            style: GoogleFonts.lexend(
              color: Colors.white54,
              fontSize: 10,
              fontWeight: FontWeight.bold,
              letterSpacing: 1,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            "The AI analyzes weighted average grades, recent performance trend, and recurring violation patterns to predict success likelihood.",
            style: GoogleFonts.lexend(
              color: Colors.white60,
              fontSize: 11,
              height: 1.4,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildNoData() {
    return Center(
      child: Padding(
        padding: const EdgeInsets.all(32),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.psychology_outlined, color: Colors.white24, size: 70),
            const SizedBox(height: 16),
            Text(
              widget.studentName,
              style: GoogleFonts.lexend(
                color: Colors.white,
                fontSize: 22,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 8),
            Text(
              "No tests yet for this student",
              style: GoogleFonts.lexend(color: Colors.white54),
            ),
            const SizedBox(height: 4),
            Text(
              "Run a test to enable predictions",
              textAlign: TextAlign.center,
              style: GoogleFonts.lexend(color: Colors.white38, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}
