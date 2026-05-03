import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import '../services/api_service.dart';
import '../providers/user_provider.dart';
import 'student_prediction_detail_screen.dart';

class PredictionsScreen extends StatefulWidget {
  const PredictionsScreen({super.key});

  @override
  State<PredictionsScreen> createState() => _PredictionsScreenState();
}

class _PredictionsScreenState extends State<PredictionsScreen> {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);
  static const Color _warningOrange = Color(0xFFFFA94C);

  // קיצורים לעבירות (כדי שתיכנס בכרטיס)
  static const Map<int, String> _violationShortNames = {
    1: "Tailgating",
    2: "Stop Sign",
    3: "Yielding",
    4: "No Entry",
  };

  static const Map<int, IconData> _violationIcons = {
    1: Icons.directions_car,
    2: Icons.stop_circle_outlined,
    3: Icons.change_history,
    4: Icons.do_not_disturb_on,
  };

  Future<List<Map<String, dynamic>>>? _future;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    _future ??= _load();
  }

  Future<List<Map<String, dynamic>>> _load() async {
    final user = Provider.of<UserProvider>(context, listen: false).user;
    if (user == null) return [];
    return ApiService.getAllPredictions();
  }

  Future<void> _refresh() async {
    final f = _load();
    if (!mounted) return;
    setState(() {
      _future = f;
    });
    try {
      await f;
    } catch (_) {
      // FutureBuilder surfaces error in snapshot
    }
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
                Row(
                  children: [
                    Icon(Icons.psychology, color: _primaryBlue, size: 28),
                    const SizedBox(width: 8),
                    Text(
                      "Predictions",
                      style: GoogleFonts.lexend(
                        fontSize: 26,
                        fontWeight: FontWeight.bold,
                        color: _primaryBlue,
                        shadows: [Shadow(color: _primaryBlue, blurRadius: 10)],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  "Driving test success forecast per student",
                  style: GoogleFonts.lexend(
                    color: Colors.white54,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 20),
                Expanded(
                  child: FutureBuilder<List<Map<String, dynamic>>>(
                    future: _future,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const Center(
                          child: CircularProgressIndicator(color: _primaryBlue),
                        );
                      }
                      final list = snapshot.data ?? [];
                      if (list.isEmpty) {
                        return _buildEmpty();
                      }
                      return RefreshIndicator(
                        onRefresh: _refresh,
                        color: _primaryBlue,
                        child: ListView.builder(
                          itemCount: list.length,
                          itemBuilder: (ctx, i) =>
                              _buildPredictionCard(list[i]),
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

  Widget _buildPredictionCard(Map<String, dynamic> data) {
    final String name = data['student_name']?.toString() ?? 'Unknown';
    final String id = data['student_id']?.toString() ?? '';
    final int? rate = data['predicted_success_rate'];
    final int testsCount = data['tests_count'] ?? 0;
    final String trend = data['trend']?.toString() ?? 'unknown';
    final List<dynamic> topViolations = data['top_violations'] ?? [];

    final Color color = rate == null
        ? Colors.white24
        : rate >= 80
        ? _activeGreen
        : rate >= 60
        ? _warningOrange
        : _errorRed;

    return GestureDetector(
      onTap: () {
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (_) =>
                StudentPredictionDetailScreen(studentId: id, studentName: name),
          ),
        );
      },
      child: Container(
        margin: const EdgeInsets.only(bottom: 14),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withOpacity(0.05),
          borderRadius: BorderRadius.circular(18),
          border: Border.all(color: color.withOpacity(0.4)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // ===== שורה ראשית: שם + תחזית =====
            Row(
              children: [
                // מעגל תחזית
                Stack(
                  alignment: Alignment.center,
                  children: [
                    SizedBox(
                      width: 56,
                      height: 56,
                      child: CircularProgressIndicator(
                        value: (rate ?? 0) / 100,
                        strokeWidth: 4.5,
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
                            color: color,
                            fontWeight: FontWeight.bold,
                            fontSize: 16,
                            height: 1,
                          ),
                        ),
                        Text(
                          "%",
                          style: GoogleFonts.lexend(
                            color: color.withOpacity(0.7),
                            fontSize: 9,
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
                const SizedBox(width: 14),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        style: GoogleFonts.lexend(
                          color: Colors.white,
                          fontWeight: FontWeight.w600,
                          fontSize: 16,
                        ),
                      ),
                      const SizedBox(height: 1),
                      Text(
                        "ID: $id",
                        style: GoogleFonts.lexend(
                          color: Colors.white54,
                          fontSize: 11,
                        ),
                      ),
                    ],
                  ),
                ),
                Icon(Icons.arrow_forward_ios, color: Colors.white24, size: 14),
              ],
            ),

            // ===== ה-2 פיסות מידע: מגמה + עבירות =====
            if (testsCount > 0) ...[
              const SizedBox(height: 14),
              const Divider(color: Colors.white12, height: 1),
              const SizedBox(height: 14),

              // === מגמה (שורה אחת בולטת) ===
              _buildTrendRow(trend, testsCount),

              // === עבירות נפוצות ===
              if (topViolations.isNotEmpty) ...[
                const SizedBox(height: 12),
                _buildViolationsSection(topViolations, testsCount),
              ] else ...[
                const SizedBox(height: 10),
                Row(
                  children: [
                    Icon(Icons.verified, color: _activeGreen, size: 14),
                    const SizedBox(width: 6),
                    Text(
                      "No recurring violations",
                      style: GoogleFonts.lexend(
                        color: _activeGreen,
                        fontSize: 11,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ],
            ] else
              Padding(
                padding: const EdgeInsets.only(top: 10),
                child: Text(
                  "No tests yet — run a test to enable predictions",
                  style: GoogleFonts.lexend(
                    color: Colors.white38,
                    fontSize: 11,
                    fontStyle: FontStyle.italic,
                  ),
                ),
              ),
          ],
        ),
      ),
    );
  }

  // ===== שורת מגמה =====
  Widget _buildTrendRow(String trend, int testsCount) {
    final IconData icon =
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

    final String label =
        {
          'improving': "Improving",
          'declining': "Declining",
          'stable': "Stable",
          'insufficient_data': "Insufficient data",
        }[trend] ??
        "Unknown";

    final String subtitle =
        {
          'improving': "Recent grades trending upward",
          'declining': "Recent grades trending downward",
          'stable': "Performance is consistent",
          'insufficient_data': "Need 3+ tests for trend",
        }[trend] ??
        "";

    return Row(
      children: [
        Container(
          padding: const EdgeInsets.all(7),
          decoration: BoxDecoration(
            color: trendColor.withOpacity(0.15),
            shape: BoxShape.circle,
          ),
          child: Icon(icon, color: trendColor, size: 16),
        ),
        const SizedBox(width: 10),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Text(
                    label,
                    style: GoogleFonts.lexend(
                      color: trendColor,
                      fontSize: 13,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(width: 6),
                  Text(
                    "•  $testsCount tests",
                    style: GoogleFonts.lexend(
                      color: Colors.white38,
                      fontSize: 10,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 1),
              Text(
                subtitle,
                style: GoogleFonts.lexend(color: Colors.white54, fontSize: 10),
              ),
            ],
          ),
        ),
      ],
    );
  }

  // ===== חלק העבירות הנפוצות =====
  Widget _buildViolationsSection(List<dynamic> violations, int testsCount) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Icon(
              Icons.report_problem_outlined,
              color: Colors.white38,
              size: 12,
            ),
            const SizedBox(width: 5),
            Text(
              "FOCUS AREAS",
              style: GoogleFonts.lexend(
                color: Colors.white38,
                fontSize: 9,
                fontWeight: FontWeight.bold,
                letterSpacing: 1,
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: violations.take(2).map<Widget>((v) {
            final int code = v['code'] ?? 0;
            final int count = v['count'] ?? 0;
            return Expanded(
              child: Padding(
                padding: EdgeInsets.only(
                  right: violations.indexOf(v) == 0 ? 8 : 0,
                ),
                child: _violationChip(code, count, testsCount),
              ),
            );
          }).toList(),
        ),
      ],
    );
  }

  Widget _violationChip(int code, int count, int testsCount) {
    final String name = _violationShortNames[code] ?? "Unknown";
    final IconData icon = _violationIcons[code] ?? Icons.warning_amber;
    final int frequencyPct = testsCount == 0
        ? 0
        : ((count / testsCount) * 100).round();

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 8),
      decoration: BoxDecoration(
        color: _errorRed.withOpacity(0.08),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: _errorRed.withOpacity(0.25)),
      ),
      child: Row(
        children: [
          Icon(icon, color: _errorRed, size: 14),
          const SizedBox(width: 6),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  name,
                  style: GoogleFonts.lexend(
                    color: Colors.white,
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                  ),
                  overflow: TextOverflow.ellipsis,
                ),
                Text(
                  "$count× ($frequencyPct%)",
                  style: GoogleFonts.lexend(
                    color: _errorRed,
                    fontSize: 9,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildEmpty() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.psychology_outlined, color: Colors.white24, size: 70),
          const SizedBox(height: 14),
          Text(
            "No students yet",
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 18,
              fontWeight: FontWeight.bold,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            "Run a test and add students to see predictions",
            textAlign: TextAlign.center,
            style: GoogleFonts.lexend(color: Colors.white38, fontSize: 13),
          ),
        ],
      ),
    );
  }
}
