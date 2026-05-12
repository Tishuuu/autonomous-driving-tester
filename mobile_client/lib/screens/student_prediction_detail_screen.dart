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
    2: "Running Stop",
    3: "Failure to Yield",
    4: "No Entry Violation",
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

  int _asInt(dynamic value, {int fallback = 0}) {
    if (value == null) return fallback;
    if (value is int) return value;
    if (value is num) return value.round();
    return int.tryParse(value.toString()) ?? fallback;
  }

  double _asDouble(dynamic value, {double fallback = 0.0}) {
    if (value == null) return fallback;
    if (value is num) return value.toDouble();
    return double.tryParse(value.toString()) ?? fallback;
  }

  List<dynamic> _asList(dynamic value) {
    if (value is List) return value;
    return const [];
  }

  Map<String, dynamic> _asMap(dynamic value) {
    if (value is Map<String, dynamic>) return value;
    if (value is Map) return value.map((key, val) => MapEntry(key.toString(), val));
    return <String, dynamic>{};
  }

  String _confidenceLabel(String confidence) {
    switch (confidence) {
      case 'high':
        return 'High Confidence';
      case 'medium':
        return 'Medium Confidence';
      case 'low':
        return 'Low Confidence';
      case 'no_model':
        return 'Model Not Ready';
      case 'no_data':
        return 'No Data';
      default:
        return 'Limited Confidence';
    }
  }

  Color _confidenceColor(String confidence) {
    switch (confidence) {
      case 'high':
        return _activeGreen;
      case 'medium':
        return _primaryBlue;
      case 'low':
        return _warningOrange;
      case 'no_model':
        return _errorRed;
      default:
        return Colors.white38;
    }
  }

  Color _scoreColor(int? rate) {
    if (rate == null) return Colors.white24;
    if (rate >= 75) return _activeGreen;
    if (rate >= 50) return _warningOrange;
    return _errorRed;
  }

  Color _riskColor(String level, int percentile) {
    if (level == 'high' || percentile >= 75) return _errorRed;
    if (level == 'medium' || percentile >= 50) return _warningOrange;
    return _primaryBlue;
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
    final int testsCount = _asInt(data['tests_count']);
    final int? rate = data['predicted_success_rate'] == null
        ? null
        : _asInt(data['predicted_success_rate']);
    final String trend = data['trend']?.toString() ?? 'unknown';
    final String confidence = data['confidence']?.toString() ?? 'no_data';
    final num avg = data['average_grade'] is num ? data['average_grade'] as num : 0;
    final List<dynamic> lastGrades = _asList(data['last_grades']);
    final List<dynamic> riskPredictions = _asList(data['risk_predictions']);
    final List<dynamic> weakest = riskPredictions.isNotEmpty
        ? riskPredictions
        : _asList(data['weakest_violations']);
    final String recommendation = data['recommendation']?.toString() ?? '';
    final Map<String, dynamic> modelInfo = _asMap(data['prediction_model']);

    if (testsCount == 0) {
      return _buildNoData();
    }

    final Color color = _scoreColor(rate);

    return SingleChildScrollView(
      padding: const EdgeInsets.fromLTRB(20, 60, 20, 30),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildPredictionHero(rate, color, confidence, modelInfo),

          const SizedBox(height: 20),
          _buildRecommendationCard(recommendation, color),

          const SizedBox(height: 28),
          _buildSectionTitle("MODEL RISK RANKING", Icons.psychology_alt_outlined),
          const SizedBox(height: 12),
          if (weakest.isNotEmpty)
            ...weakest.map((risk) => _buildRiskCard(_asMap(risk), testsCount))
          else
            _buildNoWeaknessCard(),

          const SizedBox(height: 28),
          _buildSectionTitle("PERFORMANCE TREND", Icons.timeline),
          const SizedBox(height: 12),
          _buildTrendCard(trend, lastGrades, avg),

          const SizedBox(height: 28),
          _buildSectionTitle("ABOUT THIS PREDICTION", Icons.info_outline),
          const SizedBox(height: 12),
          _buildInfoCard(testsCount, confidence, modelInfo),
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

  Widget _buildPredictionHero(
    int? rate,
    Color color,
    String confidence,
    Map<String, dynamic> modelInfo,
  ) {
    final String modelStatus = modelInfo['model_status']?.toString() ?? 'unknown';
    final bool hasModel = modelStatus == 'ok' || rate != null;
    final dynamic rawProbability = modelInfo['raw_pass_probability'];
    final String threshold = modelInfo['pass_threshold'] == null
        ? ''
        : _asDouble(modelInfo['pass_threshold']).toStringAsFixed(2);

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
          const SizedBox(height: 10),
          _buildStatusChip(_confidenceLabel(confidence), _confidenceColor(confidence)),
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
                  value: rate == null ? 0 : rate.clamp(0, 100) / 100,
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
                    "% readiness",
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
            hasModel
                ? "estimated chance of passing if tested now"
                : "student prediction model is not available yet",
            textAlign: TextAlign.center,
            style: GoogleFonts.lexend(color: Colors.white70, fontSize: 13),
          ),
          if (rawProbability != null || threshold.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              [
                if (rawProbability != null)
                  "raw=${_asDouble(rawProbability).toStringAsFixed(3)}",
                if (threshold.isNotEmpty) "threshold=$threshold",
              ].join("  •  "),
              style: GoogleFonts.lexend(color: Colors.white38, fontSize: 10),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusChip(String label, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: color.withOpacity(0.14),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(color: color.withOpacity(0.35)),
      ),
      child: Text(
        label,
        style: GoogleFonts.lexend(
          color: color,
          fontSize: 10,
          fontWeight: FontWeight.bold,
          letterSpacing: 0.4,
        ),
      ),
    );
  }

  Widget _buildRecommendationCard(String text, Color color) {
    final String safeText = text.trim().isEmpty
        ? "Run more saved tests to improve the student readiness prediction."
        : text.trim();

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
                  safeText,
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

  Widget _buildRiskCard(Map<String, dynamic> risk, int totalTests) {
    final int code = _asInt(risk['code']);
    final String label = risk['label']?.toString() ??
        _violationNames[code] ??
        "Unknown Violation";
    final IconData icon = _violationIcons[code] ?? Icons.warning_amber;

    final int historyCount = _asInt(risk['history_count'] ?? risk['count']);
    final double rawRisk = _asDouble(risk['risk'], fallback: -1.0);
    final int riskPercent = risk['risk_percent'] == null
        ? (rawRisk >= 0 ? (rawRisk * 100).round() : 0)
        : _asInt(risk['risk_percent']);
    final String riskLevel = risk['risk_level']?.toString() ??
        (riskPercent >= 75 ? 'high' : riskPercent >= 50 ? 'medium' : 'low');
    final Color color = _riskColor(riskLevel, riskPercent);

    final double progressValue = (riskPercent.clamp(0, 100)) / 100.0;
    final String levelLabel = riskLevel.isEmpty
        ? 'Unknown'
        : '${riskLevel[0].toUpperCase()}${riskLevel.substring(1)}';

    final String subtitle = rawRisk >= 0
        ? "Risk percentile: $riskPercent% • Model score: ${rawRisk.toStringAsFixed(2)}"
        : "Seen in $historyCount of $totalTests tests";

    return Container(
      margin: const EdgeInsets.only(bottom: 10),
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: color.withOpacity(0.08),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withOpacity(0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.15),
                  shape: BoxShape.circle,
                ),
                child: Icon(icon, color: color, size: 18),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      label,
                      style: GoogleFonts.lexend(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    Text(
                      subtitle,
                      style: GoogleFonts.lexend(
                        color: Colors.white54,
                        fontSize: 11,
                      ),
                    ),
                    if (historyCount > 0) ...[
                      const SizedBox(height: 2),
                      Text(
                        "Historical count: $historyCount",
                        style: GoogleFonts.lexend(
                          color: Colors.white38,
                          fontSize: 10,
                        ),
                      ),
                    ],
                  ],
                ),
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  levelLabel,
                  style: GoogleFonts.lexend(
                    color: color,
                    fontSize: 11,
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
              value: progressValue,
              minHeight: 6,
              backgroundColor: Colors.white10,
              valueColor: AlwaysStoppedAnimation<Color>(color),
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
                  "No specific risk pattern detected by the model.",
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
              "Performance is rising. Recent tests look stronger than earlier ones.",
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
    final List<num> nums = grades
        .whereType<num>()
        .map((g) => g)
        .toList();
    if (nums.isEmpty) return const SizedBox.shrink();

    const num maxGrade = 100;
    const double chartHeight = 110;
    const double maxBarHeight = 70;

    return SizedBox(
      height: chartHeight,
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.end,
        children: [
          SizedBox(
            width: 24,
            height: chartHeight,
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.end,
              children: [
                const SizedBox(height: 14),
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
                      g.toStringAsFixed(0),
                      style: GoogleFonts.lexend(
                        color: passed ? _activeGreen : _errorRed,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Container(
                      height: max(4, h.toDouble()),
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

  Widget _buildInfoCard(
    int testsCount,
    String confidence,
    Map<String, dynamic> modelInfo,
  ) {
    final Color confColor = _confidenceColor(confidence);
    final String modelVersion =
        modelInfo['model_version']?.toString() ?? 'student-success-v2.3';
    final String historyUsed = modelInfo['history_used']?.toString() ?? '$testsCount';
    final String modelStatus = modelInfo['model_status']?.toString() ?? 'unknown';

    final String confDescription =
        {
          'high': "Based on 6+ saved tests. Stronger history signal.",
          'medium': "Based on 3-5 saved tests. Moderate confidence.",
          'low': "Based on 1-2 saved tests. Limited confidence.",
          'no_model': "The student prediction model files were not loaded.",
          'no_data': "No saved tests were found for this student.",
        }[confidence] ??
        "Limited prediction confidence.";

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
              Expanded(
                child: Text(
                  _confidenceLabel(confidence),
                  style: GoogleFonts.lexend(
                    color: confColor,
                    fontSize: 13,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
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
          _buildInfoRow("Model", modelVersion),
          _buildInfoRow("Status", modelStatus),
          _buildInfoRow("History used", historyUsed),
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
            "The model analyzes the student’s saved test history with a BiLSTM sequence model. It uses repeated mistakes, positive actions, recency, and trend features to estimate readiness and rank likely risk areas.",
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

  Widget _buildInfoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 5),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 92,
            child: Text(
              label,
              style: GoogleFonts.lexend(color: Colors.white38, fontSize: 10),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: GoogleFonts.lexend(color: Colors.white60, fontSize: 10),
              overflow: TextOverflow.ellipsis,
              maxLines: 2,
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
              "Run and save tests to enable readiness predictions.",
              textAlign: TextAlign.center,
              style: GoogleFonts.lexend(color: Colors.white38, fontSize: 13),
            ),
          ],
        ),
      ),
    );
  }
}
