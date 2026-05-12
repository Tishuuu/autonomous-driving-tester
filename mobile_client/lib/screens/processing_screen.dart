import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_fonts/google_fonts.dart';

import '../services/api_service.dart';
import 'test_summary_screen.dart';

class ProcessingScreen extends StatefulWidget {
  final String videoPath;
  final String jsonPath;
  final String studentId;

  const ProcessingScreen({
    super.key,
    required this.videoPath,
    required this.jsonPath,
    this.studentId = "pending",
  });

  @override
  State<ProcessingScreen> createState() => _ProcessingScreenState();
}

class _ProcessingScreenState extends State<ProcessingScreen>
    with TickerProviderStateMixin {
  static const Color _primaryBlue = Color(0xFF3E7DEA);
  static const Color _activeGreen = Color(0xFF00FF94);
  static const Color _errorRed = Color(0xFFFF4C4C);

  late final AnimationController _pulseController;

  Timer? _progressTimer;
  Timer? _elapsedTimer;

  bool _hasError = false;
  String _errorMessage = "";
  int _elapsedSeconds = 0;

  int _currentStage = 0;
  int _percent = 0;
  String _stageMessage = "Preparing upload...";

  late final String _testId;

  final List<IconData> _stageIcons = const [
    Icons.upload_rounded,
    Icons.sensors_rounded,
    Icons.visibility_rounded,
    Icons.layers_rounded,
    Icons.psychology_rounded,
  ];

  final List<String> _stageTitles = const [
    "Uploading Files",
    "Processing Sensors",
    "Running YOLO Detection",
    "Building Feature Vector",
    "Evaluating with M11",
  ];

  @override
  void initState() {
    super.initState();

    SystemChrome.setPreferredOrientations([
      DeviceOrientation.portraitUp,
      DeviceOrientation.portraitDown,
    ]);

    _testId = "TEST_${DateTime.now().millisecondsSinceEpoch}";

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1500),
    )..repeat(reverse: true);

    _startElapsedTimer();
    _startProgressPolling();
    _startAnalysis();
  }

  void _startElapsedTimer() {
    _elapsedTimer = Timer.periodic(const Duration(seconds: 1), (_) {
      if (!mounted || _hasError) return;
      setState(() => _elapsedSeconds++);
    });
  }

  void _startProgressPolling() {
    _progressTimer = Timer.periodic(const Duration(seconds: 1), (_) async {
      if (!mounted || _hasError) return;

      final progress = await ApiService.getProgress(_testId);

      if (progress == null || !mounted) return;

      final rawPercent = progress['percent'];
      final parsedPercent = rawPercent is int
          ? rawPercent
          : rawPercent is double
              ? rawPercent.round()
              : int.tryParse(rawPercent?.toString() ?? "0") ?? 0;

      setState(() {
        _percent = parsedPercent.clamp(0, 100);
        _stageMessage =
            progress['message']?.toString() ?? "Processing analysis...";

        if (_percent < 5) {
          _currentStage = 1;
        } else if (_percent < 10) {
          _currentStage = 2;
        } else if (_percent < 80) {
          _currentStage = 3;
        } else if (_percent < 85) {
          _currentStage = 4;
        } else {
          _currentStage = 5;
        }
      });
    });
  }

  Future<void> _startAnalysis() async {
    try {
      final result = await ApiService.uploadTestFiles(
        widget.videoPath,
        widget.jsonPath,
        widget.studentId,
        testId: _testId,
      );

      if (!mounted) return;

      _progressTimer?.cancel();
      _elapsedTimer?.cancel();

      if (result != null) {
        Navigator.pushReplacement(
          context,
          MaterialPageRoute(
            builder: (_) => TestSummaryScreen(result: result),
          ),
        );
      } else {
        setState(() {
          _hasError = true;
          _errorMessage =
              "We couldn't reach the analysis server. Please check your connection and try again.";
        });
      }
    } catch (e) {
      if (!mounted) return;

      _progressTimer?.cancel();
      _elapsedTimer?.cancel();

      setState(() {
        _hasError = true;
        _errorMessage = "Analysis failed: $e";
      });
    }
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _progressTimer?.cancel();
    _elapsedTimer?.cancel();
    super.dispose();
  }

  String _formatElapsed(int seconds) {
    final minutes = (seconds ~/ 60).toString().padLeft(2, '0');
    final remainingSeconds = (seconds % 60).toString().padLeft(2, '0');
    return "$minutes:$remainingSeconds";
  }

  IconData _currentIcon() {
    if (_currentStage <= 0) return Icons.upload_rounded;

    final index = (_currentStage - 1).clamp(0, _stageIcons.length - 1);
    return _stageIcons[index];
  }

  String _currentTitle() {
    if (_currentStage <= 0) return "Starting...";

    final index = (_currentStage - 1).clamp(0, _stageTitles.length - 1);
    return _stageTitles[index];
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              Color(0xFF314972),
              Color(0xFF233452),
            ],
          ),
        ),
        child: SafeArea(
          child: Column(
            children: [
              const SizedBox(height: 30),

              if (!_hasError)
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    _buildBadge(
                      Icons.access_time,
                      _formatElapsed(_elapsedSeconds),
                    ),
                    const SizedBox(width: 8),
                    _buildBadge(Icons.bolt, "$_percent%"),
                  ],
                ),

              const Spacer(),

              ScaleTransition(
                scale: Tween<double>(
                  begin: 0.92,
                  end: 1.08,
                ).animate(_pulseController),
                child: Container(
                  padding: const EdgeInsets.all(35),
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: _hasError
                        ? _errorRed.withOpacity(0.1)
                        : _primaryBlue.withOpacity(0.1),
                    border: Border.all(
                      color: _hasError ? _errorRed : _primaryBlue,
                      width: 2,
                    ),
                    boxShadow: [
                      BoxShadow(
                        color: (_hasError ? _errorRed : _primaryBlue)
                            .withOpacity(0.3),
                        blurRadius: 30,
                        spreadRadius: 4,
                      ),
                    ],
                  ),
                  child: Icon(
                    _hasError ? Icons.error_outline : _currentIcon(),
                    color: _hasError ? _errorRed : Colors.white,
                    size: 70,
                  ),
                ),
              ),

              const SizedBox(height: 36),

              Text(
                _hasError ? "Analysis Failed" : _currentTitle(),
                textAlign: TextAlign.center,
                style: GoogleFonts.lexend(
                  color: Colors.white,
                  fontSize: 24,
                  fontWeight: FontWeight.bold,
                ),
              ),

              const SizedBox(height: 10),

              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 40),
                child: Text(
                  _hasError ? _errorMessage : _stageMessage,
                  textAlign: TextAlign.center,
                  style: GoogleFonts.lexend(
                    color: Colors.white70,
                    fontSize: 14,
                    height: 1.4,
                  ),
                ),
              ),

              const SizedBox(height: 32),

              if (!_hasError)
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  child: Column(
                    children: [
                      ClipRRect(
                        borderRadius: BorderRadius.circular(10),
                        child: LinearProgressIndicator(
                          value: _percent / 100.0,
                          minHeight: 8,
                          backgroundColor: Colors.white12,
                          valueColor: const AlwaysStoppedAnimation<Color>(
                            _primaryBlue,
                          ),
                        ),
                      ),
                      const SizedBox(height: 14),
                      Row(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: List.generate(5, (index) {
                          final stageNumber = index + 1;
                          final isActive = stageNumber == _currentStage;
                          final isDone = stageNumber < _currentStage;

                          return AnimatedContainer(
                            duration: const Duration(milliseconds: 300),
                            margin: const EdgeInsets.symmetric(horizontal: 4),
                            width: isActive ? 24 : 8,
                            height: 8,
                            decoration: BoxDecoration(
                              color: isDone
                                  ? _activeGreen
                                  : isActive
                                      ? _primaryBlue
                                      : Colors.white24,
                              borderRadius: BorderRadius.circular(4),
                            ),
                          );
                        }),
                      ),
                    ],
                  ),
                ),

              const Spacer(),

              if (_hasError) ...[
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 40),
                  child: SizedBox(
                    width: double.infinity,
                    child: ElevatedButton(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: _primaryBlue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(
                          borderRadius: BorderRadius.circular(15),
                        ),
                      ),
                      onPressed: () => Navigator.pop(context),
                      child: Text(
                        "Go Back",
                        style: GoogleFonts.lexend(
                          fontSize: 16,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 30),
              ] else
                const SizedBox(height: 50),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBadge(IconData icon, String text) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
      decoration: BoxDecoration(
        color: Colors.white.withOpacity(0.06),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: Colors.white12),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: Colors.white54,
            size: 14,
          ),
          const SizedBox(width: 6),
          Text(
            text,
            style: GoogleFonts.lexend(
              color: Colors.white70,
              fontSize: 12,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }
}
