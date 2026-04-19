import 'package:flutter/material.dart';
import '../services/api_service.dart';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import 'login_screen.dart';

class StatsScreen extends StatefulWidget {
  const StatsScreen({super.key});

  @override
  State<StatsScreen> createState() => _StatsScreenState();
}

class _StatsScreenState extends State<StatsScreen> {
  // משתנה שישמור את רשימת הטסטים שנמשוך מהשרת
  late Future<List<dynamic>> _historyFuture;
  final String studentId = "123456789"; // תעודת הזהות של התלמיד לבדיקה

  @override
  void initState() {
    super.initState();
    // ברגע שהמסך עולה, אנחנו מבקשים מהשרת את ההיסטוריה
    _historyFuture = ApiService.getStudentHistory(studentId);
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
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Image.asset('assets/images/logo.webp', height: 60),
                    IconButton(
                      icon: const Icon(Icons.logout, color: Colors.redAccent),
                      onPressed: () async {
                        await Provider.of<UserProvider>(
                          context,
                          listen: false,
                        ).logout();
                        if (context.mounted) {
                          Navigator.of(context).pushAndRemoveUntil(
                            MaterialPageRoute(
                              builder: (context) => const LoginScreen(),
                            ),
                            (route) => false,
                          );
                        }
                      },
                    ),
                  ],
                ),
                const SizedBox(height: 20),
                const Text(
                  "Test History",
                  style: TextStyle(
                    fontSize: 28,
                    fontWeight: FontWeight.bold,
                    color: Colors.white,
                  ),
                ),
                const SizedBox(height: 20),

                // ה-FutureBuilder מטפל בטעינה של הנתונים מהשרת
                Expanded(
                  child: FutureBuilder<List<dynamic>>(
                    future: _historyFuture,
                    builder: (context, snapshot) {
                      if (snapshot.connectionState == ConnectionState.waiting) {
                        return const Center(
                          child: CircularProgressIndicator(color: Colors.white),
                        );
                      } else if (snapshot.hasError) {
                        return Center(
                          child: Text(
                            "Error loading history: ${snapshot.error}",
                            style: const TextStyle(color: Colors.red),
                          ),
                        );
                      } else if (!snapshot.hasData || snapshot.data!.isEmpty) {
                        return const Center(
                          child: Text(
                            "No tests found.",
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 18,
                            ),
                          ),
                        );
                      }

                      // ברגע שיש נתונים, נציג אותם ברשימה
                      List<dynamic> tests = snapshot.data!;
                      return ListView.builder(
                        itemCount: tests.length,
                        itemBuilder: (context, index) {
                          var test = tests[index];
                          bool passed = test['status'] == 'passed';

                          return Card(
                            color: Colors.white.withOpacity(0.1),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(15),
                            ),
                            child: ListTile(
                              leading: Icon(
                                passed ? Icons.check_circle : Icons.cancel,
                                color: passed
                                    ? Colors.greenAccent
                                    : Colors.redAccent,
                                size: 40,
                              ),
                              title: Text(
                                "Score: ${test['final_score']}",
                                style: const TextStyle(
                                  color: Colors.white,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                              subtitle: Text(
                                "Date: ${DateTime.parse(test['start_time']).toString().split('.')[0]}",
                                style: const TextStyle(color: Colors.white70),
                              ),
                              trailing: const Icon(
                                Icons.arrow_forward_ios,
                                color: Colors.white54,
                              ),
                            ),
                          );
                        },
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
}
