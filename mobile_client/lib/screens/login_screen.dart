import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:google_fonts/google_fonts.dart';
import 'register_screen.dart';
import 'package:http/http.dart' as http;
import 'dart:async';
import 'package:provider/provider.dart';
import '../providers/user_provider.dart';
import 'home_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({super.key});

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passController = TextEditingController();
  final Color _primaryColor = const Color(0xFF3E7DEA);
  final Color _inputColor = const Color(0xFF2e446b);
  final Color _cardBorder = const Color(0xFF172236);
  final FocusNode _emailFocus = FocusNode();
  final FocusNode _passFocus = FocusNode();
  String? _emailError;
  String? _passError;
  bool _rememberMe = false;
  bool _hiddenpass = true;

  @override
  void initState() {
    super.initState();
    void refresh() => setState(() {});
    _emailFocus.addListener(refresh);
    _passFocus.addListener(refresh);
  }

  void _sendTORegister() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => RegisterScreen()),
    );
  }

  Future<void> loginUser() async {
    final url = Uri.parse('http://127.0.0.1:8000/api/auth/login');

    try {
      print("Attempting login for: ${emailController.text}");

      final response = await http.post(
        url,
        headers: {"Content-Type": "application/json"},
        body: jsonEncode({
          "email": emailController.text.trim(),
          "password": passController.text,
        }),
      );

      if (response.statusCode == 200) {
        print(" Login Success!");
        final responseData = jsonDecode(response.body);

        String userName = responseData['name'] ?? "Driver";
        String userEmail = responseData['email'] ?? emailController.text;

        if (!mounted) return;

        await Provider.of<UserProvider>(
          context,
          listen: false,
        ).login(userEmail, userName, _rememberMe);

        if (mounted) {
          Navigator.of(context).pushReplacement(
            MaterialPageRoute(builder: (context) => const HomeScreen()),
          );
        }
      } else if (response.statusCode == 401) {
        _showError("Invalid email or password");
      } else {
        print("Server Error: ${response.body}");
        _showError("thats not a vaild email");
      }
    } catch (e) {
      print(" CRITICAL ERROR: $e");
      _showError("Can't connect to server. Check your Wi-Fi.");
    }
  }

  void _showError(String message) {
    setState(() {
      _passError = message;
      _emailError = message;
    });
    Future.delayed(const Duration(seconds: 4), () {
      if (mounted) {
        setState(() {
          _emailError = null;
          _passError = null;
        });
      }
    });
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
        child: Padding(
          padding: const EdgeInsets.all(20.0),

          child: Column(
            children: [
              Align(
                alignment: Alignment.topLeft,
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Image.asset('assets/images/logo.webp', height: 75),
                    const SizedBox(height: 10),

                    Text(
                      "LOGIN PAGE",
                      style: GoogleFonts.lexend(
                        fontSize: 40,
                        letterSpacing: 2.0,
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        shadows: [
                          Shadow(
                            color: _primaryColor,
                            blurRadius: 10,
                            offset: Offset.zero,
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),

              Expanded(
                child: Center(
                  child: SingleChildScrollView(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        _buildFields(
                          controller: emailController,
                          focusNode: _emailFocus,
                          label: "Email",
                          icon: Icons.email,
                          errorText: _emailError,
                          onChanged: (value) {
                            setState(() {
                              if (value.trim().isEmpty) {
                                _emailError = "Cant leave an empty field!";
                              } else {
                                _emailError = null;
                              }
                            });
                          },
                        ),

                        const SizedBox(height: 50),

                        _buildFields(
                          controller: passController,
                          focusNode: _passFocus,
                          label: "Password",
                          icon: Icons.lock,
                          isPassword: true,
                          isObscured: _hiddenpass,
                          onEyeToggle: () {
                            setState(() {
                              _hiddenpass = !_hiddenpass;
                            });
                          },
                          errorText: _passError,

                          onChanged: (value) {
                            setState(() {
                              if (value.trim().isEmpty) {
                                _passError = "Cant leave an empty field!";
                              } else {
                                _passError = null;
                              }
                            });
                          },
                        ),
                        const SizedBox(height: 70),
                        Padding(
                          padding: const EdgeInsets.only(bottom: 20.0),
                          child: Row(
                            children: [
                              GestureDetector(
                                onTap: () {
                                  setState(() {
                                    _rememberMe = !_rememberMe;
                                  });
                                },
                                child: AnimatedContainer(
                                  duration: const Duration(milliseconds: 200),
                                  width: 24,
                                  height: 24,
                                  decoration: BoxDecoration(
                                    borderRadius: BorderRadius.circular(6),
                                    border: Border.all(
                                      color: _primaryColor,
                                      width: 2,
                                    ),
                                    color: _rememberMe
                                        ? _primaryColor
                                        : Colors.transparent,
                                    boxShadow: _rememberMe
                                        ? [
                                            BoxShadow(
                                              color: _primaryColor.withOpacity(
                                                0.6,
                                              ),
                                              blurRadius: 10,
                                              offset: Offset.zero,
                                            ),
                                          ]
                                        : [],
                                  ),
                                  child: _rememberMe
                                      ? const Icon(
                                          Icons.check,
                                          size: 16,
                                          color: Colors.white,
                                        )
                                      : null,
                                ),
                              ),
                              const SizedBox(width: 10),
                              Text(
                                "Remember me",
                                style: GoogleFonts.poppins(
                                  color: Colors.white,
                                  fontSize: 15,
                                  fontWeight: FontWeight.bold,
                                  shadows: [
                                    Shadow(
                                      color: _primaryColor,
                                      blurRadius: 10,
                                      offset: Offset.zero,
                                    ),
                                  ],
                                ),
                              ),
                            ],
                          ),
                        ),

                        Container(
                          width: 300,
                          height: 55,
                          decoration: BoxDecoration(
                            borderRadius: BorderRadius.circular(30),
                            boxShadow: [
                              BoxShadow(
                                color: const Color(0xFF4C9EEB).withOpacity(0.3),
                                blurRadius: 10,
                                offset: const Offset(0, 5),
                              ),
                            ],
                          ),
                          child: ElevatedButton(
                            style: ElevatedButton.styleFrom(
                              backgroundColor: _primaryColor,
                              foregroundColor: Colors.white,
                              shape: RoundedRectangleBorder(
                                borderRadius: BorderRadius.circular(30),
                              ),
                              elevation: 0,
                            ),
                            onPressed: () {
                              if (_emailError == null &&
                                  _passError == null &&
                                  emailController.text.isNotEmpty &&
                                  passController.text.isNotEmpty) {
                                loginUser();
                                print("pressed");
                              }
                            },
                            child: Text(
                              "LOG IN",
                              style: GoogleFonts.rubik(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                letterSpacing: 1.5,
                              ),
                            ),
                          ),
                        ),

                        const SizedBox(height: 30),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              "Dont have an account?",
                              style: GoogleFonts.poppins(
                                color: Colors.white,
                                fontSize: 15,
                                fontWeight: FontWeight.bold,
                                shadows: [
                                  Shadow(
                                    color: _primaryColor,
                                    blurRadius: 10,
                                    offset: Offset.zero,
                                  ),
                                ],
                              ),
                            ),
                            TextButton(
                              onPressed: _sendTORegister,
                              child: Text(
                                "Register here",
                                style: GoogleFonts.poppins(
                                  color: _primaryColor,
                                  fontSize: 15,
                                  fontWeight: FontWeight.bold,
                                  shadows: [
                                    Shadow(
                                      color: _primaryColor,
                                      blurRadius: 10,
                                      offset: Offset.zero,
                                    ),
                                  ],
                                ),
                              ),
                            ),
                          ],
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildFields({
    required TextEditingController controller,
    required FocusNode focusNode,
    required String label,
    IconData? icon,
    String? errorText,
    Function(String)? onChanged,

    bool isPassword = false,
    bool isObscured = false,
    VoidCallback? onEyeToggle,
  }) {
    return TextField(
      controller: controller,
      focusNode: focusNode,
      obscureText: isPassword ? isObscured : false,
      style: const TextStyle(color: Colors.white),
      cursorColor: _primaryColor,
      onChanged: onChanged,
      decoration: InputDecoration(
        labelText: label,
        labelStyle: GoogleFonts.poppins(
          color: Colors.white70,
          letterSpacing: 1.0,
        ),
        floatingLabelStyle: GoogleFonts.poppins(
          color: errorText != null ? Colors.redAccent : _primaryColor,

          fontWeight: FontWeight.bold,
        ),
        errorText: errorText,
        filled: true,
        fillColor: focusNode.hasFocus
            ? _inputColor.withOpacity(0.8)
            : _inputColor,

        prefixIcon: (icon != null ? Icon(icon, color: _primaryColor) : null),
        suffixIcon: isPassword
            ? IconButton(
                icon: Icon(
                  isObscured ? Icons.visibility_off : Icons.visibility,
                  color: Colors.white70,
                ),
                onPressed: onEyeToggle,
              )
            : null,

        contentPadding: const EdgeInsets.fromLTRB(12, 16, 12, 16),

        errorStyle: const TextStyle(
          color: Colors.redAccent,
          fontSize: 14,
          fontWeight: FontWeight.bold,
        ),

        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: _cardBorder, width: 2),
        ),
        focusedBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: BorderSide(color: _primaryColor, width: 3),
        ),
        errorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.redAccent),
        ),
        focusedErrorBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(12),
          borderSide: const BorderSide(color: Colors.redAccent, width: 3),
        ),
      ),
    );
  }
}
