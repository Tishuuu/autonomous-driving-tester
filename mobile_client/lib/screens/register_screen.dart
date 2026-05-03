import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart'; // 🆕 הוסף ייבוא Provider
import '../providers/user_provider.dart'; // 🆕 הוסף ייבוא UserProvider
import '../services/api_config.dart';
import 'login_screen.dart';

class RegisterScreen extends StatefulWidget {
  const RegisterScreen({super.key});

  @override
  State<RegisterScreen> createState() => _RegisterScreenState();
}

class _RegisterScreenState extends State<RegisterScreen> {
  final TextEditingController nameController = TextEditingController();
  final TextEditingController emailController = TextEditingController();
  final TextEditingController passController = TextEditingController();
  final TextEditingController confirmPassController = TextEditingController();
  final Color _primaryColor = const Color(0xFF3E7DEA);
  final Color _inputColor = const Color(0xFF2e446b);
  final Color _cardBorder = const Color(0xFF172236);
  final FocusNode _nameFocus = FocusNode();
  final FocusNode _emailFocus = FocusNode();
  final FocusNode _passFocus = FocusNode();
  final FocusNode _confirmPassFocus = FocusNode();
  String? _nameError;
  String? _emailError;
  String? _passError;
  String? _confirmPassError;

  bool _hiddenpass = true;
  @override
  void initState() {
    super.initState();
    void refresh() => setState(() {});

    _nameFocus.addListener(refresh);
    _emailFocus.addListener(refresh);
    _passFocus.addListener(refresh);
    _confirmPassFocus.addListener(refresh);
  }

  void _sendTOlogin() {
    Navigator.push(
      context,
      MaterialPageRoute(builder: (context) => const LoginScreen()),
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
    bool isConfirm = false,
    bool isMatched = false,
  }) {
    return TextField(
      controller: controller,
      focusNode: focusNode,
      obscureText: isPassword || isConfirm ? isObscured : false,
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

        prefixIcon: isConfirm
            ? Icon(
                isMatched ? Icons.check_circle : Icons.cancel,
                color: isMatched ? _primaryColor : Colors.redAccent,
              )
            : (icon != null ? Icon(icon, color: _primaryColor) : null),
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

  // הפונקציה הישנה הוסרה - הכל מטופל דרך הפרובידר

  @override
  Widget build(BuildContext context) {
    // השגת הפרובידר כדי להראות חיווי טעינה
    final userProvider = Provider.of<UserProvider>(context);

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
                      "REGISTER PAGE",
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
                          controller: nameController,
                          focusNode: _nameFocus,
                          label: "Full Name",
                          icon: Icons.person,
                          errorText: _nameError,
                          onChanged: (value) {
                            setState(() {
                              if (value.trim().length < 3) {
                                _nameError = " name is too short!";
                              } else if ((RegExp(
                                r'[!@#<>?":_`~;[\]\\|=+)(*&^%0-9-]',
                              ).hasMatch(value))) {
                                _nameError =
                                    "numbers and special characters are not allowed!";
                              } else {
                                _nameError = null;
                              }
                            });
                          },
                        ),
                        const SizedBox(height: 20),

                        _buildFields(
                          controller: emailController,
                          focusNode: _emailFocus,
                          label: "Email",
                          icon: Icons.email,
                          errorText: _emailError,
                          onChanged: (value) {
                            setState(() {
                              if (!RegExp(
                                r'^[\w-\.]+@([\w-]+\.)+[\w-]{2,4}$',
                              ).hasMatch(value)) {
                                _emailError = " thats not a vaild email!";
                              } else {
                                _emailError = null;
                              }
                            });
                          },
                        ),

                        const SizedBox(height: 20),

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
                              if (value.trim().length < 6) {
                                _passError = " pass is too short!";
                              } else if (!(RegExp('[0-9]').hasMatch(value)) ||
                                  !(RegExp('[a-z]').hasMatch(value))) {
                                _passError =
                                    "the password must contain a number or a letter!";
                              } else {
                                _passError = null;
                              }
                            });
                          },
                        ),
                        const SizedBox(height: 20),

                        _buildFields(
                          controller: confirmPassController,
                          focusNode: _confirmPassFocus,
                          label: "Confirm Password",
                          icon: Icons.verified,
                          isConfirm: true,
                          isMatched:
                              confirmPassController.text == passController.text,
                          isObscured: true,
                          errorText: _confirmPassError,
                          onChanged: (value) {
                            setState(() {
                              if (!(value == passController.text)) {
                                _confirmPassError =
                                    "the passwords doesnt match!";
                              } else {
                                _confirmPassError = null;
                              }
                            });
                          },
                        ),

                        const SizedBox(height: 30),
                        Container(
                          width: 500,
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
                            // ✅ פונקציית הלחיצה המעודכנת
                            onPressed: userProvider.isLoading
                                ? null
                                : () async {
                                    if (_nameError == null &&
                                        _emailError == null &&
                                        _passError == null &&
                                        _confirmPassError == null &&
                                        nameController.text.isNotEmpty &&
                                        emailController.text.isNotEmpty &&
                                        passController.text.isNotEmpty) {
                                      bool success =
                                          await Provider.of<UserProvider>(
                                            context,
                                            listen: false,
                                          ).register(
                                            nameController.text.trim(),
                                            emailController.text.trim(),
                                            passController.text,
                                            false, // מסך רישום לא צריך remember me כברירת מחדל
                                          );

                                      if (mounted) {
                                        if (success) {
                                          ScaffoldMessenger.of(
                                            context,
                                          ).showSnackBar(
                                            const SnackBar(
                                              content: Text(
                                                "Account created! Please log in.",
                                              ),
                                              backgroundColor: Colors.green,
                                            ),
                                          );
                                          Future.delayed(
                                            const Duration(seconds: 2),
                                            () {
                                              if (mounted) _sendTOlogin();
                                            },
                                          );
                                        } else {
                                          ScaffoldMessenger.of(
                                            context,
                                          ).showSnackBar(
                                            const SnackBar(
                                              content: Text(
                                                "Registration failed. Please try again.",
                                              ),
                                              backgroundColor: Colors.red,
                                            ),
                                          );
                                        }
                                      }
                                    }
                                  },
                            child: userProvider.isLoading
                                ? const SizedBox(
                                    height: 24,
                                    width: 24,
                                    child: CircularProgressIndicator(
                                      color: Colors.white,
                                      strokeWidth: 2.5,
                                    ),
                                  )
                                : Text(
                                    "CREATE ACCOUNT",
                                    style: GoogleFonts.rubik(
                                      fontSize: 18,
                                      fontWeight: FontWeight.bold,
                                      letterSpacing: 1.5,
                                    ),
                                  ),
                          ),
                        ),

                        const SizedBox(height: 15),

                        Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              "Already have an account?",
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
                              onPressed: _sendTOlogin,
                              child: Text(
                                "Log in",
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

  @override
  void dispose() {
    nameController.dispose();
    emailController.dispose();
    passController.dispose();
    confirmPassController.dispose();

    _nameFocus.dispose();
    _emailFocus.dispose();
    _passFocus.dispose();
    _confirmPassFocus.dispose();

    super.dispose();
  }
}
