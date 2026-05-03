import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_model.dart';
import '../services/api_service.dart';

class UserProvider with ChangeNotifier {
  User? _user;
  String? _token;
  bool _isLoading = false;

  User? get user => _user;
  String? get token => _token;
  bool get isLogged => _user != null && _token != null;
  bool get isLoading => _isLoading;

  /// 🆕 Login flow now stores JWT alongside user
  Future<bool> login(String email, String password, bool rememberMe) async {
    _isLoading = true;
    notifyListeners();

    final result = await ApiService.login(email, password);
    if (result == null) {
      _isLoading = false;
      notifyListeners();
      return false;
    }

    _token = result["access_token"];
    _user = User(email: result["email"], name: result["name"]);
    ApiService.setToken(_token);

    if (rememberMe) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('saved_user', jsonEncode(_user!.toJson()));
      await prefs.setString('saved_token', _token!);
    }

    _isLoading = false;
    notifyListeners();
    return true;
  }

  Future<bool> register(
    String name,
    String email,
    String password,
    bool rememberMe,
  ) async {
    _isLoading = true;
    notifyListeners();

    final result = await ApiService.register(name, email, password);
    if (result == null) {
      _isLoading = false;
      notifyListeners();
      return false;
    }

    _token = result["access_token"];
    _user = User(email: result["email"], name: result["name"]);
    ApiService.setToken(_token);

    if (rememberMe) {
      final prefs = await SharedPreferences.getInstance();
      await prefs.setString('saved_user', jsonEncode(_user!.toJson()));
      await prefs.setString('saved_token', _token!);
    }

    _isLoading = false;
    notifyListeners();
    return true;
  }

  /// 🆕 Restore both user JSON and JWT
  Future<void> tryAutoLogin() async {
    final prefs = await SharedPreferences.getInstance();
    final userJson = prefs.getString('saved_user');
    final token = prefs.getString('saved_token');
    if (userJson == null || token == null) return;

    _user = User.fromJson(jsonDecode(userJson));
    _token = token;
    ApiService.setToken(token);
    notifyListeners();
  }

  Future<void> logout() async {
    _user = null;
    _token = null;
    ApiService.clearToken();
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('saved_user');
    await prefs.remove('saved_token');
    notifyListeners();
  }
}
