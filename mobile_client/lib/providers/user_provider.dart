import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:shared_preferences/shared_preferences.dart';
import '../models/user_model.dart';

class UserProvider with ChangeNotifier {
  User? _user;
  bool _isLoading = false;

  User? get user => _user;
  bool get isLogged => _user != null;
  bool get isLoading => _isLoading;

  Future<void> login(String email, String name, bool rememberMe) async {
    print("Provider: Login process started for $name");

    _isLoading = true;
    notifyListeners();

    _user = User(email: email, name: name);
    print("Provider: User object created in memory");

    if (rememberMe) {
      final prefs = await SharedPreferences.getInstance();
      String userJson = jsonEncode(_user!.toJson());
      await prefs.setString('saved_user', userJson);
      print("Provider: Saved to local storage");
    }

    _isLoading = false;

    notifyListeners();

    print("Provider: Listeners notified! UI should update now.");
  }

  Future<void> tryAutoLogin() async {
    final prefs = await SharedPreferences.getInstance();
    if (!prefs.containsKey('saved_user')) return;

    final String? userJson = prefs.getString('saved_user');
    if (userJson != null) {
      _user = User.fromJson(jsonDecode(userJson));
      notifyListeners();
    }
  }

  Future<void> logout() async {
    _user = null;
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('saved_user');
    notifyListeners();
  }
}
