import obd
import time

print("🔍 Searching for OBD-II adapter...")

# הספריה יודעת לחפש את הפורט (COM) של הבלוטות' באופן אוטומטי
connection = obd.OBD()

if connection.is_connected():
    print("✅ Successfully connected to the car!")
    print(f"Protocol: {connection.protocol_info().name}\n")
    
    print("📊 Starting data read (Make sure the engine is running or switch is ON)...")
    
    for i in range(1, 6):
        # יצירת שאילתות למהירות ולסל"ד
        cmd_rpm = obd.commands.RPM
        cmd_speed = obd.commands.SPEED
        
        # שליחת השאילתות לרכב
        response_rpm = connection.query(cmd_rpm)
        response_speed = connection.query(cmd_speed)
        
        # חילוץ הערכים המספריים (אם אין נתון, יציג None)
        rpm_val = response_rpm.value.magnitude if response_rpm.value else 0
        speed_val = response_speed.value.magnitude if response_speed.value else 0
        
        print(f"[{i}/5] RPM: {rpm_val} | Speed: {speed_val} km/h")
        
        time.sleep(1) # המתנה של שנייה בין קריאות
        
    connection.close()
    print("\n👋 Disconnected.")
else:
    print("❌ Failed to connect.")
    print("Tip: Check if the iCar Pro is paired to Windows Bluetooth, and that the car switch is ON.")