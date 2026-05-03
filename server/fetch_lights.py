import requests
import json
import os

def fetch_traffic_lights(city_name_hebrew="נס ציונה"):
    print(f"🌍 מתחבר ל-OpenStreetMap כדי למשוך רמזורים ב{city_name_hebrew}...")
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    # שאילתה שמחפשת את גבולות העיר ואז את כל הרמזורים בתוכה
    overpass_query = f"""
    [out:json];
    area["name"="{city_name_hebrew}"]->.searchArea;
    node["highway"="traffic_signals"](area.searchArea);
    out center;
    """
    
    # הפתרון ל-406: אנחנו אומרים לשרת מי אנחנו כדי שלא יחסום אותנו
    headers = {
        "User-Agent": "AutoTesterProject/1.0 (Student Research - jonathan)"
    }
    
    try:
        # שינינו ל-POST וצירפנו את ה-Headers!
        response = requests.post(overpass_url, data={'data': overpass_query}, headers=headers)
        
        # בודק אם השרת עדיין כועס
        response.raise_for_status() 
        data = response.json()
        
        lights = []
        # בודק קודם כל אם יש בכלל elements בתשובה
        if 'elements' in data:
            for element in data['elements']:
                if element['type'] == 'node':
                    lights.append({
                        "lat": element['lat'], 
                        "lon": element['lon']
                    })
        
        # שמירת התוצאות לתיקיית ai_models 
        save_path = os.path.join("app", "ai_models", "ness_ziona_lights.json")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(lights, f, indent=4)
            
        print(f"✅ נשאבו בהצלחה {len(lights)} רמזורים ב{city_name_hebrew}!")
        print(f"📁 הקובץ נשמר בנתיב: {save_path}")
        
    except Exception as e:
        print(f"❌ שגיאה בשאיבת הנתונים: {e}")
        # במקרה של שגיאה נוספת נדפיס גם את הטקסט מהשרת כדי להבין למה
        if hasattr(e, 'response') and e.response is not None:
            print(f"תשובת השרת: {e.response.text}")

if __name__ == "__main__":
    fetch_traffic_lights()