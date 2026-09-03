import sys, time, random
sys.path.insert(0, '.')
from event_bus import EventBus

bus = EventBus()
patient_id = "PT_ICU_099"

print("🔬 M3 Lab Analyzer: Starting real-time stream...")
current_creat = 0.9

try:
    for i in range(8):
        # Simulate AKI: creatinine rises over 8 hours
        current_creat += random.uniform(0.15, 0.35) 
        
        event = {
            "patient_id": patient_id,
            "analyte": "creatinine",
            "value": round(current_creat, 2),
            "unit": "mg/dL",
            "timestamp": time.time(),
            "provenance_id": f"lab_creat_{i}"
        }
        
        print(f"  📤 Publishing: Creatinine = {event['value']} mg/dL")
        bus.publish("lab_results", event)
        time.sleep(2) # Wait 2 seconds between "hours"
except KeyboardInterrupt:
    pass