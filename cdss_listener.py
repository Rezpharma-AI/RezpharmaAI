import sys, json, time, queue
sys.path.insert(0, '.')
from event_bus import EventBus
from src.m3_analysis_lab.kinetic_gfr import KineticGFR
from src.blackboard.blackboard import Blackboard

bus = EventBus()
bb = Blackboard(patient_id="PT_ICU_099")
kgfr = KineticGFR(n_particles=500)

print("🎧 CDSS Blackboard: Listening for lab events...")
print("   (Press Ctrl+C to stop)\n")

creatinine_history = []

def process_message(msg):
    data = json.loads(msg)
    analyte = data.get("analyte")
    value = data.get("value")
    prov_id = data.get("provenance_id")
    
    print(f"  📥 Received: {analyte} = {value}")
    
    if analyte == "creatinine":
        creatinine_history.append((len(creatinine_history), value))
        
        if len(creatinine_history) >= 2:
            gfr_post = kgfr.estimate(creatinine_history, muscle_mass=1.1)
            gfr_normal = gfr_post.to_normal_approximation()
            
            print(f"     ⚙️  Kinetic GFR updated: {gfr_normal.mu:.1f} mL/min")
            
            try:
                bb.update_latent("GFR", gfr_normal, prov_id, "M3")
            except Exception:
                pass
                
            # Trigger Alert if GFR drops below 60 (AKI)
            if gfr_normal.mu < 60:
                alert_event = {
                    "patient_id": data["patient_id"],
                    "module": "M3",
                    "harm": "acute_kidney_injury",
                    "message": f"Kinetic GFR dropped to {gfr_normal.mu:.0f} mL/min. Adjust renally cleared drugs!"
                }
                print(f"     🚨 ALERT PUBLISHED: {alert_event['message']}")
                bus.publish("cdss_alerts", alert_event)

sub = bus.subscribe("lab_results")

if bus.use_redis:
    for message in sub.listen():
        if message['type'] == 'message':
            process_message(message['data'])
else:
    try:
        while True:
            try:
                msg = sub.get(timeout=1.0)
                process_message(msg)
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        print("\nStopping listener.")