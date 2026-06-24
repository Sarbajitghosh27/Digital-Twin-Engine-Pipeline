from aiml_core.fault_classifier import FaultDiagnosis

def generate_reasoning(state: dict) -> str:
    fd = state['fault_diagnosis'] # FaultDiagnosis object
    hi = state['health_index']
    r50, r10, r90 = state['rul_p50'], state['rul_p10'], state['rul_p90']
    pa = state['pma_attributions'] # {sensor: attribution_value}
    pct = lambda v: f"{abs(v)*100:.0f}%"
    direction = lambda v: 'above' if v > 0 else 'below'
    top_s = fd.primary_sensor
    top_v = pa.get(top_s, 0.0)
    sec_s = fd.secondary_sensor
    sec_v = pa.get(sec_s, 0.0)
    hi_status = 'critical' if hi < 40 else ('degraded' if hi < 70 else 'healthy')
    return (
        f"Engine #{state['engine_id']:03d} | Cycle {state['cycle']} | "
        f"RUL: {r50} cycles ({r10}-{r90} band) | HI: {hi:.1f}% [{hi_status}]. "
        f"Primary driver: {top_s} running {pct(top_v)} {direction(top_v)} healthy baseline "
        f"(attribution {top_v:+.2f}), with {sec_s} at {sec_v:+.2f}. "
        f"Fault mode: {fd.fault_mode} – {fd.explanation}. "
        f"Confidence: {fd.confidence*100:.0f}%. "
        f"Recommended: {fd.recommended_action}."
    )
