from dataclasses import dataclass
from typing import Dict, Tuple

# Fault mode table keyed by frozenset of top-2 anomaly-driving sensors.
# Covers all realistic sensor pair combinations from both HPC Degradation and Fan Fault modes
# as produced by the CMAPSS LSTM-AE + Isolation Forest pipeline.
#
# HPC Degradation signature — rising sensors:  T30, T50, htBleed, HPT_coolant
#                           — falling sensors: P30, Ps30, phi, NRc
# Fan Fault signature       — rising sensors:  T24, T50, LPT_coolant
#                           — falling sensors: Nf, NRf, BPR
FAULT_MODE_TABLE: Dict[frozenset, Tuple[str, str, str]] = {

    # ── HPC Degradation combos ──────────────────────────────────────────────
    frozenset(['T30', 'Ps30']):         ('HPC Degradation',         'Compressor fouling / tip clearance increase',        'Borescope HPC stages 3-5'),
    frozenset(['T30', 'P30']):          ('HPC Degradation',         'HPC pressure-temperature imbalance',                 'Borescope HPC + bleed valve audit'),
    frozenset(['T30', 'phi']):          ('HPC Degradation',         'Compressor delivery temperature vs fuel-flow anomaly','HPC borescope + FADEC trim check'),
    frozenset(['T30', 'NRc']):          ('HPC Degradation',         'HPC corrected speed coupling anomaly',               'Surge margin check + HPC stage inspect'),
    frozenset(['T30', 'htBleed']):      ('HPC Degradation',         'HPC bleed-off valve degradation',                   'Bleed valve inspect + HPC borescope'),
    frozenset(['T30', 'HPT_coolant']):  ('HPT Coolant Loss',        'HPT cooling flow restriction',                       'HPT borescope + coolant manifold inspect'),
    frozenset(['T30', 'T50']):          ('HPC-HPT Thermal Fault',   'Cascading thermal rise across compressor-turbine',   'Full gas-path borescope + thermal survey'),

    frozenset(['htBleed', 'Ps30']):     ('HPC Degradation',         'Bleed extraction pressure anomaly',                  'Borescope HPC stages 3-5 + bleed manifold check'),
    frozenset(['htBleed', 'P30']):      ('HPC Degradation',         'HPC delivery pressure drop with bleed rise',         'Bleed valve inspect + compressor wash'),
    frozenset(['htBleed', 'phi']):      ('HPC Degradation',         'Bleed-fuel coupling degradation',                    'FADEC fuel trim + bleed valve calibration'),
    frozenset(['htBleed', 'NRc']):      ('HPC Degradation',         'Corrected speed anomaly with bleed rise',            'Surge margin check + bleed valve inspect'),
    frozenset(['htBleed', 'HPT_coolant']): ('HPT Coolant Loss',     'HPT coolant pressure loss with bleed anomaly',       'HPT borescope + coolant manifold + bleed inspect'),
    frozenset(['htBleed', 'T50']):      ('HPC-LPT Thermal Fault',   'Bleed-off heat contaminating LPT stage temperatures','HPT/LPT stage inspect + bleed flow check'),

    frozenset(['HPT_coolant', 'Ps30']): ('HPT Coolant Loss',        'HPT cooling pressure drop',                          'HPT coolant manifold inspect + borescope'),
    frozenset(['HPT_coolant', 'P30']):  ('HPT Coolant Loss',        'HPT delivery pressure coupled with coolant loss',    'HPT borescope + P30 port inspect'),
    frozenset(['HPT_coolant', 'phi']):  ('HPT Coolant Loss',        'HPT coolant loss with fuel-air ratio shift',         'HPT borescope + FADEC fuel audit'),
    frozenset(['HPT_coolant', 'NRc']):  ('HPT Coolant Loss',        'HPT coolant degradation with corrected speed drop',  'HPT borescope + compressor surge margin check'),
    frozenset(['HPT_coolant', 'T50']):  ('HPT Coolant Loss',        'HPT coolant loss cascading to LPT thermal rise',     'HPT + LPT borescope + coolant manifold inspect'),

    frozenset(['T50', 'Ps30']):         ('LPT Thermal Wear',        'LPT stage thermal rise with HPC pressure drop',      'LPT borescope + HPC stages 3-5 inspect'),
    frozenset(['T50', 'P30']):          ('LPT Thermal Wear',        'Stage 4-5 hotspot with delivery pressure anomaly',   'LPT borescope + P30 port inspect'),
    frozenset(['T50', 'phi']):          ('LPT Thermal Wear',        'LPT thermal rise with fuel-metering deviation',      'LPT borescope + FADEC fuel trim calibration'),
    frozenset(['T50', 'NRc']):          ('LPT Thermal Wear',        'LPT thermal wear with compressor speed coupling',    'LPT borescope + surge margin check'),
    frozenset(['T50', 'LPT_coolant']):  ('LPT Thermal Wear',        'Stage 4-5 hotspot, coolant degradation',             'Coolant system pressure check'),

    frozenset(['Ps30', 'phi']):         ('HPC Degradation',         'Compressor pressure loss with fuel-metering shift',  'HPC wash + FADEC trim check'),
    frozenset(['Ps30', 'P30']):         ('HPC Degradation',         'HPC static pressure anomaly across delivery ports',  'Borescope HPC stages 3-5 + port inspect'),
    frozenset(['Ps30', 'NRc']):         ('HPC Degradation',         'HPC pressure drop with corrected speed anomaly',     'Surge margin check + HPC borescope'),
    frozenset(['P30', 'phi']):          ('Fuel Metering Fault',      'Fuel-air ratio deviation vs compressor delivery',    'FADEC fuel trim calibration'),
    frozenset(['P30', 'NRc']):          ('HPC Degradation',         'HPC delivery pressure drop with speed coupling',     'Bleed valve audit + HPC borescope'),
    frozenset(['phi', 'NRc']):          ('Fuel Metering Fault',      'Fuel-air ratio and corrected speed anomaly',         'FADEC fuel trim + surge margin check'),

    # ── Fan Fault combos ────────────────────────────────────────────────────
    frozenset(['Nf', 'BPR']):           ('Fan Mechanical Fault',     'Fan blade erosion / imbalance',                      'Vibration survey + fan balance check'),
    frozenset(['Nf', 'NRf']):           ('Fan Mechanical Fault',     'Fan speed-corrected speed mismatch',                 'Vibration survey + fan borescope'),
    frozenset(['Nf', 'T24']):           ('Fan Mechanical Fault',     'Fan speed loss with inlet temperature rise',          'Fan blade inspect + inlet distortion check'),
    frozenset(['Nf', 'T50']):           ('Fan Mechanical Fault',     'Fan performance loss cascading to LPT heating',       'Fan borescope + LPT stage inspect'),
    frozenset(['Nf', 'LPT_coolant']):   ('Fan Mechanical Fault',     'Fan anomaly with LPT coolant degradation',            'Fan balance check + LPT coolant system inspect'),

    frozenset(['NRf', 'BPR']):          ('Fan Aerodynamic Fault',    'Fan bypass ratio and corrected speed deviation',      'Fan blade inspect + inlet guide vane check'),
    frozenset(['NRf', 'T24']):          ('Fan Aerodynamic Fault',    'Fan corrected speed loss with inlet temperature rise','Fan borescope + inlet distortion survey'),
    frozenset(['NRf', 'T50']):          ('Fan Aerodynamic Fault',    'Fan corrected speed anomaly cascading to LPT',        'Fan borescope + LPT thermal survey'),
    frozenset(['NRf', 'LPT_coolant']):  ('Fan Aerodynamic Fault',    'Fan corrected speed loss with LPT coolant drop',      'Fan blade inspect + LPT coolant manifold check'),

    frozenset(['BPR', 'T24']):          ('Fan Bypass Fault',         'Bypass ratio drop with inlet temperature rise',       'Bypass valve inspect + fan blade survey'),
    frozenset(['BPR', 'T50']):          ('Fan Bypass Fault',         'Bypass ratio drop with LPT stage thermal rise',       'Bypass valve inspect + LPT borescope'),
    frozenset(['BPR', 'LPT_coolant']):  ('Fan Bypass Fault',         'Bypass ratio drop with LPT coolant degradation',      'Bypass valve + LPT coolant system inspect'),

    frozenset(['T24', 'T50']):          ('Fan-LPT Thermal Fault',    'Inlet-to-turbine thermal cascade',                   'Full gas-path thermal survey + fan inspect'),
    frozenset(['T24', 'LPT_coolant']):  ('Fan-LPT Fault',            'Inlet temperature rise with LPT coolant drop',        'Fan blade inspect + LPT coolant manifold check'),
    frozenset(['T24', 'Nc']):           ('Fan Mechanical Fault',     'Fan-core speed mismatch with inlet temperature rise', 'Fan + core speed survey + inlet inspect'),
    frozenset(['T24', 'NRc']):          ('Fan Mechanical Fault',     'Inlet temperature with core corrected speed anomaly', 'Fan borescope + core surge margin check'),

    # ── Mixed / cross-system combos ─────────────────────────────────────────
    frozenset(['Nc', 'NRc']):           ('Core Compressor Surge',    'NRc/Nc coupling anomaly',                            'Bleed valve audit + surge margin check'),
    frozenset(['Nc', 'T30']):           ('HPC Degradation',          'Core speed drop with HPC temperature rise',          'HPC borescope + core speed survey'),
    frozenset(['Nc', 'Ps30']):          ('HPC Degradation',          'Core speed and static pressure anomaly',             'HPC borescope + bleed valve audit'),
    frozenset(['Nc', 'phi']):           ('Fuel Metering Fault',       'Core speed-fuel ratio coupling anomaly',             'FADEC trim + core speed survey'),
    frozenset(['Nc', 'htBleed']):       ('HPC Degradation',          'Core speed anomaly with bleed rise',                 'HPC borescope + bleed valve calibration'),
    frozenset(['Nc', 'HPT_coolant']):   ('HPT Coolant Loss',         'Core speed anomaly with HPT coolant loss',           'HPT borescope + core speed survey'),
    frozenset(['Nc', 'Nf']):            ('Shaft Bearing Fault',       'Fan-core spool speed decoupling',                    'Shaft bearing inspect + vibration survey'),
    frozenset(['Nc', 'BPR']):           ('Shaft Bearing Fault',       'Core speed-bypass ratio coupling anomaly',           'Shaft bearing inspect + fan blade survey'),
    frozenset(['Nc', 'T50']):           ('LPT Thermal Wear',          'Core speed drop with LPT thermal rise',              'LPT borescope + core speed survey'),

    frozenset(['LPT_coolant', 'Ps30']): ('LPT Thermal Wear',         'LPT coolant degradation with HPC pressure drop',     'LPT coolant system + HPC borescope'),
    frozenset(['LPT_coolant', 'P30']):  ('LPT Thermal Wear',         'LPT coolant loss with HPC delivery pressure drop',   'LPT coolant manifold + P30 port inspect'),
    frozenset(['LPT_coolant', 'phi']):  ('LPT Thermal Wear',         'LPT coolant degradation with fuel-metering shift',   'LPT coolant system + FADEC fuel audit'),
    frozenset(['LPT_coolant', 'NRc']): ('LPT Thermal Wear',          'LPT coolant loss with HPC corrected speed drop',     'LPT coolant + HPC surge margin check'),
    frozenset(['LPT_coolant', 'htBleed']): ('LPT Thermal Wear',      'LPT coolant degradation with HPC bleed rise',        'LPT coolant system + bleed valve inspect'),
    frozenset(['LPT_coolant', 'HPT_coolant']): ('Dual Coolant Fault','Both HPT and LPT cooling circuits degraded',         'Full coolant manifold overhaul + borescope'),
    frozenset(['LPT_coolant', 'Nc']):   ('LPT Thermal Wear',         'LPT coolant loss with core speed drop',              'LPT coolant manifold + core speed survey'),
    frozenset(['LPT_coolant', 'Nf']):   ('Fan-LPT Fault',            'Fan speed anomaly with LPT coolant degradation',     'Fan balance check + LPT coolant inspect'),

    frozenset(['NRc', 'NRf']):          ('Dual Spool Speed Fault',   'Both fan and core corrected speeds anomalous',        'Full spool speed survey + bearing inspect'),
    frozenset(['NRc', 'Nf']):           ('Shaft Bearing Fault',       'Fan speed vs core corrected speed mismatch',         'Shaft bearing inspect + vibration survey'),

    # ── Missing Combinations / Spool Speed Coupling and Mixed Faults ─────────
    frozenset(['T24', 'T30']):          ('HPC-Fan Thermal Fault',   'Inlet-to-compressor thermal cascade',               'Full gas-path borescope + thermal survey'),
    frozenset(['T24', 'Ps30']):         ('HPC-Fan Fault',           'Inlet temperature rise with HPC static pressure drop','Fan blade survey + HPC borescope'),
    frozenset(['T24', 'P30']):          ('HPC-Fan Fault',           'Inlet temperature rise with HPC delivery pressure drop','Fan blade survey + P30 port inspect'),
    frozenset(['T24', 'htBleed']):      ('HPC Degradation',         'HPC bleed anomaly with inlet temperature rise',      'Bleed valve audit + fan inlet inspect'),
    frozenset(['T24', 'HPT_coolant']):  ('HPT Coolant Loss',        'HPT coolant loss with inlet temperature rise',       'HPT borescope + inlet distortion check'),
    frozenset(['T24', 'phi']):          ('Fuel Metering Fault',     'Fuel-air ratio shift with inlet temperature rise',   'FADEC fuel trim + fan inlet inspect'),
    frozenset(['T30', 'Nf']):           ('HPC-Fan Mechanical Fault', 'HPC temperature rise with fan speed drop',          'HPC borescope + fan balance check'),
    frozenset(['T30', 'NRf']):          ('HPC-Fan Mechanical Fault', 'HPC temperature rise with fan corrected speed drop', 'HPC borescope + fan balance check'),
    frozenset(['T30', 'BPR']):          ('HPC Degradation',         'HPC temperature rise with bypass ratio drop',        'HPC borescope + bypass valve check'),
    frozenset(['T30', 'LPT_coolant']):  ('LPT Thermal Wear',        'LPT coolant degradation with HPC temperature rise',  'LPT coolant system check + HPC borescope'),
    frozenset(['P30', 'Nf']):           ('HPC-Fan Mechanical Fault', 'HPC delivery pressure drop with fan speed loss',    'HPC borescope + fan balance check'),
    frozenset(['P30', 'NRf']):          ('HPC Degradation',         'HPC delivery pressure drop with fan speed mismatch', 'P30 port inspect + fan speed survey'),
    frozenset(['P30', 'BPR']):          ('HPC Degradation',         'HPC delivery pressure drop with bypass anomaly',     'Borescope HPC + bypass valve inspect'),
    frozenset(['P30', 'Nc']):           ('HPC Degradation',         'Core compressor pressure-speed mismatch',            'Borescope HPC stages 3-5 + speed survey'),
    frozenset(['Ps30', 'Nf']):          ('HPC-Fan Mechanical Fault', 'Core pressure drop with fan speed loss',            'HPC stages 3-5 borescope + fan blade inspect'),
    frozenset(['Ps30', 'NRf']):         ('HPC-Fan Aerodynamic Fault','HPC static pressure anomaly with fan speed deviation','HPC borescope + fan blade inspect'),
    frozenset(['Ps30', 'BPR']):         ('HPC Degradation',         'HPC static pressure anomaly with bypass ratio drop', 'Borescope HPC stages 3-5 + bypass valve check'),
    frozenset(['phi', 'Nf']):           ('Fuel Metering Fault',     'Fuel-air ratio shift with fan speed loss',           'FADEC fuel trim + fan blade survey'),
    frozenset(['phi', 'NRf']):          ('Fuel Metering Fault',     'Fuel-air ratio deviation with fan speed drop',       'FADEC fuel trim + fan balance check'),
    frozenset(['phi', 'BPR']):          ('Fuel Metering Fault',     'Fuel-air ratio deviation with bypass ratio drop',    'FADEC fuel trim + bypass valve inspect'),
    frozenset(['htBleed', 'Nf']):       ('HPC Degradation',         'HPC bleed anomaly with fan speed loss',              'Bleed valve audit + fan balance check'),
    frozenset(['htBleed', 'NRf']):      ('HPC Degradation',         'HPC bleed anomaly with fan speed deviation',         'Bleed valve inspect + fan balance check'),
    frozenset(['htBleed', 'BPR']):      ('HPC Degradation',         'HPC bleed anomaly with bypass ratio deviation',      'Bleed valve inspect + bypass valve audit'),
    frozenset(['HPT_coolant', 'Nf']):   ('HPT Coolant Loss',        'HPT coolant degradation with fan speed loss',        'HPT borescope + fan balance check'),
    frozenset(['HPT_coolant', 'NRf']):  ('HPT Coolant Loss',        'HPT coolant loss with fan speed anomaly',            'HPT borescope + fan vibration survey'),
    frozenset(['HPT_coolant', 'BPR']):  ('HPT Coolant Loss',        'HPT coolant loss with bypass ratio deviation',       'HPT borescope + bypass valve inspect'),
    frozenset(['Nc', 'NRf']):           ('Shaft Bearing Fault',     'Fan-core spool speed decoupling / mismatch',         'Shaft bearing inspect + vibration survey'),
    frozenset(['NRc', 'BPR']):          ('Shaft Bearing Fault',     'Core-to-fan spool decoupling with bypass ratio shift','Shaft bearing inspect + spool speed survey'),
}

@dataclass
class FaultDiagnosis:
    fault_mode: str
    confidence: float
    primary_sensor: str
    secondary_sensor: str
    explanation: str
    recommended_action: str

def classify_fault(attributions: Dict[str, float]) -> FaultDiagnosis:
    ranked = sorted(attributions.items(), key=lambda x: abs(x[1]), reverse=True)
    if len(ranked) < 2:
        # Fallback if there are fewer than 2 sensors
        primary = ranked[0][0] if len(ranked) >= 1 else "Unknown"
        secondary = "Unknown"
        top2 = frozenset([primary, secondary])
    else:
        primary = ranked[0][0]
        secondary = ranked[1][0]
        top2 = frozenset([primary, secondary])
        
    mode, expl, action = FAULT_MODE_TABLE.get(top2, ('Unknown Degradation', 'No matching sensor pattern', 'Full inspection recommended'))
    total = sum(abs(v) for v in attributions.values()) + 1e-9
    
    if len(ranked) < 2:
        confidence = abs(ranked[0][1]) / total if len(ranked) >= 1 else 0.0
    else:
        confidence = (abs(ranked[0][1]) + abs(ranked[1][1])) / total
        
    return FaultDiagnosis(mode, confidence, primary, secondary, expl, action)
