PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS drugs (
    drug_id TEXT PRIMARY KEY,
    generic_name TEXT NOT NULL UNIQUE,
    therapeutic_class TEXT,
    cyp_substrate TEXT,
    cyp_inhibitor TEXT,
    cyp_inducer TEXT,
    qt_liability_mv REAL DEFAULT 0,
    renal_clearance_fraction REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ddi_interactions (
    interaction_id TEXT PRIMARY KEY,
    perpetrator_drug_id TEXT REFERENCES drugs(drug_id),
    victim_drug_id TEXT REFERENCES drugs(drug_id),
    mechanism TEXT NOT NULL,
    severity TEXT CHECK(severity IN ('mild','moderate','severe','contraindicated')),
    log_lr REAL NOT NULL DEFAULT 0,
    evidence_level TEXT CHECK(evidence_level IN ('curated','signal','sccs_confirmed')),
    UNIQUE(perpetrator_drug_id, victim_drug_id, mechanism)
);

CREATE TABLE IF NOT EXISTS pkpd_parameters (
    param_id TEXT PRIMARY KEY,
    drug_id TEXT REFERENCES drugs(drug_id),
    param_name TEXT NOT NULL,
    population_mean REAL,
    population_cv REAL,
    iiv_omega REAL
);

CREATE TABLE IF NOT EXISTS biological_variation (
    analyte TEXT PRIMARY KEY,
    cv_within REAL NOT NULL,
    cv_between REAL NOT NULL,
    unit TEXT
);

CREATE TABLE IF NOT EXISTS patients (
    patient_id TEXT PRIMARY KEY,
    age INTEGER, sex TEXT, weight_kg REAL
);

CREATE TABLE IF NOT EXISTS lab_observations (
    obs_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    analyte TEXT NOT NULL,
    value REAL NOT NULL,
    timestamp DATETIME NOT NULL,
    provenance_id TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS imaging_findings (
    finding_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    modality TEXT NOT NULL,
    covariate_name TEXT NOT NULL,
    value_mean REAL NOT NULL,
    value_variance REAL NOT NULL,
    provenance_id TEXT UNIQUE NOT NULL
);

CREATE TABLE IF NOT EXISTS alerts (
    alert_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    module TEXT NOT NULL,
    harm_proposition TEXT NOT NULL,
    posterior_probability REAL,
    log_lr REAL,
    provenance_id TEXT UNIQUE NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS advisor_recommendations (
    rec_id TEXT PRIMARY KEY,
    patient_id TEXT REFERENCES patients(patient_id),
    harm_proposition TEXT NOT NULL,
    recommended_action TEXT NOT NULL,
    expected_utility REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS clinician_feedback (
    feedback_id TEXT PRIMARY KEY,
    rec_id TEXT REFERENCES advisor_recommendations(rec_id),
    action_taken TEXT CHECK(action_taken IN ('accepted','overridden','ignored')),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_labs_patient ON lab_observations(patient_id);
CREATE INDEX IF NOT EXISTS idx_alerts_patient ON alerts(patient_id);
