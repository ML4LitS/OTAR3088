headers = ["indication", "nct_id", "study_title", "study_phase", "adverse_event", "event_type", "organ_system", "subjects_at_risk", "subjects_affected", "ctgov_group_code", "result_group_id"]
nct_id = "NCT01753193"
# nct_id = "NCT05034952" # Efficacy and Safety VX-548
query = f"""
SELECT 
	c.names AS indication,
    s.nct_id,
    s.brief_title AS study_title,
    s.phase AS study_phase,
	ae.adverse_event_term as adverse_event,
    ae.event_type,
	ae.organ_system,
	ae.subjects_at_risk,
    ae.subjects_affected as subjects_affected,
	ae.ctgov_group_code,
    -- ae.description,
	ae.result_group_id
FROM 
    "reported_events" as ae
JOIN
	all_conditions as c on c.nct_id = ae.nct_id
JOIN
    studies as s ON s.nct_id = ae.nct_id
WHERE
	ae.nct_id = '{nct_id}';
"""

# Postgres account for AACT
user="YOUR_USERNAME"
password="YOUR_PASSWORD"