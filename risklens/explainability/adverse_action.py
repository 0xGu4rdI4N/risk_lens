"""Regulatory Adverse Action Notice generator for declined credit applicants."""

from typing import Dict, List, Union


# Standard adverse action mapping for fair lending compliance
REASON_CODE_MAP = {
    "cibil_score": {
        "code": "AAR-01",
        "description": "Credit bureau score does not meet minimum underwriting criteria",
        "guidance": "Improve credit score by paying outstanding accounts on time."
    },
    "revolving_utilization": {
        "code": "AAR-02",
        "description": "Proportion of revolving credit balances to total credit limits is excessively elevated",
        "guidance": "Pay down existing revolving credit card balances below 30% utilization."
    },
    "debt_to_income": {
        "code": "AAR-03",
        "description": "Total monthly debt obligations too high relative to verified gross monthly income",
        "guidance": "Reduce non-mortgage monthly loan obligations or verify additional income sources."
    },
    "delinquencies_2yr": {
        "code": "AAR-04",
        "description": "Presence of recent delinquent credit payments or past-due accounts in bureau history",
        "guidance": "Ensure no 30+ day delinquencies occur over the next 12-24 months."
    },
    "inquiries_last_6m": {
        "code": "AAR-05",
        "description": "Too many recent hard inquiries for new credit lines indicating credit hunger",
        "guidance": "Avoid applying for multiple credit cards or consumer loans within short periods."
    },
    "credit_history_months": {
        "code": "AAR-06",
        "description": "Insufficient length of credit bureau history",
        "guidance": "Establish seasoned credit trade lines over longer continuous timeframes."
    },
    "device_risk_score": {
        "code": "AAR-07",
        "description": "Digital security risk: Device anomalies or authentication risk factors identified",
        "guidance": "Contact fraud prevention desk to complete secondary two-factor verification."
    },
    "transaction_velocity_24h": {
        "code": "AAR-08",
        "description": "Atypical transaction frequency or volume spikes observed in past 24 hours",
        "guidance": "Verify authorized activity with customer care operations."
    },
    "annual_income": {
        "code": "AAR-09",
        "description": "Stated verifiable income does not meet minimum threshold for requested line limit",
        "guidance": "Consider requesting a lower starting credit line or providing supplementary ITR returns."
    },
}


class AdverseActionGenerator:
    """Translates high-risk SHAP drivers into FCRA / ECOA / RBI compliant decline notices."""

    @staticmethod
    def generate_reasons(
        shap_impacts: List[Dict[str, Union[str, float]]],
        max_reasons: int = 4
    ) -> List[str]:
        """Extract top positive SHAP attributions (risk increasing) and convert to legal notices."""
        # Filter to only risk-increasing features
        risk_drivers = [
            item for item in shap_impacts
            if item.get("shap_value", 0) > 0 or item.get("direction") == "RISK_INCREASING"
        ]
        
        # Sort by magnitude
        risk_drivers.sort(key=lambda x: abs(x.get("shap_value", 0)), reverse=True)
        
        reasons: List[str] = []
        for item in risk_drivers[:max_reasons]:
            raw_feature = item["feature"]
            # Extract base feature name if one-hot encoded (e.g. merchant_category_crypto -> merchant_category)
            matched_key = None
            for key in REASON_CODE_MAP:
                if key in raw_feature:
                    matched_key = key
                    break
                    
            if matched_key and matched_key in REASON_CODE_MAP:
                meta = REASON_CODE_MAP[matched_key]
                reasons.append(f"[{meta['code']}] {meta['description']}")
            else:
                clean_name = raw_feature.replace("_", " ").title()
                reasons.append(f"[AAR-GEN] Adverse factor: {clean_name}")
                
        if not reasons:
            reasons.append("[AAR-10] Combined risk factors exceeded the portfolio cost-sensitive cutoff threshold.")
            
        return reasons
