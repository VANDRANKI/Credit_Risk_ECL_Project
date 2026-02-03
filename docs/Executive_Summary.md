# Executive Summary

## Agricultural Loan Portfolio CECL and Stress Testing

---

**Prepared for:** Executive Leadership
**Author:** Prabhu

---

## Overview

This report summarizes the findings from our CECL (Current Expected Credit Losses) credit risk analysis of the loan portfolio, with specific focus on the agricultural lending segment. The analysis includes expected credit loss estimation and stress testing under adverse economic scenarios.

---

## 1. Portfolio Description

### Portfolio Composition

| Segment | Loan Count | Total Exposure | Share of Portfolio |
|---------|------------|----------------|-------------------|
| **Agricultural** | ~15,000+ | $300M+ | ~1.5% |
| **Non-Agricultural** | ~1.2M | $16B+ | ~98.5% |
| **Total Portfolio** | ~1.2M | $16.5B+ | 100% |

### Agricultural Segment Definition

The agricultural portfolio consists of:
- **Small business loans** in the top 10 agricultural states
- **Geographic focus:** CA, IA, NE, TX, MN, IL, KS, WI, IN, NC
- Represents productive/commercial lending in agricultural regions

### Key Characteristics

| Metric | Agricultural | Non-Agricultural |
|--------|-------------|------------------|
| Average Loan Size | $18,000+ | $14,000+ |
| Average FICO Score | ~670 | ~695 |
| Average DTI | ~19% | ~17% |

---

## 2. Baseline Risk Assessment

### Default Rates

| Segment | Default Rate |
|---------|-------------|
| **Agricultural Portfolio** | ~26% |
| **Non-Agricultural Portfolio** | ~20% |
| **Overall Portfolio** | ~20% |

**Key Finding:** The agricultural segment exhibits a default rate approximately 1.3x higher than the non-agricultural segment, driven by the higher-risk profile of small business loans.

### Expected Credit Loss (ECL)

| Metric | Agricultural | Non-Agricultural | Total Portfolio |
|--------|-------------|------------------|-----------------|
| **Total ECL** | $24M+ | $1.1B+ | $1.2B+ |
| **ECL Rate** | ~8.0% | ~7.0% | ~7.2% |

**Key Finding:** The agricultural segment contributes disproportionately to total ECL relative to its share of exposure, reflecting its higher risk profile.

---

## 3. Stress Testing Results

### Scenario Definitions

| Scenario | Economic Conditions | Income Impact | DTI Impact |
|----------|---------------------|---------------|------------|
| **Baseline** | Current conditions | 0% | 0% |
| **Moderate Stress** | Economic slowdown | -10% | +15% |
| **Severe Stress** | Recession | -20% | +30% |
| **Agricultural Crisis** | Sector downturn | -25% | +35% |

### Portfolio-Level Impact

| Scenario | Total ECL | ECL Change | Additional Provision |
|----------|-----------|------------|---------------------|
| **Baseline** | $1.2B | - | - |
| **Moderate Stress** | $1.5B | +26% | +$310M |
| **Severe Stress** | $1.9B | +56% | +$670M |
| **Agricultural Crisis** | $2.1B | +76% | +$910M |

### Agricultural Segment Vulnerability

| Scenario | Agri ECL | Agri ECL Change | Non-Agri ECL Change |
|----------|----------|-----------------|---------------------|
| **Moderate Stress** | $32M | +33% | +26% |
| **Severe Stress** | $43M | +79% | +55% |
| **Agricultural Crisis** | $50M | +109% | +74% |

**Key Finding:** The agricultural segment is significantly more sensitive to economic stress, with ECL increases 1.3-1.5x greater than the non-agricultural segment under identical scenarios.

---

## 4. Key Implications

### Risk Appetite

1. **Elevated Agricultural Risk:** The agricultural segment requires enhanced monitoring and potentially higher risk-adjusted returns to compensate for elevated default risk
2. **Concentration Risk:** Geographic concentration in agricultural states creates correlated default risk during regional economic downturns
3. **Stress Sensitivity:** Agricultural borrowers are more vulnerable to income shocks and economic stress

### Capital Planning

1. **Baseline Allowance:** Current ECL estimates should inform CECL allowance requirements
2. **Stress Buffers:** Consider maintaining capital buffers of 50-75% of baseline ECL to absorb severe stress scenarios
3. **Agricultural Overlay:** Additional provisions may be warranted for the agricultural segment given its elevated risk profile

### Lending Policy Recommendations

1. **Underwriting Standards:**
   - Consider tighter DTI limits for small business loans in agricultural regions
   - Enhanced income verification for agricultural segment borrowers
   - FICO floor considerations for higher-risk segments

2. **Portfolio Limits:**
   - Monitor agricultural segment concentration
   - Consider geographic diversification within agricultural states
   - Balance risk-return trade-offs across segments

3. **Monitoring Enhancement:**
   - Establish early warning indicators for agricultural segment
   - Track income and employment trends in agricultural regions
   - Monitor commodity prices and agricultural economic indicators

---

## 5. Summary

### Key Takeaways

1. **CECL Readiness:** The portfolio ECL analysis provides a foundation for CECL compliance and allowance calculation

2. **Agricultural Risk:** The agricultural segment, while small, presents elevated credit risk requiring careful management

3. **Stress Resilience:** Under severe stress, portfolio ECL could increase by 50-75%, requiring adequate capital buffers

4. **Proactive Management:** Early identification of deteriorating conditions through monitoring can mitigate losses

### Recommended Actions

| Priority | Action |
|----------|--------|
| **High** | Review agricultural segment underwriting standards |
| **High** | Establish stress testing capital buffers |
| **Medium** | Implement enhanced monitoring for agricultural segment |
| **Medium** | Review portfolio concentration limits |
| **Low** | Develop regional economic early warning indicators |

---

## Appendix: Methodology Notes

### Data Source
- Lending Club loan data (Zenodo)
- ~1.3 million loans, 2007-2018 vintages

### Model Approach
- **PD Model:** Machine learning (Logistic Regression, XGBoost)
- **LGD:** Fixed 45% assumption (Basel II standard)
- **EAD:** Loan amount (conservative)
- **ECL:** PD x LGD x EAD

### Limitations
- Agricultural segment is a proxy constructed from consumer loan data
- Historical data may not fully reflect future economic conditions
- Fixed LGD assumption may not capture actual recovery experience

---

*For detailed methodology and technical specifications, refer to the Model Development Document.*

---

**Contact:** Prabhu | Data Analyst | Data Specialist |Quantitative Risk Analyst

*This analysis was prepared for internal decision-making purposes and should be interpreted in conjunction with other risk management information.*
