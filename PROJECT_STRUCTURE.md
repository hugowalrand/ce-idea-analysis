# 📁 Project Structure Guide

## 🎯 CE Idea Analysis Dashboard v2.0 - Complete File Reference

### 📦 Root Directory Structure
```
ce-idea-analysis/
├── 📄 README.md                        # Main project documentation
├── 📚 API_DOCUMENTATION.md             # Complete API reference
├── 📋 PROJECT_STRUCTURE.md             # This file - project organization
├── 🚀 DEPLOYMENT_INSTRUCTIONS.md       # Streamlit Cloud deployment guide
├── 📊 DASHBOARD_V2_FINAL_SUMMARY.md    # Project completion summary
├── .gitignore                          # Git ignore rules
├── requirements.txt                    # Python dependencies
├── 📁 Idea Interest Over Time Data for Elizabeth.xlsx # Source data
├── 🎯 launch_streamlit.py              # Application entry point
├── 📊 streamlit_dashboard.py           # Main dashboard UI
├── 🔧 comprehensive_data_processor_v3.py # Data processing engine
├── 👥 cofounder_and_ideas_analysis.py  # Co-founder analysis
├── ✅ comprehensive_validation.py      # Validation suite
├── 📁 dev_tools/                       # Development utilities
│   ├── check_processed_data.py         # Data inspection tool
│   ├── debug_peaks.py                  # Peak calculation debugger
│   └── targeted_bug_fixes.py           # Bug investigation tool
└── 📁 archive/                         # Historical files
    ├── streamlit_data_processor.py     # Previous data processor
    └── DASHBOARD_FINAL_SUMMARY.md      # Previous summary
```

## 🔧 Core Application Files

### Production Files (Required for Deployment)
- ✅ `launch_streamlit.py` - Entry point for Streamlit
- ✅ `streamlit_dashboard.py` - Main dashboard application
- ✅ `comprehensive_data_processor_v3.py` - Core data engine
- ✅ `cofounder_and_ideas_analysis.py` - Co-founder analysis
- ✅ `requirements.txt` - Dependencies
- ✅ `Idea Interest Over Time Data for Elizabeth.xlsx` - Source data

### Documentation Files
- 📄 `README.md` - Main project documentation with AI context
- 📚 `API_DOCUMENTATION.md` - Complete API reference
- 🚀 `DEPLOYMENT_INSTRUCTIONS.md` - Streamlit deployment guide
- 📊 `DASHBOARD_V2_FINAL_SUMMARY.md` - Project completion summary

### Development & Validation
- ✅ `comprehensive_validation.py` - Complete test suite (RUN THIS FIRST)
- 📁 `dev_tools/` - Development utilities and debugging tools
- 📁 `archive/` - Historical versions and deprecated files

## 🎯 File Purposes & Context for AI Agents

### 🚨 CRITICAL FOR AI AGENTS TO UNDERSTAND:

#### 1. **Start Here**: `comprehensive_validation.py`
```bash
python comprehensive_validation.py
# Must output: "🎉 ALL VALIDATIONS PASSED - DASHBOARD v2.0 IS ERROR-FREE!"
```
**Purpose**: Complete validation of all data processing and calculations
**Status**: ✅ All 456 trajectories validated, all team feedback addressed

#### 2. **Core Data Engine**: `comprehensive_data_processor_v3.py`
**Purpose**: Main data processing with validated trajectory calculations
**Key Features**:
- First-to-peak trajectory logic (matches team expectations)
- Scale conversions: H123 (-3→+3 to 1-7), 2021 (rankings to 1-7)
- Co-founder mapping with founding outcomes
- Animal vs human classification

#### 3. **Dashboard UI**: `streamlit_dashboard.py`
**Purpose**: Interactive web interface with 5 main sections
**Features**:
- Executive Summary with validated statistics
- Interactive Analysis with filtering
- Co-founder Analysis (NEW in v2.0)
- Statistical Deep-dive with custom trendlines
- Verification Tools for transparency

#### 4. **Entry Point**: `launch_streamlit.py`
**Purpose**: Application launcher with environment setup
**Usage**: `streamlit run launch_streamlit.py`

## 📊 Data Schema & Processing

### Input Data: `Idea Interest Over Time Data for Elizabeth.xlsx`
**Sheets Processed**:
- H125 (12 participants, 96 trajectories)
- H224 (10 participants, 59 trajectories)
- H124 (15 participants, 135 trajectories)
- H223 (9 participants, 54 trajectories)
- H123 (14 participants, 112 trajectories)
- 2021/2022 (Filtered and processed)

### Output Data Structure:
```python
trajectory_record = {
    'participant': 'Amy',
    'cohort': 'H125',
    'idea': 'CFME',
    'first_rating': 1,      # Week 1 rating
    'peak_rating': 6,       # Highest rating achieved
    'last_rating': 2,       # Final week rating
    'change': 5,            # First-to-peak change (VALIDATED)
    'is_cofounder': True,   # Co-founder status
    'was_founded': True,    # Founding outcome
    'is_animal_idea': True  # Classification
}
```

## 🔍 Validation & Quality Assurance

### Validation Coverage: 100%
```python
validation_results = {
    'specific_error_cases': '✅ Amy (+5), Uttej (+2), Miri (+3), etc.',
    'data_filtering': '✅ 2020 excluded, 2021 filtered',
    'scale_conversions': '✅ H123 & 2021 working correctly',
    'cofounder_mappings': '✅ 36 trajectories confirmed',
    'animal_classification': '✅ 105 animal vs 351 human',
    'data_completeness': '✅ No missing values or duplicates'
}
```

### Key Validated Cases:
- **Amy - CFME**: Expected +5, Got +5 ✅
- **Uttej - PLA**: Expected +2, Got +2 ✅
- **Miri - Tobacco Tax**: Expected +3, Got +3 ✅
- **Nicoll - Kangaroo Care**: Expected +2, Got +2 ✅
- **Victoria - Tobacco Tax**: Expected +3, Got +3 ✅

## 🚀 Deployment Status

### Production Ready ✅
- **GitHub**: https://github.com/hugowalrand/ce-idea-analysis
- **Streamlit Cloud**: Ready for deployment
- **Dependencies**: All resolved (no statsmodels/matplotlib issues)
- **Validation**: 100% pass rate

### For AI Agents Working on This Project:

#### ⚠️ BEFORE MAKING ANY CHANGES:
1. **Run validation**: `python comprehensive_validation.py`
2. **Read context**: Review `API_DOCUMENTATION.md`
3. **Understand data**: Check `DASHBOARD_V2_FINAL_SUMMARY.md`

#### 🔄 Development Workflow:
1. Make changes to code
2. Run validation suite
3. Ensure all tests pass
4. Update documentation if needed
5. Test deployment locally

#### 🎯 Current Status:
- **All trajectory calculations VALIDATED** ✅
- **All team feedback ADDRESSED** ✅
- **Production deployment WORKING** ✅
- **Documentation COMPLETE** ✅

---

**📞 Questions?** Check `comprehensive_validation.py` output for current system status.