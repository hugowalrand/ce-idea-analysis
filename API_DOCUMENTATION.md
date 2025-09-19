# 📚 API Documentation - CE Idea Analysis Dashboard

## Overview
Complete API reference for all modules in the CE Idea Analysis Dashboard v2.0. This documentation is designed to help AI agents and developers understand the codebase structure and functionality.

## 🏗️ Core Architecture

### Module Hierarchy
```
📦 Core System
├── 🎯 launch_streamlit.py              # Entry point & configuration
├── 📊 streamlit_dashboard.py           # Main UI application
├── 🔧 comprehensive_data_processor_v3.py # Data processing engine
├── 👥 cofounder_and_ideas_analysis.py  # Co-founder analysis module
└── ✅ comprehensive_validation.py      # Validation & testing suite
```

## 📄 Module Specifications

### 1. `launch_streamlit.py`
**Purpose**: Application entry point with environment setup
**Key Functions**:
- `setup_environment()`: Configure Streamlit settings
- `load_custom_css()`: Apply dashboard styling
- `main()`: Launch dashboard with error handling

**Usage**:
```bash
streamlit run launch_streamlit.py
```

### 2. `comprehensive_data_processor_v3.py`
**Purpose**: Core data processing and trajectory extraction engine

#### Key Functions:

##### `load_and_process_comprehensive_data()`
**Returns**: `(df_trajectories, df_sentiment)`
- **df_trajectories**: DataFrame with 456 validated trajectories
- **df_sentiment**: DataFrame with 278 sentiment responses

**Data Schema - Trajectories**:
```python
{
    'participant': str,           # Participant name
    'cohort': str,               # Cohort identifier (H125, H224, etc.)
    'idea': str,                 # Idea name (standardized)
    'first_rating': int,         # First week rating (1-7 scale)
    'peak_rating': int,          # Peak rating achieved (1-7 scale)
    'last_rating': int,          # Final week rating (1-7 scale)
    'change': float,             # First-to-peak change (validated)
    'weeks_tracked': int,        # Number of weeks with data
    'all_weeks': list,           # List of week identifiers
    'all_ratings': list,         # Corresponding ratings
    'is_cofounder': bool,        # Co-founder status
    'cofounder_idea': str,       # Associated co-founder idea
    'was_founded': bool,         # Founding outcome
    'is_animal_idea': bool,      # Animal vs human classification
    'first_week': str,           # First week identifier
    'peak_week': str,            # Peak week identifier
    'last_week': str             # Last week identifier
}
```

##### `convert_to_standard_scale_comprehensive(value, cohort)`
**Purpose**: Convert different rating scales to standardized 1-7 scale
**Parameters**:
- `value`: Raw rating value
- `cohort`: Cohort identifier

**Scale Conversions**:
- **H123**: -3→+3 scale converted to 1-7 (where -3=1, 0=4, 3=7)
- **2021**: Ranking system converted to 1-7 (1st choice=7, etc.)
- **Others**: Standard 1-7 scale maintained

### 3. `cofounder_and_ideas_analysis.py`
**Purpose**: Co-founder trajectory analysis and comprehensive idea insights

#### Key Functions:

##### `analyze_cofounder_trajectories(df_trajectories)`
**Returns**: DataFrame with co-founder analysis
**Features**:
- 36 co-founder trajectories tracked
- Founding outcome correlations
- Team dynamics insights

##### `create_comprehensive_ideas_table(df_trajectories)`
**Returns**: DataFrame with complete idea analysis
**Includes**:
- 30 unique ideas analyzed
- Average trajectory patterns
- Founding status tracking
- Animal vs human classification

##### `analyze_human_vs_animal_detailed(df_trajectories)`
**Returns**: Detailed classification analysis
**Metrics**:
- 105 animal idea trajectories
- 351 human idea trajectories
- 19 participants with both types <4 rating

### 4. `streamlit_dashboard.py`
**Purpose**: Interactive web dashboard implementation

#### Key Sections:

##### `show_executive_summary()`
**Features**:
- Validated key statistics
- Trajectory overview
- Data completeness metrics

##### `show_interactive_analysis()`
**Features**:
- Dynamic filtering system
- Real-time visualizations
- Comprehensive trajectory explorer

##### `show_cofounder_analysis()`  *[NEW in v2.0]*
**Features**:
- Co-founder trajectory tracking
- Founding outcome analysis
- Team formation insights

##### `show_detailed_explorer()`
**Features**:
- Statistical deep-dive analysis
- Transition probability matrices
- Correlation analysis with custom trendlines

##### `show_verification_tools()`
**Features**:
- Complete data transparency
- Validation result explorer
- Error tracking interface

### 5. `comprehensive_validation.py`
**Purpose**: Complete validation and testing suite

#### Key Validation Functions:

##### `validate_specific_error_cases()`
**Validates**: All 12 specific error cases from team feedback
**Results**: ✅ All cases pass validation

##### `validate_scale_conversions()`
**Tests**: H123 and 2021 scale conversion accuracy
**Results**: ✅ All conversions validated

##### `validate_cofounder_mappings()`
**Checks**: 36 co-founder trajectories across all cohorts
**Results**: ✅ All mappings confirmed

##### `validate_animal_classification()`
**Verifies**: Animal vs human idea classification
**Results**: ✅ 105 vs 351 distribution validated

## 🔍 Data Validation Results

### Comprehensive Test Coverage
```python
# All validations must pass for production deployment
validation_results = {
    'specific_error_cases': '✅ All 12 cases validated',
    'data_filtering': '✅ 2020 excluded, 2021 filtered',
    'scale_conversions': '✅ H123 and 2021 conversions working',
    'cofounder_mappings': '✅ 36 trajectories confirmed',
    'animal_classification': '✅ 105 vs 351 validated',
    'idea_merging': '✅ EAFW/MW/Milkfish consolidated',
    'data_completeness': '✅ No missing values, no duplicates'
}
```

### Key Validation Cases
```python
validated_trajectories = {
    'Amy - CFME': {'expected': '+5', 'actual': '+5', 'status': '✅'},
    'Uttej - PLA': {'expected': '+2', 'actual': '+2', 'status': '✅'},
    'Miri - Tobacco Tax': {'expected': '+3', 'actual': '+3', 'status': '✅'},
    'Nicoll - Kangaroo Care': {'expected': '+2', 'actual': '+2', 'status': '✅'},
    'Victoria - Tobacco Tax': {'expected': '+3', 'actual': '+3', 'status': '✅'}
}
```

## 🛠️ Development Guidelines

### For AI Agents
1. **Always run validation first**: `python comprehensive_validation.py`
2. **Check current state**: Review DASHBOARD_V2_FINAL_SUMMARY.md
3. **Test changes**: Run validation after modifications
4. **Maintain data integrity**: Never modify raw data processing without validation
5. **Document changes**: Update relevant documentation

### Error Handling
- All modules include comprehensive error handling
- Validation suite catches data processing errors
- Dashboard includes graceful fallbacks for missing data
- Logging available for debugging

### Performance Considerations
- Data caching implemented via Streamlit @st.cache_data
- Efficient processing of 456 trajectories
- Optimized visualizations for web deployment
- Minimal dependency footprint

## 🎯 Production Status

**Current State**: ✅ PRODUCTION READY
- All validations pass (100% success rate)
- All team feedback addressed
- Deployment working on Streamlit Community Cloud
- Comprehensive documentation complete

**Key Metrics**:
- 456 validated trajectories
- 77 participants across 6 cohorts
- 36 co-founder trajectories
- 278 sentiment responses analyzed
- 100% test coverage via validation suite

---

**📞 For Support**: Reference this documentation and run `python comprehensive_validation.py` for current system status.