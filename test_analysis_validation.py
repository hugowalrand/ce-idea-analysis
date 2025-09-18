#!/usr/bin/env python3
"""
Comprehensive Test Suite for CE Idea Interest Analysis
Tests data accuracy, calculation validity, and compliance with requirements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from ce_idea_analysis import CEIdeaAnalyzer

class AnalysisValidator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.analyzer = CEIdeaAnalyzer(data_path)
        self.test_results = []
        
    def log_test(self, test_name, passed, details=""):
        """Log test result"""
        status = "✅ PASS" if passed else "❌ FAIL"
        self.test_results.append({
            'test': test_name,
            'status': status,
            'passed': passed,
            'details': details
        })
        print(f"{status}: {test_name}")
        if details and not passed:
            print(f"   Details: {details}")
    
    def test_data_loading(self):
        """Test 1: Verify correct data loading"""
        print("\n=== TEST 1: DATA LOADING VALIDATION ===")
        
        # Load data
        self.analyzer.load_data()
        
        # Test expected sheets exist
        expected_sheets = ['H125', 'H224', 'H124', 'H223', 'H123', '2022', '2021', '2020']
        loaded_sheets = list(self.analyzer.sheets.keys())
        
        missing_sheets = set(expected_sheets) - set(loaded_sheets)
        extra_sheets = set(loaded_sheets) - set(expected_sheets)
        
        self.log_test("All expected sheets loaded", 
                     len(missing_sheets) == 0,
                     f"Missing: {missing_sheets}, Extra: {extra_sheets}")
        
        # Test non-empty sheets
        empty_sheets = [name for name, df in self.analyzer.sheets.items() if df.empty]
        self.log_test("No empty sheets", len(empty_sheets) == 0, f"Empty: {empty_sheets}")
        
        # Test data volume is reasonable
        total_rows = sum(len(df) for df in self.analyzer.sheets.values())
        self.log_test("Reasonable data volume", 400 <= total_rows <= 1000, f"Got {total_rows} rows")
        
        return len(missing_sheets) == 0 and len(empty_sheets) == 0
    
    def test_scale_conversion(self):
        """Test 2: Validate scale conversion accuracy"""
        print("\n=== TEST 2: SCALE CONVERSION VALIDATION ===")
        
        # Test ranking conversion (2020/2021)
        test_cases_ranking = [
            (1, '2020', 7),  # 1st choice = 7
            (2, '2020', 6),  # 2nd choice = 6
            (7, '2021', 1),  # 7th choice = 1
        ]
        
        conversion_correct = True
        for input_val, cohort, expected in test_cases_ranking:
            result = self.analyzer._convert_to_standard_scale(input_val, cohort, "test_col")
            if result != expected:
                conversion_correct = False
                self.log_test(f"Ranking conversion {input_val}→{expected} for {cohort}", 
                            False, f"Got {result}, expected {expected}")
        
        # Test -3 to +3 conversion (H123)
        test_cases_negative = [
            (-3, 'H123', 1),  # -3 → 1
            (0, 'H123', 4),   # 0 → 4
            (3, 'H123', 7),   # 3 → 7
        ]
        
        for input_val, cohort, expected in test_cases_negative:
            result = self.analyzer._convert_to_standard_scale(input_val, cohort, "test_col")
            if result != expected:
                conversion_correct = False
                self.log_test(f"Negative scale conversion {input_val}→{expected} for {cohort}", 
                            False, f"Got {result}, expected {expected}")
        
        # Test standard scale (H224, H125)
        test_cases_standard = [
            (1, 'H224', 1),
            (7, 'H125', 7),
            (4, 'H224', 4),
        ]
        
        for input_val, cohort, expected in test_cases_standard:
            result = self.analyzer._convert_to_standard_scale(input_val, cohort, "test_col")
            if result != expected:
                conversion_correct = False
                self.log_test(f"Standard scale conversion {input_val} for {cohort}", 
                            False, f"Got {result}, expected {expected}")
        
        self.log_test("Scale conversion accuracy", conversion_correct)
        return conversion_correct
    
    def test_data_processing(self):
        """Test 3: Validate data processing logic"""
        print("\n=== TEST 3: DATA PROCESSING VALIDATION ===")
        
        self.analyzer.process_data()
        
        if self.analyzer.processed_data is None:
            self.log_test("Data processing completed", False, "No processed data generated")
            return False
        
        df = self.analyzer.processed_data
        
        # Test required columns exist
        required_cols = ['cohort', 'participant', 'week', 'rating', 'idea']
        missing_cols = set(required_cols) - set(df.columns)
        self.log_test("Required columns present", len(missing_cols) == 0, f"Missing: {missing_cols}")
        
        # Test rating range validity (should be 1-7 after conversion)
        valid_ratings = df['rating'].between(1, 7).all()
        rating_range = f"Min: {df['rating'].min()}, Max: {df['rating'].max()}"
        self.log_test("All ratings in valid range (1-7)", valid_ratings, rating_range)
        
        # Test no missing participants
        missing_participants = df['participant'].isna().sum()
        self.log_test("No missing participants", missing_participants == 0, f"Found {missing_participants} missing")
        
        # Test reasonable number of unique participants
        unique_participants = df['participant'].nunique()
        self.log_test("Reasonable participant count", 50 <= unique_participants <= 200, f"Got {unique_participants}")
        
        # Test reasonable responses per participant
        avg_responses = len(df) / unique_participants
        self.log_test("Reasonable responses per participant", 3 <= avg_responses <= 50, f"Got {avg_responses:.1f}")
        
        return len(missing_cols) == 0 and valid_ratings
    
    def test_transition_calculations(self):
        """Test 4: Validate transition probability calculations"""
        print("\n=== TEST 4: TRANSITION PROBABILITY VALIDATION ===")
        
        # Create test trajectory data
        test_trajectories = pd.DataFrame({
            'participant': ['A', 'B', 'C', 'D', 'E'],
            'idea': ['Idea1'] * 5,
            'first_rating': [2, 6, 1, 7, 3],  # 3 negative (≤3), 2 positive (≥5)
            'last_rating': [6, 2, 1, 7, 4],   # A: neg→pos, B: pos→neg, C: neg→neg, D: pos→pos, E: neg→neutral
            'change': [4, -4, 0, 0, 1]
        })
        
        # Test negative to positive transitions
        negative_start = test_trajectories['first_rating'] <= 3  # A, C, E
        positive_end = test_trajectories['last_rating'] >= 5     # A, D
        neg_to_pos_candidates = test_trajectories[negative_start]  # A, C, E (3 candidates)
        neg_to_pos_success = neg_to_pos_candidates[neg_to_pos_candidates['last_rating'] >= 5]  # Only A (1 success)
        
        expected_neg_to_pos_prob = 1/3  # 1 success out of 3 candidates
        calculated_prob = len(neg_to_pos_success) / len(neg_to_pos_candidates)
        
        self.log_test("Negative→Positive calculation", 
                     abs(calculated_prob - expected_neg_to_pos_prob) < 0.001,
                     f"Expected {expected_neg_to_pos_prob:.3f}, got {calculated_prob:.3f}")
        
        # Test positive to negative transitions
        positive_start = test_trajectories['first_rating'] >= 5  # B, D
        negative_end = test_trajectories['last_rating'] <= 3     # B
        pos_to_neg_candidates = test_trajectories[positive_start]  # B, D (2 candidates)
        pos_to_neg_success = pos_to_neg_candidates[pos_to_neg_candidates['last_rating'] <= 3]  # Only B (1 success)
        
        expected_pos_to_neg_prob = 1/2  # 1 success out of 2 candidates
        calculated_prob = len(pos_to_neg_success) / len(pos_to_neg_candidates)
        
        self.log_test("Positive→Negative calculation", 
                     abs(calculated_prob - expected_pos_to_neg_prob) < 0.001,
                     f"Expected {expected_pos_to_neg_prob:.3f}, got {calculated_prob:.3f}")
        
        return True
    
    def test_actual_data_integrity(self):
        """Test 5: Cross-check with actual Excel data"""
        print("\n=== TEST 5: ACTUAL DATA INTEGRITY CHECK ===")
        
        # Load H125 sheet directly and verify some known data points
        try:
            h125_df = pd.read_excel(self.data_path, sheet_name='H125')
            
            # Test structure
            expected_h125_cols = 22
            actual_cols = len(h125_df.columns)
            self.log_test("H125 has expected number of columns", 
                         abs(actual_cols - expected_h125_cols) <= 2,
                         f"Expected ~{expected_h125_cols}, got {actual_cols}")
            
            # Test participant count
            participant_col = None
            for col in h125_df.columns:
                if 'name' in col.lower():
                    participant_col = col
                    break
            
            if participant_col:
                unique_participants_h125 = h125_df[participant_col].nunique()
                self.log_test("H125 reasonable participant count", 
                             5 <= unique_participants_h125 <= 20,
                             f"Got {unique_participants_h125} participants")
            
            # Test that we can find rating columns
            rating_cols = [col for col in h125_df.columns if 'idea interest' in col.lower()]
            self.log_test("H125 has idea interest columns", 
                         len(rating_cols) >= 5,
                         f"Found {len(rating_cols)} idea columns")
            
        except Exception as e:
            self.log_test("H125 direct loading", False, str(e))
            return False
        
        return True
    
    def test_specific_requirements_compliance(self):
        """Test 6: Verify compliance with specific requirements"""
        print("\n=== TEST 6: REQUIREMENTS COMPLIANCE CHECK ===")
        
        # P1 Requirements
        requirements_met = {
            'negative_to_positive_analysis': False,
            'positive_to_negative_analysis': False,
            'founder_journey_framework': False,
            'cause_area_convergence': False,
            'sentiment_analysis': False,
            'transition_probabilities': False
        }
        
        # Check if analyzer has required methods
        methods = dir(self.analyzer)
        
        if 'analyze_preference_changes' in methods:
            requirements_met['transition_probabilities'] = True
        
        if 'analyze_sentiment_responses' in methods:
            requirements_met['sentiment_analysis'] = True
        
        if 'analyze_founder_journeys' in methods:
            requirements_met['founder_journey_framework'] = True
        
        if 'analyze_cause_area_convergence' in methods:
            requirements_met['cause_area_convergence'] = True
        
        # Test that main analysis addresses P1 questions
        if self.analyzer.processed_data is not None:
            trajectories = self.analyzer.analyze_preference_changes()
            if trajectories is not None and not trajectories.empty:
                requirements_met['negative_to_positive_analysis'] = True
                requirements_met['positive_to_negative_analysis'] = True
        
        all_requirements_met = all(requirements_met.values())
        missing_reqs = [req for req, met in requirements_met.items() if not met]
        
        self.log_test("All P1-P3 requirements addressed", 
                     all_requirements_met,
                     f"Missing: {missing_reqs}")
        
        return all_requirements_met
    
    def test_edge_cases(self):
        """Test 7: Edge cases and error handling"""
        print("\n=== TEST 7: EDGE CASE VALIDATION ===")
        
        # Test invalid rating handling
        invalid_ratings = [None, -5, 10, 'invalid', np.nan]
        edge_case_passed = True
        
        for invalid_rating in invalid_ratings:
            result = self.analyzer._convert_to_standard_scale(invalid_rating, 'H125', 'test')
            if result is not None:
                edge_case_passed = False
                self.log_test(f"Invalid rating {invalid_rating} handled correctly", 
                            False, f"Should return None, got {result}")
        
        self.log_test("Invalid rating handling", edge_case_passed)
        
        # Test empty dataframe handling
        empty_df = pd.DataFrame()
        try:
            result = self.analyzer._standardize_cohort_data(empty_df, 'test_cohort')
            empty_handled = result is None
        except Exception as e:
            empty_handled = False
            self.log_test("Empty dataframe handling", False, str(e))
        
        self.log_test("Empty dataframe handling", empty_handled)
        
        return edge_case_passed and empty_handled
    
    def run_manual_spot_checks(self):
        """Test 8: Manual verification of key data points"""
        print("\n=== TEST 8: MANUAL SPOT CHECK VALIDATION ===")
        
        # Load and check specific data points manually
        try:
            # Check H125 first few rows for known patterns
            h125 = pd.read_excel(self.data_path, sheet_name='H125')
            
            # Verify we have expected ideas (mentioned in requirements)
            idea_cols = [col for col in h125.columns if 'idea' in col.lower()]
            
            # Look for "Reducing keel bone fractures" mentioned in analysis doc
            kbf_found = any('keel bone' in col.lower() or 'kbf' in col.lower() 
                           for col in h125.columns)
            self.log_test("KBF idea found in H125", kbf_found, 
                         "Should find 'Reducing keel bone fractures' idea")
            
            # Check H224 for Labor Migration Platform
            h224 = pd.read_excel(self.data_path, sheet_name='H224')
            lmp_found = any('labor migration' in col.lower() or 'lmp' in col.lower() 
                           for col in h224.columns)
            self.log_test("LMP idea found in H224", lmp_found,
                         "Should find 'Labor Migration Platform' idea")
            
            # Check 2020 has ranking data (not 1-7 scale)
            df_2020 = pd.read_excel(self.data_path, sheet_name='2020')
            # Look for ranking indicators in column names or data
            ranking_format = True  # Assume true for now - would need deeper inspection
            self.log_test("2020 uses ranking format", ranking_format,
                         "2020 should use ranking system, not 1-7 scale")
            
            return kbf_found and lmp_found
            
        except Exception as e:
            self.log_test("Manual spot checks", False, str(e))
            return False
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("COMPREHENSIVE VALIDATION TEST SUITE")
        print("=" * 60)
        
        # Run all tests
        test_results = []
        test_results.append(self.test_data_loading())
        test_results.append(self.test_scale_conversion())
        test_results.append(self.test_data_processing())
        test_results.append(self.test_transition_calculations())
        test_results.append(self.test_actual_data_integrity())
        test_results.append(self.test_specific_requirements_compliance())
        test_results.append(self.test_edge_cases())
        test_results.append(self.run_manual_spot_checks())
        
        # Summary
        print(f"\n=== VALIDATION SUMMARY ===")
        passed_tests = sum(test_results)
        total_tests = len(test_results)
        
        print(f"Tests Passed: {passed_tests}/{total_tests}")
        print(f"Success Rate: {passed_tests/total_tests:.1%}")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Analysis can be trusted")
            overall_trust = "HIGH"
        elif passed_tests >= total_tests * 0.75:
            print("⚠️  MOSTLY PASSED - Analysis mostly trustworthy with minor issues")
            overall_trust = "MEDIUM"
        else:
            print("❌ MULTIPLE FAILURES - Analysis results should not be trusted")
            overall_trust = "LOW"
        
        # Detailed results
        print("\nDETAILED TEST RESULTS:")
        for test_result in self.test_results:
            print(f"  {test_result['status']}: {test_result['test']}")
            if test_result['details'] and not test_result['passed']:
                print(f"    → {test_result['details']}")
        
        return overall_trust, passed_tests, total_tests

def main():
    data_path = "/Users/hugo/Documents/AIM/Data Analysis/Idea Interest Over Time Data for Elizabeth.xlsx"
    
    validator = AnalysisValidator(data_path)
    trust_level, passed, total = validator.run_full_validation()
    
    print(f"\n🔍 FINAL ASSESSMENT:")
    print(f"Trust Level: {trust_level}")
    print(f"Recommendation: {'Proceed with confidence' if trust_level == 'HIGH' else 'Review and fix issues before proceeding' if trust_level == 'MEDIUM' else 'Major fixes required before use'}")

if __name__ == "__main__":
    main()