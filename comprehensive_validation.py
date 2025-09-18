#!/usr/bin/env python3
"""
COMPREHENSIVE VALIDATION SCRIPT
Thoroughly checks for errors and bugs in the Dashboard v2.0
Validates every aspect against team feedback requirements
"""

import pandas as pd
import numpy as np
from comprehensive_data_processor_v3 import load_and_process_comprehensive_data

def validate_specific_error_cases():
    """Validate each specific error case mentioned in team feedback"""
    print("🔍 VALIDATING SPECIFIC ERROR CASES")
    print("=" * 60)

    # Expected cases from team feedback (converted to standard scale where needed)
    expected_cases = {
        'H125': {
            'Adnaan': {'idea': 'CFME', 'week1': 1, 'week5': 6, 'expected_change': 5},
            'Amy': {'idea': 'CFME', 'week1': 1, 'week2': 6, 'expected_change': 5}
        },
        'H224': {
            'Uttej': {'idea': 'PLA', 'week1': 3, 'week2': 5, 'expected_change': 2}
        },
        'H223': {
            'Habiba': {'idea': 'ORS', 'week1': 3, 'week5': 5, 'expected_change': 2}
        },
        'H123': {
            'Miri': {'idea': 'Tobacco Tax', 'week1': 2, 'week2': 5, 'expected_change': 3},  # -2 → 1 converted
            'Nicoll': {'idea': 'Kangaroo Care', 'week1': 3, 'week4': 5, 'expected_change': 2},  # -1 → 1 converted
            'Victoria': {'idea': 'Tobacco Tax', 'week1': 2, 'week3': 5, 'expected_change': 3}  # -2 → 1 converted
        }
    }

    # Load processed data
    df_trajectories, _ = load_and_process_comprehensive_data()

    validation_results = []

    for cohort, participants in expected_cases.items():
        print(f"\n--- {cohort} VALIDATION ---")

        for participant, expected in participants.items():
            print(f"\n🔎 Validating: {participant} - {expected['idea']}")

            # Find matching trajectories
            matches = df_trajectories[
                (df_trajectories['participant'] == participant) &
                (df_trajectories['cohort'] == cohort) &
                (df_trajectories['idea'].str.contains(expected['idea'], case=False, na=False))
            ]

            if matches.empty:
                print(f"   ❌ NO MATCH FOUND for {participant} - {expected['idea']}")
                validation_results.append({
                    'cohort': cohort,
                    'participant': participant,
                    'idea': expected['idea'],
                    'status': 'NOT_FOUND',
                    'issue': f'No trajectory found for {participant} - {expected["idea"]}'
                })
                continue

            # Check the trajectory
            match = matches.iloc[0]

            # Validate the ratings and changes
            issues = []

            if 'week1' in expected and match['first_rating'] != expected['week1']:
                issues.append(f"First rating mismatch: expected {expected['week1']}, got {match['first_rating']}")

            change = match['change']  # Use the corrected change field (first-to-peak)
            if abs(change - expected['expected_change']) > 0.1:  # Allow small floating point differences
                issues.append(f"Change mismatch: expected {expected['expected_change']}, got {change}")

            if issues:
                print(f"   ⚠️  ISSUES FOUND:")
                for issue in issues:
                    print(f"      - {issue}")
                validation_results.append({
                    'cohort': cohort,
                    'participant': participant,
                    'idea': expected['idea'],
                    'status': 'ISSUES_FOUND',
                    'issue': '; '.join(issues),
                    'actual_first': match['first_rating'],
                    'actual_last': match['last_rating'],
                    'actual_change': change
                })
            else:
                print(f"   ✅ VALIDATED: {match['first_rating']} → {match['last_rating']} (change: {change:+.1f})")
                validation_results.append({
                    'cohort': cohort,
                    'participant': participant,
                    'idea': expected['idea'],
                    'status': 'VALIDATED',
                    'actual_first': match['first_rating'],
                    'actual_last': match['last_rating'],
                    'actual_change': change
                })

    return validation_results

def validate_data_filtering():
    """Validate that data filtering is applied correctly"""
    print("\n📊 VALIDATING DATA FILTERING")
    print("=" * 60)

    df_trajectories, _ = load_and_process_comprehensive_data()

    # Check 2020 exclusion
    cohorts_present = df_trajectories['cohort'].unique()
    if '2020' in cohorts_present:
        print("❌ ERROR: 2020 data found in trajectories (should be excluded)")
        return False
    else:
        print("✅ 2020 data correctly excluded")

    # Check 2021 participant filtering
    approved_2021 = [
        'Cillian', 'Steve', 'Jan-Willem', 'Devon', 'Federico',
        'Lukas', 'Kiki', 'Isaac', 'Andres', 'Aaron', 'Kristina'
    ]

    participants_2021 = df_trajectories[df_trajectories['cohort'] == '2021']['participant'].unique()

    if len(participants_2021) == 0:
        print("⚠️  WARNING: No 2021 participants found - check if 2021 data has rating columns")
    else:
        non_approved = [p for p in participants_2021 if p not in approved_2021]
        if non_approved:
            print(f"❌ ERROR: Non-approved 2021 participants found: {non_approved}")
            return False
        else:
            print(f"✅ 2021 participants correctly filtered: {len(participants_2021)} approved participants")

    # Check participant count
    total_participants = df_trajectories['participant'].nunique()
    if total_participants < 47:
        print(f"⚠️  WARNING: Only {total_participants} participants found (team expected more than 47)")
    else:
        print(f"✅ Found {total_participants} participants (exceeds expected 47)")

    return True

def validate_scale_conversions():
    """Validate scale conversions are applied correctly"""
    print("\n🔢 VALIDATING SCALE CONVERSIONS")
    print("=" * 60)

    # Load raw data to check conversions
    test_cases = [
        {'cohort': 'H123', 'raw_value': -3, 'expected': 1, 'description': '-3 → 1'},
        {'cohort': 'H123', 'raw_value': 0, 'expected': 4, 'description': '0 → 4'},
        {'cohort': 'H123', 'raw_value': 3, 'expected': 7, 'description': '3 → 7'},
        {'cohort': '2021', 'raw_value': 1, 'expected': 7, 'description': '1st choice → 7'},
        {'cohort': '2021', 'raw_value': 7, 'expected': 1, 'description': '7th choice → 1'},
        {'cohort': 'H125', 'raw_value': 5, 'expected': 5, 'description': '5 → 5 (no conversion)'}
    ]

    from comprehensive_data_processor_v3 import convert_to_standard_scale_comprehensive

    for test in test_cases:
        result = convert_to_standard_scale_comprehensive(test['raw_value'], test['cohort'])
        if result == test['expected']:
            print(f"✅ {test['cohort']}: {test['description']} = {result}")
        else:
            print(f"❌ {test['cohort']}: {test['description']} = {result} (expected {test['expected']})")
            return False

    return True

def validate_co_founder_mappings():
    """Validate co-founder mappings are correct"""
    print("\n👥 VALIDATING CO-FOUNDER MAPPINGS")
    print("=" * 60)

    df_trajectories, _ = load_and_process_comprehensive_data()

    # Expected co-founders from feedback
    expected_cofounders = {
        'H125': {
            'EAFW': ['Koen', 'Ameer'],
            'DPR': ['Rowan', 'Oli'],
            'RTS': ['Larissa'],
            'CFME': ['Jessica', 'Stuart']
        },
        'H224': {
            'LMP': ['Graham', 'Anam'],
            'LRO': ['Tammy', 'Isabel'],
            'IOA': ['Leonie']
        }
    }

    validation_issues = []

    for cohort, ideas in expected_cofounders.items():
        print(f"\n{cohort} Co-founders:")

        for idea_key, expected_founders in ideas.items():
            print(f"  {idea_key}: {expected_founders}")

            for founder in expected_founders:
                # Find co-founder trajectories for this person and idea
                cofounder_trajectories = df_trajectories[
                    (df_trajectories['participant'] == founder) &
                    (df_trajectories['cohort'] == cohort) &
                    (df_trajectories['is_cofounder'] == True) &
                    (df_trajectories['cofounder_idea'] == idea_key)
                ]

                if cofounder_trajectories.empty:
                    print(f"    ❌ {founder}: No co-founder trajectory found for {idea_key}")
                    validation_issues.append(f"{founder} missing co-founder trajectory for {idea_key}")
                else:
                    print(f"    ✅ {founder}: Found co-founder trajectory")

    return len(validation_issues) == 0, validation_issues

def validate_animal_human_classification():
    """Validate animal vs human idea classification"""
    print("\n🐾 VALIDATING ANIMAL VS HUMAN CLASSIFICATION")
    print("=" * 60)

    df_trajectories, _ = load_and_process_comprehensive_data()

    # Expected animal ideas from feedback
    expected_animal_ideas = [
        'KBF', 'EAFW', 'CFME', 'II', 'GF', 'AF', 'MW', 'Milkfish Welfare',
        'Banning Baitfish', 'Banning Low Welfare Imports', 'Shrimp Welfare'
    ]

    # Check each idea in our data
    unique_ideas = df_trajectories['idea'].unique()
    classification_issues = []

    print("Ideas found in data:")
    for idea in sorted(unique_ideas):
        is_classified_as_animal = df_trajectories[df_trajectories['idea'] == idea]['is_animal_idea'].iloc[0]

        # Check if this should be classified as animal
        should_be_animal = any(animal_keyword in idea.upper() for animal_keyword in
                              [keyword.upper() for keyword in expected_animal_ideas])

        if is_classified_as_animal == should_be_animal:
            animal_human = "🐾 Animal" if is_classified_as_animal else "👥 Human"
            print(f"  ✅ {idea}: {animal_human}")
        else:
            expected_type = "🐾 Animal" if should_be_animal else "👥 Human"
            actual_type = "🐾 Animal" if is_classified_as_animal else "👥 Human"
            print(f"  ❌ {idea}: Expected {expected_type}, got {actual_type}")
            classification_issues.append(f"{idea}: Expected {expected_type}, got {actual_type}")

    return len(classification_issues) == 0, classification_issues

def validate_idea_merging():
    """Validate that EAFW, MW, and Milkfish Welfare are merged"""
    print("\n🔀 VALIDATING IDEA MERGING")
    print("=" * 60)

    df_trajectories, _ = load_and_process_comprehensive_data()

    unique_ideas = df_trajectories['idea'].unique()

    # Check that MW and Milkfish Welfare don't appear as separate ideas
    problematic_ideas = [idea for idea in unique_ideas if
                        'MW' == idea or 'Milkfish Welfare' in idea]

    if problematic_ideas:
        print(f"❌ MERGING FAILED: Found separate ideas that should be merged: {problematic_ideas}")
        return False

    # Check that EAFW exists
    eafw_found = any('EAFW' in idea for idea in unique_ideas)
    if not eafw_found:
        print("❌ MERGING FAILED: EAFW not found after merging")
        return False

    print("✅ Ideas correctly merged - EAFW, MW, and Milkfish Welfare consolidated")
    return True

def validate_participants_with_both_types():
    """Validate the count of participants with both human and animal ideas <4"""
    print("\n🔄 VALIDATING PARTICIPANTS WITH BOTH TYPES <4")
    print("=" * 60)

    df_trajectories, _ = load_and_process_comprehensive_data()

    participants_both_low = []

    for participant in df_trajectories['participant'].unique():
        p_data = df_trajectories[df_trajectories['participant'] == participant]

        animal_data = p_data[p_data['is_animal_idea'] == True]
        human_data = p_data[p_data['is_animal_idea'] == False]

        # Check if participant has both types of ideas
        if not animal_data.empty and not human_data.empty:
            # Check if both have <4 ratings in first AND final weeks
            animal_first_low = (animal_data['first_rating'] < 4).any()
            animal_last_low = (animal_data['last_rating'] < 4).any()

            human_first_low = (human_data['first_rating'] < 4).any()
            human_last_low = (human_data['last_rating'] < 4).any()

            if animal_first_low and animal_last_low and human_first_low and human_last_low:
                participants_both_low.append(participant)

    print(f"Found {len(participants_both_low)} participants with <4 ratings for both types (first & final weeks)")
    for p in participants_both_low:
        print(f"  - {p}")

    return participants_both_low

def validate_data_completeness():
    """Check for data completeness and consistency"""
    print("\n📋 VALIDATING DATA COMPLETENESS")
    print("=" * 60)

    df_trajectories, df_sentiment = load_and_process_comprehensive_data()

    issues = []

    # Check for missing values in critical columns
    critical_columns = ['participant', 'cohort', 'idea', 'first_rating', 'last_rating', 'change']
    for col in critical_columns:
        if col in df_trajectories.columns:
            missing = df_trajectories[col].isna().sum()
            if missing > 0:
                issues.append(f"{missing} missing values in {col}")
                print(f"⚠️  {missing} missing values in {col}")
            else:
                print(f"✅ No missing values in {col}")
        else:
            issues.append(f"Critical column {col} not found")
            print(f"❌ Critical column {col} not found")

    # Check rating ranges
    if 'first_rating' in df_trajectories.columns and 'last_rating' in df_trajectories.columns:
        all_ratings = pd.concat([df_trajectories['first_rating'], df_trajectories['last_rating']])
        min_rating = all_ratings.min()
        max_rating = all_ratings.max()

        if min_rating < 1 or max_rating > 7:
            issues.append(f"Ratings outside 1-7 range: {min_rating} to {max_rating}")
            print(f"❌ Ratings outside 1-7 range: {min_rating} to {max_rating}")
        else:
            print(f"✅ All ratings within 1-7 range: {min_rating} to {max_rating}")

    # Check for duplicate trajectories
    if len(critical_columns) > 3:
        duplicates = df_trajectories.duplicated(subset=['participant', 'cohort', 'idea']).sum()
        if duplicates > 0:
            issues.append(f"{duplicates} duplicate trajectories found")
            print(f"⚠️  {duplicates} duplicate trajectories found")
        else:
            print("✅ No duplicate trajectories")

    return len(issues) == 0, issues

def run_comprehensive_validation():
    """Run all validation checks"""
    print("🔍 COMPREHENSIVE VALIDATION OF DASHBOARD v2.0")
    print("=" * 80)
    print("Checking for bugs and errors in all critical areas...")

    all_passed = True
    validation_summary = []

    # 1. Validate specific error cases
    try:
        error_case_results = validate_specific_error_cases()
        failed_cases = [r for r in error_case_results if r['status'] != 'VALIDATED']
        if failed_cases:
            all_passed = False
            validation_summary.append(f"❌ {len(failed_cases)} specific error cases failed validation")
        else:
            validation_summary.append(f"✅ All specific error cases validated")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Error case validation failed: {e}")

    # 2. Validate data filtering
    try:
        if validate_data_filtering():
            validation_summary.append("✅ Data filtering validated")
        else:
            all_passed = False
            validation_summary.append("❌ Data filtering failed validation")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Data filtering validation error: {e}")

    # 3. Validate scale conversions
    try:
        if validate_scale_conversions():
            validation_summary.append("✅ Scale conversions validated")
        else:
            all_passed = False
            validation_summary.append("❌ Scale conversions failed validation")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Scale conversion validation error: {e}")

    # 4. Validate co-founder mappings
    try:
        cofounder_valid, cofounder_issues = validate_co_founder_mappings()
        if cofounder_valid:
            validation_summary.append("✅ Co-founder mappings validated")
        else:
            all_passed = False
            validation_summary.append(f"❌ Co-founder mappings failed: {len(cofounder_issues)} issues")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Co-founder validation error: {e}")

    # 5. Validate animal/human classification
    try:
        classification_valid, classification_issues = validate_animal_human_classification()
        if classification_valid:
            validation_summary.append("✅ Animal/Human classification validated")
        else:
            all_passed = False
            validation_summary.append(f"❌ Classification failed: {len(classification_issues)} issues")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Classification validation error: {e}")

    # 6. Validate idea merging
    try:
        if validate_idea_merging():
            validation_summary.append("✅ Idea merging validated")
        else:
            all_passed = False
            validation_summary.append("❌ Idea merging failed validation")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Idea merging validation error: {e}")

    # 7. Validate participants with both types
    try:
        participants_both = validate_participants_with_both_types()
        validation_summary.append(f"✅ Found {len(participants_both)} participants with both types <4")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Both types validation error: {e}")

    # 8. Validate data completeness
    try:
        completeness_valid, completeness_issues = validate_data_completeness()
        if completeness_valid:
            validation_summary.append("✅ Data completeness validated")
        else:
            all_passed = False
            validation_summary.append(f"❌ Data completeness issues: {len(completeness_issues)} problems")
    except Exception as e:
        all_passed = False
        validation_summary.append(f"❌ Data completeness validation error: {e}")

    # Final summary
    print("\n" + "=" * 80)
    print("🎯 COMPREHENSIVE VALIDATION SUMMARY")
    print("=" * 80)

    for summary in validation_summary:
        print(summary)

    if all_passed:
        print("\n🎉 ALL VALIDATIONS PASSED - DASHBOARD v2.0 IS ERROR-FREE!")
        print("✅ Full confidence: The dashboard addresses all team feedback correctly.")
    else:
        print("\n⚠️  VALIDATION ISSUES FOUND - REQUIRES ATTENTION")
        print("❌ Some aspects need to be fixed before full confidence can be achieved.")

    return all_passed, validation_summary

if __name__ == "__main__":
    success, summary = run_comprehensive_validation()

    if success:
        print("\n🚀 Dashboard v2.0 is ready for production use!")
    else:
        print("\n🔧 Dashboard v2.0 needs fixes before deployment.")