#!/usr/bin/env python3
"""
TARGETED BUG FIXES for Dashboard v2.0
Addresses specific validation failures found in comprehensive validation
"""

import pandas as pd
import numpy as np

def investigate_specific_error_cases():
    """Deep dive into the specific error cases to understand the discrepancies"""
    print("🔍 INVESTIGATING SPECIFIC ERROR CASE DISCREPANCIES")
    print("=" * 70)

    data_path = "Idea Interest Over Time Data for Elizabeth.xlsx"

    # Cases that failed validation
    problem_cases = [
        {'cohort': 'H125', 'participant': 'Amy', 'idea': 'CFME', 'expected_change': 5},
        {'cohort': 'H224', 'participant': 'Uttej', 'idea': 'PLA', 'expected_change': 2},
        {'cohort': 'H123', 'participant': 'Miri', 'idea': 'Tobacco Tax', 'expected_change': 3},
        {'cohort': 'H123', 'participant': 'Nicoll', 'idea': 'Kangaroo Care', 'expected_change': 2},
        {'cohort': 'H123', 'participant': 'Victoria', 'idea': 'Tobacco Tax', 'expected_change': 3}
    ]

    for case in problem_cases:
        print(f"\n--- INVESTIGATING {case['participant']} - {case['idea']} ({case['cohort']}) ---")

        try:
            # Load raw data
            df = pd.read_excel(data_path, sheet_name=case['cohort'])

            # Find name column
            name_col = None
            for col in df.columns:
                if 'name' in col.lower():
                    name_col = col
                    break

            if not name_col:
                print("❌ Name column not found")
                continue

            # Find participant data
            p_data = df[df[name_col] == case['participant']]
            if p_data.empty:
                print(f"❌ Participant {case['participant']} not found")
                continue

            print(f"✅ Found {len(p_data)} rows for {case['participant']}")

            # Find stage column
            stage_col = None
            for col in df.columns:
                if 'stage' in col.lower() or 'submission' in col.lower():
                    stage_col = col
                    break

            # Find idea column (more thorough search)
            idea_col = find_idea_column_thorough(df, case['idea'], case['cohort'])

            if not idea_col:
                print(f"❌ Could not find column for {case['idea']}")
                print(f"Available columns: {[col for col in df.columns if 'interest' in col.lower() or 'idea' in col.lower()]}")
                continue

            print(f"✅ Using column: {idea_col}")

            # Show ALL ratings for this participant-idea
            print("Raw data progression:")
            week_ratings = {}

            for _, row in p_data.iterrows():
                if pd.notna(row[idea_col]):
                    week_info = str(row[stage_col]) if stage_col and pd.notna(row[stage_col]) else f"Row {row.name}"
                    raw_rating = row[idea_col]
                    converted_rating = convert_rating_fixed(raw_rating, case['cohort'])

                    print(f"  {week_info}: Raw={raw_rating}, Converted={converted_rating}")

                    # Extract week number for trajectory calculation
                    import re
                    week_match = re.search(r'week\s*(\d+)', str(week_info).lower())
                    if week_match:
                        week_num = int(week_match.group(1))
                        week_ratings[week_num] = converted_rating

            # Calculate trajectory the correct way
            if len(week_ratings) >= 2:
                sorted_weeks = sorted(week_ratings.keys())
                first_week = sorted_weeks[0]
                last_week = sorted_weeks[-1]

                first_rating = week_ratings[first_week]
                last_rating = week_ratings[last_week]
                actual_change = last_rating - first_rating

                print(f"TRAJECTORY: Week {first_week} ({first_rating}) → Week {last_week} ({last_rating}) = {actual_change:+.0f}")
                print(f"EXPECTED: {case['expected_change']:+.0f}")

                if abs(actual_change - case['expected_change']) > 0.1:
                    print(f"❌ MISMATCH: Expected {case['expected_change']}, got {actual_change}")
                else:
                    print("✅ MATCHES expectation")

        except Exception as e:
            print(f"❌ Error investigating {case['participant']}: {e}")

def find_idea_column_thorough(df, idea_name, cohort):
    """More thorough search for idea columns"""
    # Create comprehensive search patterns
    search_patterns = {
        'CFME': ['cfme', 'cage-free', 'cage free', 'middle east'],
        'PLA': ['pla', 'groups'],
        'Tobacco Tax': ['tobacco', 'taxation'],
        'Kangaroo Care': ['kangaroo', 'care', 'kc']
    }

    patterns = search_patterns.get(idea_name, [idea_name.lower()])

    # Search in all columns
    for col in df.columns:
        col_lower = str(col).lower()
        if any(pattern in col_lower for pattern in patterns):
            return col

    return None

def convert_rating_fixed(value, cohort):
    """Fixed conversion function"""
    if pd.isna(value):
        return None

    try:
        # Handle string values with parentheses
        if isinstance(value, str) and '(' in value:
            value = value.split('(')[0].strip()

        value = float(value)

        if cohort == 'H123':
            # -3 to +3 scale → 1-7 scale
            if -3 <= value <= 3:
                return int(value + 4)
        elif cohort in ['2021', '2022']:
            # Ranking system: 1st = 7, 2nd = 6, etc.
            if 1 <= value <= 7:
                return int(8 - value)
        else:
            # Standard 1-7 scale
            if 1 <= value <= 7:
                return int(value)

        return None

    except (ValueError, TypeError):
        return None

def fix_animal_human_classification():
    """Fix the animal vs human classification issues"""
    print("\n🐾 FIXING ANIMAL VS HUMAN CLASSIFICATION")
    print("=" * 50)

    # The validation found these issues:
    # - "Banning Baitfish": Expected Animal, got Human
    # - "Banning Low Welfare Imports": Expected Animal, got Human

    print("Issues found:")
    print("  - 'Banning Baitfish' should be classified as Animal (not Human)")
    print("  - 'Banning Low Welfare Imports' should be classified as Animal (not Human)")

    # These are in the team's animal list, so need to update the classification logic
    return True

def check_2021_2022_data_structure():
    """Check why 2021/2022 have no rating columns"""
    print("\n📊 INVESTIGATING 2021/2022 DATA STRUCTURE")
    print("=" * 50)

    data_path = "Idea Interest Over Time Data for Elizabeth.xlsx"

    for cohort in ['2021', '2022']:
        try:
            df = pd.read_excel(data_path, sheet_name=cohort)
            print(f"\n{cohort} - {len(df)} rows, {len(df.columns)} columns")
            print("Column names:")
            for i, col in enumerate(df.columns):
                print(f"  {i+1}. {col}")

            # Look for any columns that might contain ratings
            rating_like_columns = []
            for col in df.columns:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['rank', 'choice', 'preference', 'priority', 'rating', 'score']):
                    rating_like_columns.append(col)

            print(f"Rating-like columns: {rating_like_columns}")

        except Exception as e:
            print(f"Error loading {cohort}: {e}")

def create_comprehensive_fix():
    """Create comprehensive fix for all identified issues"""
    print("\n🔧 CREATING COMPREHENSIVE FIX")
    print("=" * 50)

    fixes_needed = [
        "1. Trajectory calculation logic - need to handle all weeks correctly",
        "2. Animal classification - add 'Banning Baitfish' and 'Banning Low Welfare Imports' to animal list",
        "3. 2021/2022 data handling - may need different column search patterns",
        "4. Week extraction - handle Week 3.5 and other edge cases",
        "5. Rating conversion - ensure H123 scale conversion is applied correctly"
    ]

    for fix in fixes_needed:
        print(f"  {fix}")

    return fixes_needed

if __name__ == "__main__":
    print("🔧 TARGETED BUG FIX INVESTIGATION")
    print("=" * 70)

    investigate_specific_error_cases()
    fix_animal_human_classification()
    check_2021_2022_data_structure()
    fixes = create_comprehensive_fix()

    print(f"\n✅ Investigation complete - {len(fixes)} fixes identified")