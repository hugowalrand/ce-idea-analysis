#!/usr/bin/env python3
"""
Check what the actual processed data contains for our problem cases
"""

from comprehensive_data_processor_v3 import load_and_process_comprehensive_data

def check_specific_trajectories():
    """Check the exact trajectories being produced for our problem cases"""
    print("🔍 CHECKING ACTUAL PROCESSED DATA")
    print("=" * 50)

    df_trajectories, _ = load_and_process_comprehensive_data()

    problem_cases = [
        {'participant': 'Amy', 'cohort': 'H125', 'idea': 'CFME'},
        {'participant': 'Uttej', 'cohort': 'H224', 'idea': 'PLA'},
        {'participant': 'Miri', 'cohort': 'H123', 'idea': 'Tobacco Tax'},
        {'participant': 'Nicoll', 'cohort': 'H123', 'idea': 'Kangaroo Care'},
        {'participant': 'Victoria', 'cohort': 'H123', 'idea': 'Tobacco Tax'}
    ]

    for case in problem_cases:
        print(f"\n--- {case['participant']} - {case['idea']} ({case['cohort']}) ---")

        # Find exact matches
        exact_matches = df_trajectories[
            (df_trajectories['participant'] == case['participant']) &
            (df_trajectories['cohort'] == case['cohort']) &
            (df_trajectories['idea'].str.contains(case['idea'], case=False, na=False))
        ]

        if exact_matches.empty:
            print("❌ No exact matches found")

            # Try broader search
            participant_matches = df_trajectories[
                (df_trajectories['participant'] == case['participant']) &
                (df_trajectories['cohort'] == case['cohort'])
            ]

            if not participant_matches.empty:
                print(f"Found {len(participant_matches)} trajectories for this participant:")
                for _, row in participant_matches.iterrows():
                    print(f"  - {row['idea']}: {row['first_rating']} → {row['last_rating']} (change: {row['change']:+})")
            else:
                print("❌ No trajectories found for this participant at all")
        else:
            print(f"✅ Found {len(exact_matches)} matching trajectories:")
            for _, row in exact_matches.iterrows():
                print(f"  Idea: {row['idea']}")
                print(f"  First rating: {row['first_rating']}")
                if 'peak_rating' in row:
                    print(f"  Peak rating: {row['peak_rating']}")
                print(f"  Last rating: {row['last_rating']}")
                print(f"  Change: {row['change']:+}")
                print(f"  All weeks: {row['all_weeks']}")
                print(f"  All ratings: {row['all_ratings']}")

if __name__ == "__main__":
    check_specific_trajectories()