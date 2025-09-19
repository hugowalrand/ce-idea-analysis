#!/usr/bin/env python3
"""
Debug peak calculations to understand trajectory discrepancies
"""

import pandas as pd

def debug_specific_cases():
    """Debug the specific cases to see what peaks we're finding"""

    cases = [
        {
            'participant': 'Amy',
            'idea': 'CFME',
            'cohort': 'H125',
            'expected_trajectory': 'Week 1 (1) → Week 2 (6) = +5',
            'weeks': ['Week 1', 'Week 2', 'Week 3', 'Week 3.5', 'Week 4', 'Week 5'],
            'ratings': [1, 6, 4, 3, 1, 2]
        },
        {
            'participant': 'Miri',
            'idea': 'Tobacco Tax',
            'cohort': 'H123',
            'expected_trajectory': 'Week 1 (2) → Week 2 (5) = +3',
            'weeks': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
            'ratings': [2, 5, 2, 3, 4]  # H123 converted
        },
        {
            'participant': 'Nicoll',
            'idea': 'Kangaroo Care',
            'cohort': 'H123',
            'expected_trajectory': 'Week 1 (3) → Week 4 (5) = +2',
            'weeks': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
            'ratings': [3, 3, 2, 5, 2]  # H123 converted
        },
        {
            'participant': 'Victoria',
            'idea': 'Tobacco Tax',
            'cohort': 'H123',
            'expected_trajectory': 'Week 1 (2) → Week 3 (5) = +3',
            'weeks': ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5'],
            'ratings': [2, 4, 5, 2, 1]  # H123 converted
        }
    ]

    print("🔍 DEBUGGING PEAK CALCULATIONS")
    print("=" * 50)

    for case in cases:
        print(f"\n--- {case['participant']} - {case['idea']} ({case['cohort']}) ---")
        print(f"Expected: {case['expected_trajectory']}")
        print(f"Data progression:")

        # Show week-by-week data
        week_data = {}
        for i, (week, rating) in enumerate(zip(case['weeks'], case['ratings'])):
            week_data[week] = rating
            print(f"  {week}: {rating}")

        # Apply my peak-finding algorithm
        sorted_weeks = sorted(week_data.keys(), key=lambda x: float(str(x).replace('Week ', '').replace('week ', '')))

        first_week = sorted_weeks[0]
        first_rating = week_data[first_week]

        # Find peak
        peak_rating = max([week_data[w] for w in sorted_weeks])
        peak_week = None
        for w in sorted_weeks:
            if week_data[w] == peak_rating:
                peak_week = w
                break

        change_first_to_peak = peak_rating - first_rating

        # Also calculate final week
        last_week = sorted_weeks[-1]
        last_rating = week_data[last_week]
        change_first_to_last = last_rating - first_rating

        print(f"My algorithm:")
        print(f"  First week: {first_week} ({first_rating})")
        print(f"  Peak week: {peak_week} ({peak_rating})")
        print(f"  Last week: {last_week} ({last_rating})")
        print(f"  First→Peak: {first_rating} → {peak_rating} = {change_first_to_peak:+}")
        print(f"  First→Last: {first_rating} → {last_rating} = {change_first_to_last:+}")

        # Compare with team expectation
        team_parts = case['expected_trajectory'].split(' = ')
        team_change = int(team_parts[1])

        if change_first_to_peak == team_change:
            print(f"  ✅ MATCHES: First→Peak ({change_first_to_peak:+}) = Team expectation ({team_change:+})")
        elif change_first_to_last == team_change:
            print(f"  ⚠️  Team expects First→Last ({change_first_to_last:+}), not First→Peak ({change_first_to_peak:+})")
        else:
            print(f"  ❌ MISMATCH: Team expects {team_change:+}, got First→Peak {change_first_to_peak:+}, First→Last {change_first_to_last:+}")

if __name__ == "__main__":
    debug_specific_cases()