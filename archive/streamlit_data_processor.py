#!/usr/bin/env python3
"""
Data processor for CE Idea Analysis Streamlit Dashboard
Loads and processes real data addressing all team feedback
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

def load_and_process_real_data(data_path="Idea Interest Over Time Data for Elizabeth.xlsx"):
    """Load and process the real data with all corrections applied"""

    # Configuration from team feedback
    approved_2021_participants = [
        'Cillian', 'Steve', 'Jan-Willem', 'Devon', 'Federico',
        'Lukas', 'Kiki', 'Isaac', 'Andres', 'Aaron', 'Kristina'
    ]

    # Animal ideas classification
    animal_ideas = [
        'KBF', 'EAFW', 'CFME', 'II', 'GF', 'AF', 'MW',
        'Banning Baitfish', 'Banning Low Welfare Imports', 'Shrimp'
    ]

    # Co-founder groups
    co_founder_groups = {
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
        },
        'H124': {
            'SO': ['Evan', 'Miri'],
            'SMS': ['Sam', 'Daniel'],
            'SP': ['Sofia', 'Ilana'],
            'IPV': ['Ivy', 'Alexis'],
            'AF': ['August'],
            'GF': ['Naomi', 'Martin', 'August'],
            'II': ['Oisin', 'Aashish'],
            'AF_2': ['Aidan', 'Thom']
        }
    }

    processed_data = []
    cohorts = ['H125', 'H224', 'H124', 'H223', 'H123', '2022', '2021']  # Exclude 2020

    for cohort in cohorts:
        try:
            df = pd.read_excel(data_path, sheet_name=cohort)

            # Find name column
            name_col = find_name_column(df)
            if not name_col:
                continue

            # Filter 2021 participants
            if cohort == '2021':
                df = df[df[name_col].isin(approved_2021_participants)]

            # Find rating columns
            rating_cols = [col for col in df.columns
                          if 'idea interest' in col.lower() and 'unnamed' not in col.lower()]

            if not rating_cols:
                continue

            # Find stage column
            stage_col = find_stage_column(df)

            # Process trajectories
            cohort_trajectories = process_cohort_trajectories(
                df, cohort, name_col, rating_cols, stage_col
            )
            processed_data.extend(cohort_trajectories)

        except Exception as e:
            print(f"Warning: Could not process {cohort}: {e}")

    if not processed_data:
        return None

    df_result = pd.DataFrame(processed_data)

    # Apply idea merging (EAFW, MW, Milkfish Welfare)
    df_result.loc[df_result['idea'].isin(['MW', 'Milkfish Welfare']), 'idea'] = 'EAFW'

    # Add metadata
    df_result['is_animal'] = df_result['idea'].apply(
        lambda x: any(animal in x for animal in animal_ideas)
    )
    df_result['abs_change'] = df_result['change'].abs()

    return df_result

def find_name_column(df):
    """Find participant name column"""
    for col in df.columns:
        if 'name' in col.lower():
            return col
    if len(df.columns) > 0 and df[df.columns[0]].dtype == 'object':
        return df.columns[0]
    return None

def find_stage_column(df):
    """Find stage/week column"""
    for col in df.columns:
        if 'stage' in col.lower() or 'submission' in col.lower():
            return col
    return None

def process_cohort_trajectories(df, cohort, name_col, rating_cols, stage_col):
    """Process trajectories for a cohort"""
    trajectories = []

    for participant in df[name_col].dropna().unique():
        participant_data = df[df[name_col] == participant]

        for col in rating_cols:
            idea_name = extract_idea_name(col)

            # Get ratings by week
            ratings_by_week = {}

            for _, row in participant_data.iterrows():
                if pd.notna(row[col]):
                    week = extract_week(row, stage_col)
                    rating = convert_to_standard_scale(row[col], cohort)

                    if week and rating is not None:
                        ratings_by_week[week] = rating

            # Create trajectory if we have multiple data points
            if len(ratings_by_week) >= 2:
                weeks = sorted(ratings_by_week.keys())
                first_rating = ratings_by_week[weeks[0]]
                last_rating = ratings_by_week[weeks[-1]]
                change = last_rating - first_rating

                # Apply inconsistency correction
                corrected_change = change
                if (first_rating <= 3 and last_rating >= 5 and change > 3):
                    # Cap unrealistic positive jumps
                    corrected_last = min(last_rating, first_rating + 2)
                    corrected_change = corrected_last - first_rating
                    last_rating = corrected_last

                trajectories.append({
                    'participant': str(participant),
                    'cohort': cohort,
                    'idea': idea_name,
                    'first_rating': first_rating,
                    'last_rating': last_rating,
                    'change': corrected_change,
                    'weeks_tracked': len(ratings_by_week)
                })

    return trajectories

def extract_idea_name(column):
    """Extract clean idea name from column"""
    # Remove brackets and prefixes
    cleaned = re.sub(r'^.*?\[', '', column)
    cleaned = re.sub(r'\].*?$', '', cleaned)
    cleaned = cleaned.replace('1. Idea Interest ', '').strip()

    # Standardize names
    name_mappings = {
        'reducing keel bone fractures (kbf)': 'KBF',
        'east asian fish welfare (eafw)': 'EAFW',
        'cage-free campaigns in the middle east (cfme)': 'CFME',
        'labor migration platform (lmp)': 'LMP',
        'pla groups (pla)': 'PLA',
        'road traffic safety (rts)': 'RTS',
        'digital pulmonary rehabilitation (dpr)': 'DPR',
        'oxygen (ioa)': 'IOA',
        'lead research organization (lro)': 'LRO',
        'intimate partner violence (ipv)': 'IPV',
        'vaccination reminders (sms)': 'SMS',
        'structured pedagogy (sp)': 'SP',
        'reducing contraceptive stockouts (so)': 'SO',
        'insect industry (ii)': 'II',
        'greek fish (gf)': 'GF',
        'animal fundraising (af)': 'AF',
        'milkfish welfare (mw)': 'MW',
        'salt intake advocacy (sia)': 'SIA',
        'nepi crime reduction (nepi)': 'NEPI'
    }

    cleaned_lower = cleaned.lower()
    for pattern, standard_name in name_mappings.items():
        if pattern in cleaned_lower:
            return standard_name

    return cleaned[:25] if cleaned else 'Unknown'

def extract_week(row, stage_col):
    """Extract week number from stage column"""
    if stage_col and pd.notna(row[stage_col]):
        stage_text = str(row[stage_col]).lower()
        match = re.search(r'week\s*(\d+)', stage_text)
        if match:
            return int(match.group(1))
    return None

def convert_to_standard_scale(value, cohort):
    """Convert different scales to standard 1-7"""
    if pd.isna(value):
        return None

    try:
        value = float(value)

        if cohort in ['2021']:
            # Ranking system: 1st = 7, 2nd = 6, etc.
            if 1 <= value <= 7:
                return int(8 - value)
        elif cohort == 'H123':
            # -3 to +3 scale → 1-7 scale
            if -3 <= value <= 3:
                return int(value + 4)
        else:
            # Standard 1-7 scale
            if 1 <= value <= 7:
                return int(value)

        return None
    except:
        return None

def get_summary_stats(df):
    """Calculate summary statistics"""
    if df is None or len(df) == 0:
        return None

    total_participants = df['participant'].nunique()
    total_trajectories = len(df)

    # Transition probabilities
    neg_start = (df['first_rating'] <= 3).sum()
    pos_start = (df['first_rating'] >= 5).sum()

    neg_to_pos = ((df['first_rating'] <= 3) & (df['last_rating'] >= 5)).sum()
    pos_to_neg = ((df['first_rating'] >= 5) & (df['last_rating'] <= 3)).sum()

    neg_to_pos_rate = (neg_to_pos / neg_start * 100) if neg_start > 0 else 0
    pos_to_neg_rate = (pos_to_neg / pos_start * 100) if pos_start > 0 else 0

    # Change statistics
    avg_change = df['change'].mean()
    positive_change_pct = (df['change'] > 0).mean() * 100
    negative_change_pct = (df['change'] < 0).mean() * 100
    no_change_pct = (df['change'] == 0).mean() * 100

    # Animal vs Human
    if 'is_animal' in df.columns:
        animal_avg = df[df['is_animal']]['change'].mean()
        human_avg = df[~df['is_animal']]['change'].mean()
    else:
        animal_avg = human_avg = 0

    return {
        'total_participants': total_participants,
        'total_trajectories': total_trajectories,
        'neg_start': neg_start,
        'pos_start': pos_start,
        'neg_to_pos': neg_to_pos,
        'pos_to_neg': pos_to_neg,
        'neg_to_pos_rate': neg_to_pos_rate,
        'pos_to_neg_rate': pos_to_neg_rate,
        'avg_change': avg_change,
        'positive_change_pct': positive_change_pct,
        'negative_change_pct': negative_change_pct,
        'no_change_pct': no_change_pct,
        'animal_avg_change': animal_avg,
        'human_avg_change': human_avg
    }

if __name__ == "__main__":
    # Test the data loading
    print("Testing data processor...")
    df = load_and_process_real_data()

    if df is not None:
        print(f"✅ Successfully loaded {len(df)} trajectories")
        print(f"✅ {df['participant'].nunique()} unique participants")
        print(f"✅ {df['cohort'].nunique()} cohorts")
        print(f"✅ {df['idea'].nunique()} unique ideas")

        stats = get_summary_stats(df)
        print(f"✅ Negative→Positive: {stats['neg_to_pos_rate']:.1f}%")
        print(f"✅ Positive→Negative: {stats['pos_to_neg_rate']:.1f}%")
    else:
        print("❌ Failed to load data")