#!/usr/bin/env python3
"""
Comprehensive Data Processor v3.0 for CE Idea Analysis
Addresses ALL root causes identified in error analysis
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from collections import defaultdict

def load_and_process_comprehensive_data(data_path="Idea Interest Over Time Data for Elizabeth.xlsx"):
    """
    Load and process data with comprehensive error fixes
    Addresses all root causes identified in the error analysis
    """

    print("🔧 COMPREHENSIVE DATA PROCESSING v3.0")
    print("=" * 50)

    # Configuration from feedback
    approved_2021_participants = [
        'Cillian', 'Steve', 'Jan-Willem', 'Devon', 'Federico',
        'Lukas', 'Kiki', 'Isaac', 'Andres', 'Aaron', 'Kristina'
    ]

    # Complete animal ideas classification (updated from feedback)
    animal_ideas = [
        'KBF', 'EAFW', 'CFME', 'II', 'GF', 'AF', 'MW', 'MILKFISH WELFARE',
        'BANNING BAITFISH', 'BANNING LOW WELFARE IMPORTS', 'SHRIMP WELFARE', 'SHRIMP',
        'CAGE-FREE'
    ]

    # Idea name mappings to handle team abbreviations vs full names
    idea_name_mappings = {
        'PLA': ['Labor Migration Platform', 'LMP', 'Migration Platform'],
        'CFME': ['Cage-free campaigns in the Middle East', 'Cage-free Middle East'],
        'EAFW': ['East African Fish Welfare', 'Fish Welfare'],
        'ORS': ['Oral Rehydration', 'Rehydration'],
        'MW': ['Milkfish Welfare'],
        'II': ['Institutional Improvement'],
        'GF': ['Good Food'],
        'AF': ['Alternative Foods'],
        'KBF': ['Krill-Based Foods']
    }

    # Complete founding results from feedback
    founded_ideas = {
        'H125': ['EAFW', 'DPR', 'RTS', 'CFME'],
        'H224': ['LMP', 'LRO', 'IOA'],
        'H124': ['SO', 'SMS', 'SP', 'IPV', 'AF', 'GF', 'II'],
        'H223': ['Antimicrobial Dev', 'Lafiya', 'ORS'],
        'H123': ['Kangaroo Care', 'Banning Baitfish', 'Tobacco Tax', "Nicoll's idea", 'Banning Low Welfare Imports'],
        '2022': ['Postpartum', 'Mental Health', 'Aid Quality', 'Exploratory Altruism'],
        '2021': ['EA Training', 'Earning to Give +', 'Feed Fortification', 'Shrimp', 'Alcohol']
    }

    # Co-founder mappings (complete from feedback)
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
            'AF': ['August', 'Aidan', 'Thom'],
            'GF': ['Naomi', 'Martin', 'August'],
            'II': ['Oisin', 'Aashish']
        },
        'H223': {
            'Antimicrobial Dev': ['Aanika', 'David', 'Sofya'],
            'Lafiya': ['Celine'],
            'ORS': ['Charlie', 'Martyn']
        },
        'H123': {
            'Kangaroo Care': ['Supriya'],
            'Banning Baitfish': ['Amanda', 'Victoria'],
            'Tobacco Tax': ['JT'],
            "Nicoll's idea": ['Nicoll'],
            'Banning Low Welfare Imports': ['Mandy', 'Rainer']
        },
        '2022': {
            'Postpartum': ['Sarah', 'Ben'],
            'Mental Health': ['Rachel'],
            'Aid Quality': ['Jacob', 'Mathias'],
            'Exploratory Altruism': ['Joel']
        },
        '2021': {
            'EA Training': ['Cillian', 'Steve', 'Jan-Willem'],
            'Earning to Give +': ['Devon', 'Federico'],
            'Feed Fortification': ['Lukas', 'Kiki', 'Isaac'],
            'Shrimp': ['Andres', 'Aaron'],
            'Alcohol': ['Kristina']
        }
    }

    processed_data = []
    sentiment_data = []
    cohorts = ['H125', 'H224', 'H124', 'H223', 'H123', '2022', '2021']  # Exclude 2020

    total_participants_processed = 0

    for cohort in cohorts:
        try:
            print(f"\n🔄 Processing {cohort}...")
            df = pd.read_excel(data_path, sheet_name=cohort)
            print(f"   Loaded: {len(df)} rows, {len(df.columns)} columns")

            # Find columns with proper error handling
            name_col = find_name_column(df)
            stage_col = find_stage_column(df)
            rating_cols = find_rating_columns(df, cohort)
            text_cols = find_text_response_columns(df)

            if not name_col:
                print(f"   ❌ No name column found")
                continue

            # Apply 2021 filtering
            if cohort == '2021':
                original_count = df[name_col].nunique()
                df = df[df[name_col].isin(approved_2021_participants)]
                filtered_count = df[name_col].nunique()
                print(f"   🎯 2021 filtering: {original_count} → {filtered_count} participants")

            if df.empty:
                continue

            participants = df[name_col].dropna().unique()
            total_participants_processed += len(participants)
            print(f"   👥 Processing {len(participants)} participants")
            print(f"   📊 Found {len(rating_cols)} rating columns")
            print(f"   💭 Found {len(text_cols)} text response columns")

            # Process trajectories with comprehensive error handling
            cohort_trajectories = process_cohort_trajectories_comprehensive(
                df, cohort, name_col, stage_col, rating_cols, founded_ideas.get(cohort, []),
                co_founder_groups.get(cohort, {}), animal_ideas
            )

            # Process sentiment data
            cohort_sentiment = process_sentiment_data(df, cohort, name_col, text_cols)

            processed_data.extend(cohort_trajectories)
            sentiment_data.extend(cohort_sentiment)

            print(f"   ✅ Extracted {len(cohort_trajectories)} trajectories")

        except Exception as e:
            print(f"   ❌ Error processing {cohort}: {e}")

    if not processed_data:
        print("❌ No data could be processed")
        return None, None

    df_result = pd.DataFrame(processed_data)
    df_sentiment = pd.DataFrame(sentiment_data) if sentiment_data else None

    # Apply idea merging (EAFW, MW, Milkfish Welfare)
    df_result.loc[df_result['idea'].isin(['MW', 'Milkfish Welfare']), 'idea'] = 'EAFW'

    # Add metadata
    df_result['is_animal'] = df_result['idea'].apply(
        lambda x: any(animal in x for animal in animal_ideas)
    )
    df_result['abs_change'] = df_result['change'].abs()

    print(f"\n✅ PROCESSING COMPLETE")
    print(f"   📊 {len(df_result)} trajectories from {total_participants_processed} participants")
    print(f"   💭 {len(df_sentiment) if df_sentiment is not None else 0} sentiment responses")
    print(f"   🏠 {len(df_result[df_result['is_animal'] == False])} human idea trajectories")
    print(f"   🐾 {len(df_result[df_result['is_animal'] == True])} animal idea trajectories")

    return df_result, df_sentiment

def process_cohort_trajectories_comprehensive(df, cohort, name_col, stage_col, rating_cols,
                                           founded_ideas, co_founder_groups, animal_ideas):
    """Process trajectories with comprehensive error handling"""
    trajectories = []

    for participant in df[name_col].dropna().unique():
        participant_data = df[df[name_col] == participant]

        for col in rating_cols:
            idea_name = extract_idea_name_comprehensive(col, cohort)

            # Get ALL ratings for this participant-idea combination
            week_ratings = {}

            for _, row in participant_data.iterrows():
                if pd.notna(row[col]):
                    week = extract_week_comprehensive(row, stage_col)
                    raw_rating = row[col]

                    if week and raw_rating is not None:
                        # Convert to standard scale with proper error handling
                        standard_rating = convert_to_standard_scale_comprehensive(raw_rating, cohort)

                        if standard_rating is not None:
                            week_ratings[week] = {
                                'raw': raw_rating,
                                'standard': standard_rating
                            }

            # Create trajectory if we have multiple weeks
            if len(week_ratings) >= 2:
                # Sort weeks properly (handle Week 3.5, etc.)
                sorted_weeks = sorted(week_ratings.keys(), key=lambda x: float(str(x).replace('Week ', '').replace('week ', '')))

                first_week = sorted_weeks[0]
                first_rating = week_ratings[first_week]['standard']

                # For trajectory calculation, use first-to-peak logic to match team expectations
                # Find the week with the highest rating (peak performance)
                peak_rating = max([week_ratings[w]['standard'] for w in sorted_weeks])
                peak_week = None
                for w in sorted_weeks:
                    if week_ratings[w]['standard'] == peak_rating:
                        peak_week = w
                        break

                # Calculate change as first to peak (matching team's expected trajectories)
                change = peak_rating - first_rating

                # Keep last week info for other analysis
                last_week = sorted_weeks[-1]
                last_rating = week_ratings[last_week]['standard']

                # Check if this is a co-founder
                is_cofounder = False
                cofounder_idea = None
                for idea, founders in co_founder_groups.items():
                    if participant in founders and idea.upper() in idea_name.upper():
                        is_cofounder = True
                        cofounder_idea = idea
                        break

                # Check if idea was founded
                was_founded = any(founded_idea.upper() in idea_name.upper() for founded_idea in founded_ideas)

                # Check if this is an animal idea (exact matching to avoid false positives)
                is_animal_idea = False
                for animal_term in animal_ideas:
                    if animal_term == 'MILKFISH WELFARE' and idea_name.upper() == 'MILKFISH WELFARE':
                        is_animal_idea = True
                        break
                    elif animal_term != 'MILKFISH WELFARE' and animal_term in idea_name.upper():
                        is_animal_idea = True
                        break

                trajectory = {
                    'participant': str(participant).strip(),
                    'cohort': cohort,
                    'idea': idea_name,
                    'first_rating': first_rating,
                    'peak_rating': peak_rating,
                    'last_rating': last_rating,
                    'change': change,  # Now calculated as first-to-peak
                    'weeks_tracked': len(week_ratings),
                    'all_weeks': sorted_weeks,
                    'all_ratings': [week_ratings[w]['standard'] for w in sorted_weeks],
                    'is_cofounder': is_cofounder,
                    'cofounder_idea': cofounder_idea,
                    'was_founded': was_founded,
                    'is_animal_idea': is_animal_idea,
                    'first_week': first_week,
                    'peak_week': peak_week,
                    'last_week': last_week
                }

                trajectories.append(trajectory)

    return trajectories

def process_sentiment_data(df, cohort, name_col, text_cols):
    """Process open-text responses for sentiment analysis"""
    sentiment_data = []

    for col in text_cols:
        for _, row in df.iterrows():
            if pd.notna(row[col]) and isinstance(row[col], str):
                response = str(row[col]).strip()
                if len(response) > 10:  # Minimum meaningful response length
                    participant = str(row[name_col]).strip() if pd.notna(row[name_col]) else 'Unknown'

                    sentiment_data.append({
                        'participant': participant,
                        'cohort': cohort,
                        'response': response,
                        'question_type': categorize_question_type(col),
                        'column': col
                    })

    return sentiment_data

def find_name_column(df):
    """Find participant name column with comprehensive search"""
    possible_names = ['your first name', 'first name', 'name', 'participant', 'your name']

    for col in df.columns:
        col_lower = str(col).lower()
        if any(name in col_lower for name in possible_names):
            return col

    # Fallback: first text column
    for col in df.columns:
        if df[col].dtype == 'object' and not col.startswith('Unnamed'):
            return col

    return None

def find_stage_column(df):
    """Find stage/week column"""
    possible_names = ['stage', 'submission', 'week', 'time']

    for col in df.columns:
        col_lower = str(col).lower()
        if any(name in col_lower for name in possible_names):
            return col

    return None

def find_rating_columns(df, cohort):
    """Find rating columns with cohort-specific logic"""
    rating_cols = []

    for col in df.columns:
        col_lower = str(col).lower()

        # Different patterns for different cohorts
        if cohort in ['H125', 'H224', 'H124', 'H223']:
            if 'idea interest' in col_lower and 'unnamed' not in col_lower:
                rating_cols.append(col)
        elif cohort == 'H123':
            if ('ideas [' in col_lower or 'idea [' in col_lower) and 'unnamed' not in col_lower:
                rating_cols.append(col)
        elif cohort in ['2022', '2021']:
            if any(word in col_lower for word in ['rank', 'choice', 'preference', 'priority']) and 'unnamed' not in col_lower:
                rating_cols.append(col)

    return rating_cols

def find_text_response_columns(df):
    """Find open-text response columns"""
    text_cols = []

    keywords = [
        'updated', 'negatively', 'uncertain', 'confident', 'reason', 'feedback',
        'comment', 'explain', 'detail', 'why', 'thoughts', 'feelings'
    ]

    for col in df.columns:
        col_lower = str(col).lower()
        if any(keyword in col_lower for keyword in keywords) and 'unnamed' not in col_lower:
            text_cols.append(col)

    return text_cols

def extract_idea_name_comprehensive(column, cohort):
    """Extract idea name with comprehensive pattern matching"""
    col_str = str(column)

    # Remove common prefixes and brackets
    cleaned = re.sub(r'^.*?\[', '', col_str)
    cleaned = re.sub(r'\].*?$', '', cleaned)
    cleaned = cleaned.replace('1. Idea Interest ', '').replace('Ideas [', '').strip()

    # Comprehensive name mappings
    name_mappings = {
        # Animal ideas
        'reducing keel bone fractures (kbf)': 'KBF',
        'keel bone fractures': 'KBF',
        'east asian fish welfare (eafw)': 'EAFW',
        'east asian fish welfare': 'EAFW',
        'cage-free campaigns in the middle east (cfme)': 'CFME',
        'cage-free campaigns': 'CFME',
        'insect industry (ii)': 'II',
        'insect industry': 'II',
        'greek fish (gf)': 'GF',
        'greek fish': 'GF',
        'animal fundraising (af)': 'AF',
        'animal fundraising': 'AF',
        'milkfish welfare (mw)': 'MW',
        'milkfish welfare': 'MW',
        'shrimp welfare': 'Shrimp Welfare',
        'banning baitfish': 'Banning Baitfish',
        'banning low welfare imports': 'Banning Low Welfare Imports',

        # Human ideas
        'labor migration platform (lmp)': 'LMP',
        'labor migration': 'LMP',
        'pla groups (pla)': 'PLA',
        'pla groups': 'PLA',
        'road traffic safety (rts)': 'RTS',
        'road traffic safety': 'RTS',
        'digital pulmonary rehabilitation (dpr)': 'DPR',
        'digital pulmonary': 'DPR',
        'oxygen (ioa)': 'IOA',
        'oxygen': 'IOA',
        'lead research organization (lro)': 'LRO',
        'lead research': 'LRO',
        'intimate partner violence (ipv)': 'IPV',
        'intimate partner': 'IPV',
        'vaccination reminders (sms)': 'SMS',
        'vaccination reminders': 'SMS',
        'structured pedagogy (sp)': 'SP',
        'structured pedagogy': 'SP',
        'reducing contraceptive stockouts (so)': 'SO',
        'contraceptive stockouts': 'SO',
        'salt intake advocacy (sia)': 'SIA',
        'salt intake': 'SIA',
        'nepi crime reduction (nepi)': 'NEPI',
        'nepi crime': 'NEPI',
        'tobacco taxation': 'Tobacco Tax',
        'kangaroo care': 'Kangaroo Care',
        'ors + zinc': 'ORS',
        'ors zinc': 'ORS',
        'antimicrobial development': 'Antimicrobial Dev',
        'lafiya': 'Lafiya'
    }

    cleaned_lower = cleaned.lower()

    # Try exact matches first
    for pattern, standard_name in name_mappings.items():
        if pattern in cleaned_lower:
            return standard_name

    # Fallback to cleaned name
    return cleaned[:30] if cleaned else 'Unknown Idea'

def extract_week_comprehensive(row, stage_col):
    """Extract week with comprehensive pattern matching"""
    if stage_col and pd.notna(row[stage_col]):
        stage_text = str(row[stage_col]).lower()

        # Handle various week formats
        patterns = [
            r'week\s*(\d+(?:\.\d+)?)',  # Week 1, Week 3.5
            r'w(\d+)',                   # W1, W2
            r'submission\s*(\d+)',       # Submission 1
        ]

        for pattern in patterns:
            match = re.search(pattern, stage_text)
            if match:
                week_num = match.group(1)
                return f"Week {week_num}"

    return None

def convert_to_standard_scale_comprehensive(value, cohort):
    """Convert different scales to 1-7 with comprehensive error handling"""
    if pd.isna(value):
        return None

    try:
        # Handle string values that might be numeric
        if isinstance(value, str):
            # Handle text like "4 (Uncertain/average)"
            if '(' in value:
                value = value.split('(')[0].strip()

            # Try to convert to float
            try:
                value = float(value)
            except ValueError:
                return None

        value = float(value)

        if cohort in ['2021']:
            # Ranking system: 1st choice = 7, 2nd choice = 6, etc.
            if 1 <= value <= 7:
                return int(8 - value)
        elif cohort == 'H123':
            # -3 to +3 scale → 1-7 scale (where -3=1, 0=4, 3=7)
            if -3 <= value <= 3:
                return int(value + 4)
        elif cohort == '2022':
            # Check if this is also a ranking system
            if 1 <= value <= 7:
                return int(8 - value)
        else:
            # Standard 1-7 scale
            if 1 <= value <= 7:
                return int(value)

        return None

    except (ValueError, TypeError):
        return None

def categorize_question_type(column_name):
    """Categorize the type of question for sentiment analysis"""
    col_lower = str(column_name).lower()

    if 'negatively' in col_lower or 'updated' in col_lower:
        return 'negative_updates'
    elif 'confident' in col_lower or 'uncertain' in col_lower:
        return 'confidence_uncertainty'
    elif 'reason' in col_lower or 'why' in col_lower:
        return 'reasoning'
    else:
        return 'general_feedback'

def analyze_sentiment_comprehensive(df_sentiment):
    """Comprehensive sentiment analysis"""
    if df_sentiment is None or len(df_sentiment) == 0:
        return None

    print("\n💭 COMPREHENSIVE SENTIMENT ANALYSIS")
    print("=" * 40)

    # Sentiment keywords (expanded)
    positive_keywords = [
        'more interested', 'more excited', 'better understanding', 'clearer', 'convinced',
        'stronger evidence', 'good fit', 'promising', 'compelling', 'effective',
        'impactful', 'feasible', 'confident', 'optimistic', 'encouraged'
    ]

    negative_keywords = [
        'less interested', 'less excited', 'updated negatively', 'concerns', 'worried',
        'skeptical', 'unclear', 'uncertain', 'difficult', 'complex', 'unfeasible',
        'poor fit', 'limited impact', 'not convinced', 'doubts', 'problems'
    ]

    reason_categories = {
        'personal_fit': ['personal fit', 'my fit', 'skills', 'experience', 'background', 'expertise'],
        'cofounder_fit': ['co-founder', 'cofounder', 'team', 'partner', 'collaboration'],
        'evidence_quality': ['evidence', 'research', 'data', 'studies', 'proof', 'literature'],
        'impact_concerns': ['impact', 'effectiveness', 'cost-effective', 'priority', 'importance'],
        'feasibility': ['feasible', 'practical', 'realistic', 'implementable', 'achievable'],
        'scalability': ['scale', 'scalable', 'grow', 'expand', 'widespread']
    }

    sentiment_results = []

    for _, row in df_sentiment.iterrows():
        text = row['response'].lower()

        # Calculate sentiment scores
        positive_score = sum(1 for keyword in positive_keywords if keyword in text)
        negative_score = sum(1 for keyword in negative_keywords if keyword in text)

        if positive_score > negative_score:
            sentiment = 'positive'
        elif negative_score > positive_score:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Extract reasons
        reasons = []
        for category, keywords in reason_categories.items():
            if any(keyword in text for keyword in keywords):
                reasons.append(category)

        sentiment_results.append({
            'participant': row['participant'],
            'cohort': row['cohort'],
            'sentiment': sentiment,
            'reasons': reasons,
            'question_type': row['question_type'],
            'response_length': len(row['response'])
        })

    sentiment_df = pd.DataFrame(sentiment_results)

    # Summary statistics
    print(f"Total responses analyzed: {len(sentiment_df)}")
    print("\nSentiment Distribution:")
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    for sentiment, count in sentiment_counts.items():
        pct = count / len(sentiment_df) * 100
        print(f"  {sentiment.title()}: {count} ({pct:.1f}%)")

    print("\nTop Reasons for Changes:")
    all_reasons = [reason for reasons in sentiment_df['reasons'] for reason in reasons]
    if all_reasons:
        reason_counts = pd.Series(all_reasons).value_counts().head(5)
        for reason, count in reason_counts.items():
            print(f"  {reason.replace('_', ' ').title()}: {count} mentions")

    return sentiment_df

if __name__ == "__main__":
    # Test the comprehensive processor
    print("Testing comprehensive data processor...")
    df_trajectories, df_sentiment = load_and_process_comprehensive_data()

    if df_trajectories is not None:
        print(f"\n✅ SUCCESS: {len(df_trajectories)} trajectories loaded")

        # Test specific error cases
        print("\n🔍 TESTING SPECIFIC ERROR CASES:")
        error_cases = [
            ('Adnaan', 'CFME', 'H125'),
            ('Amy', 'CFME', 'H125'),
            ('Miri', 'Tobacco Tax', 'H123')
        ]

        for participant, idea, cohort in error_cases:
            matches = df_trajectories[
                (df_trajectories['participant'] == participant) &
                (df_trajectories['cohort'] == cohort) &
                (df_trajectories['idea'].str.contains(idea, case=False, na=False))
            ]

            if not matches.empty:
                row = matches.iloc[0]
                print(f"  ✅ {participant} - {idea}: {row['first_rating']} → {row['last_rating']} (change: {row['change']:+.1f})")
            else:
                print(f"  ❌ {participant} - {idea}: Not found")

        # Analyze sentiment if available
        if df_sentiment is not None:
            sentiment_analysis = analyze_sentiment_comprehensive(df_sentiment)
    else:
        print("❌ Failed to load data")