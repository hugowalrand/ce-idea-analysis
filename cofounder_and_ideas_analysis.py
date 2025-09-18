#!/usr/bin/env python3
"""
Co-founder and Ideas Analysis Module
Comprehensive analysis of co-founder trajectories and idea performance
"""

import pandas as pd
import numpy as np
from comprehensive_data_processor_v3 import load_and_process_comprehensive_data

def analyze_cofounder_trajectories(df_trajectories):
    """Analyze co-founder interest trajectories over time"""
    print("\n👥 CO-FOUNDER TRAJECTORY ANALYSIS")
    print("=" * 50)

    # Filter only co-founder trajectories
    cofounder_trajectories = df_trajectories[df_trajectories['is_cofounder'] == True]

    if cofounder_trajectories.empty:
        print("No co-founder trajectories found")
        return None

    print(f"Analyzing {len(cofounder_trajectories)} co-founder trajectories")

    # Group by cohort and co-founder idea
    cofounder_analysis = []

    for cohort in cofounder_trajectories['cohort'].unique():
        cohort_data = cofounder_trajectories[cofounder_trajectories['cohort'] == cohort]

        print(f"\n{cohort} Co-founders:")

        for cofounder_idea in cohort_data['cofounder_idea'].unique():
            if pd.isna(cofounder_idea):
                continue

            idea_data = cohort_data[cohort_data['cofounder_idea'] == cofounder_idea]

            # Calculate statistics
            avg_first = idea_data['first_rating'].mean()
            avg_last = idea_data['last_rating'].mean()
            avg_change = idea_data['change'].mean()
            num_founders = len(idea_data)

            # Get individual trajectories
            founders = []
            for _, row in idea_data.iterrows():
                founders.append({
                    'name': row['participant'],
                    'first': row['first_rating'],
                    'last': row['last_rating'],
                    'change': row['change']
                })

            was_founded = idea_data['was_founded'].iloc[0] if not idea_data.empty else False

            analysis_entry = {
                'cohort': cohort,
                'idea': cofounder_idea,
                'num_founders': num_founders,
                'avg_first_rating': avg_first,
                'avg_last_rating': avg_last,
                'avg_change': avg_change,
                'was_founded': was_founded,
                'founders': founders
            }

            cofounder_analysis.append(analysis_entry)

            # Print detailed results
            print(f"  {cofounder_idea}:")
            print(f"    Founders: {num_founders}")
            print(f"    Average trajectory: {avg_first:.1f} → {avg_last:.1f} (change: {avg_change:+.1f})")
            print(f"    Founded: {'✅ Yes' if was_founded else '❌ No'}")

            for founder in founders:
                print(f"      {founder['name']}: {founder['first']} → {founder['last']} ({founder['change']:+.0f})")

    return pd.DataFrame(cofounder_analysis)

def create_comprehensive_ideas_table(df_trajectories):
    """Create comprehensive table of all ideas with founding status and performance"""
    print("\n📊 COMPREHENSIVE IDEAS ANALYSIS TABLE")
    print("=" * 50)

    ideas_analysis = []

    for idea in df_trajectories['idea'].unique():
        idea_data = df_trajectories[df_trajectories['idea'] == idea]

        # Basic statistics
        avg_first = idea_data['first_rating'].mean()
        avg_last = idea_data['last_rating'].mean()
        avg_change = idea_data['change'].mean()
        num_participants = len(idea_data)

        # Check if founded (any participant has was_founded=True)
        was_founded = idea_data['was_founded'].any()

        # Classify as animal or human
        is_animal = idea_data['is_animal_idea'].any()

        # Calculate trajectory statistics
        positive_changes = (idea_data['change'] > 0).sum()
        negative_changes = (idea_data['change'] < 0).sum()
        no_changes = (idea_data['change'] == 0).sum()

        # Cohort distribution
        cohorts = idea_data['cohort'].unique()

        ideas_analysis.append({
            'idea': idea,
            'avg_first_score': avg_first,
            'avg_last_score': avg_last,
            'avg_change': avg_change,
            'num_participants': num_participants,
            'positive_changes': positive_changes,
            'negative_changes': negative_changes,
            'no_changes': no_changes,
            'was_founded': was_founded,
            'idea_type': 'Animal' if is_animal else 'Human',
            'cohorts': ', '.join(sorted(cohorts))
        })

    ideas_df = pd.DataFrame(ideas_analysis).sort_values('avg_change', ascending=False)

    print("Complete Ideas Analysis:")
    print("=" * 80)
    print(f"{'Idea':<25} {'Avg First':<10} {'Avg Last':<10} {'Avg Change':<12} {'Participants':<12} {'Founded':<8} {'Type':<8}")
    print("=" * 80)

    for _, row in ideas_df.iterrows():
        founded_status = '✅' if row['was_founded'] else '❌'
        print(f"{row['idea'][:24]:<25} {row['avg_first_score']:<10.1f} {row['avg_last_score']:<10.1f} {row['avg_change']:<+12.2f} {row['num_participants']:<12} {founded_status:<8} {row['idea_type']:<8}")

    return ideas_df

def analyze_human_vs_animal_detailed(df_trajectories):
    """Detailed analysis of human vs animal idea preferences"""
    print("\n🐾 DETAILED HUMAN VS ANIMAL ANALYSIS")
    print("=" * 50)

    # Classify trajectories
    animal_trajectories = df_trajectories[df_trajectories['is_animal_idea'] == True]
    human_trajectories = df_trajectories[df_trajectories['is_animal_idea'] == False]

    print(f"Animal idea trajectories: {len(animal_trajectories)}")
    print(f"Human idea trajectories: {len(human_trajectories)}")

    # Statistics by type
    animal_stats = {
        'count': len(animal_trajectories),
        'avg_first': animal_trajectories['first_rating'].mean() if not animal_trajectories.empty else 0,
        'avg_last': animal_trajectories['last_rating'].mean() if not animal_trajectories.empty else 0,
        'avg_change': animal_trajectories['change'].mean() if not animal_trajectories.empty else 0,
        'positive_changes': (animal_trajectories['change'] > 0).sum() if not animal_trajectories.empty else 0,
        'negative_changes': (animal_trajectories['change'] < 0).sum() if not animal_trajectories.empty else 0
    }

    human_stats = {
        'count': len(human_trajectories),
        'avg_first': human_trajectories['first_rating'].mean() if not human_trajectories.empty else 0,
        'avg_last': human_trajectories['last_rating'].mean() if not human_trajectories.empty else 0,
        'avg_change': human_trajectories['change'].mean() if not human_trajectories.empty else 0,
        'positive_changes': (human_trajectories['change'] > 0).sum() if not human_trajectories.empty else 0,
        'negative_changes': (human_trajectories['change'] < 0).sum() if not human_trajectories.empty else 0
    }

    print(f"\nAnimal Ideas Statistics:")
    print(f"  Average first rating: {animal_stats['avg_first']:.2f}")
    print(f"  Average last rating: {animal_stats['avg_last']:.2f}")
    print(f"  Average change: {animal_stats['avg_change']:+.2f}")
    print(f"  Positive changes: {animal_stats['positive_changes']} ({animal_stats['positive_changes']/animal_stats['count']*100:.1f}%)")
    print(f"  Negative changes: {animal_stats['negative_changes']} ({animal_stats['negative_changes']/animal_stats['count']*100:.1f}%)")

    print(f"\nHuman Ideas Statistics:")
    print(f"  Average first rating: {human_stats['avg_first']:.2f}")
    print(f"  Average last rating: {human_stats['avg_last']:.2f}")
    print(f"  Average change: {human_stats['avg_change']:+.2f}")
    print(f"  Positive changes: {human_stats['positive_changes']} ({human_stats['positive_changes']/human_stats['count']*100:.1f}%)")
    print(f"  Negative changes: {human_stats['negative_changes']} ({human_stats['negative_changes']/human_stats['count']*100:.1f}%)")

    # Find participants with both types <4 in first AND final week
    participants_with_both = []

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
                participants_with_both.append({
                    'participant': participant,
                    'cohort': p_data['cohort'].iloc[0],
                    'animal_trajectories': len(animal_data),
                    'human_trajectories': len(human_data)
                })

    print(f"\nParticipants with <4 ratings for BOTH animal and human ideas (first & final weeks): {len(participants_with_both)}")
    for p in participants_with_both:
        print(f"  {p['participant']} ({p['cohort']}): {p['animal_trajectories']} animal + {p['human_trajectories']} human trajectories")

    return {
        'animal_stats': animal_stats,
        'human_stats': human_stats,
        'participants_with_both_low': participants_with_both
    }

def analyze_sentiment_by_idea(df_sentiment, df_trajectories):
    """Analyze sentiment patterns by specific ideas"""
    if df_sentiment is None or df_sentiment.empty:
        print("\nNo sentiment data available for analysis")
        return None

    print("\n💭 SENTIMENT ANALYSIS BY IDEA TYPE")
    print("=" * 50)

    # Get ideas that were not founded for special attention
    ideas_df = create_comprehensive_ideas_table(df_trajectories)
    unfounded_ideas = ideas_df[ideas_df['was_founded'] == False]['idea'].tolist()

    print(f"Focusing on {len(unfounded_ideas)} unfounded ideas for sentiment analysis:")
    for idea in unfounded_ideas[:10]:  # Show first 10
        print(f"  - {idea}")

    # Analyze sentiment patterns
    sentiment_by_type = df_sentiment.groupby(['question_type', 'sentiment']).size().unstack(fill_value=0)

    print("\nSentiment by Question Type:")
    print(sentiment_by_type)

    # Analyze reasons for negative sentiment
    negative_sentiment = df_sentiment[df_sentiment['sentiment'] == 'negative']
    if not negative_sentiment.empty:
        print(f"\nNegative Sentiment Analysis ({len(negative_sentiment)} responses):")

        # Count reasons
        all_reasons = [reason for reasons in negative_sentiment['reasons'] for reason in reasons]
        if all_reasons:
            reason_counts = pd.Series(all_reasons).value_counts().head(5)
            print("Top reasons for negative sentiment:")
            for reason, count in reason_counts.items():
                print(f"  {reason.replace('_', ' ').title()}: {count} mentions")

    return {
        'sentiment_by_type': sentiment_by_type,
        'unfounded_ideas': unfounded_ideas,
        'negative_reasons': reason_counts.to_dict() if 'reason_counts' in locals() else {}
    }

def main():
    """Run comprehensive analysis"""
    print("🔬 COMPREHENSIVE CO-FOUNDER AND IDEAS ANALYSIS")
    print("=" * 60)

    # Load data using the comprehensive processor
    df_trajectories, df_sentiment = load_and_process_comprehensive_data()

    if df_trajectories is None:
        print("❌ Failed to load trajectory data")
        return

    # Run all analyses
    cofounder_df = analyze_cofounder_trajectories(df_trajectories)
    ideas_df = create_comprehensive_ideas_table(df_trajectories)
    human_animal_analysis = analyze_human_vs_animal_detailed(df_trajectories)
    sentiment_analysis = analyze_sentiment_by_idea(df_sentiment, df_trajectories)

    # Save results
    if cofounder_df is not None:
        cofounder_df.to_csv('cofounder_analysis_results.csv', index=False)
        print(f"\n💾 Co-founder analysis saved to: cofounder_analysis_results.csv")

    if ideas_df is not None:
        ideas_df.to_csv('comprehensive_ideas_analysis.csv', index=False)
        print(f"💾 Ideas analysis saved to: comprehensive_ideas_analysis.csv")

    print(f"\n🎉 Comprehensive analysis complete!")
    print(f"   📊 {len(df_trajectories)} trajectories analyzed")
    print(f"   👥 {len(cofounder_df) if cofounder_df is not None else 0} co-founder trajectories")
    print(f"   💡 {len(ideas_df) if ideas_df is not None else 0} unique ideas")
    print(f"   💭 {len(df_sentiment) if df_sentiment is not None else 0} sentiment responses")

    return {
        'trajectories': df_trajectories,
        'sentiment': df_sentiment,
        'cofounder_analysis': cofounder_df,
        'ideas_analysis': ideas_df,
        'human_animal_analysis': human_animal_analysis,
        'sentiment_analysis': sentiment_analysis
    }

if __name__ == "__main__":
    results = main()