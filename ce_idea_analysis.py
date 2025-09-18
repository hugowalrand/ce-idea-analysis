#!/usr/bin/env python3
"""
CE Idea Interest Over Time Analysis

This script analyzes how incubatee idea preferences change over time across cohorts,
focusing on trajectory patterns, transition probabilities, and founding outcomes.

Based on requirements from "CE Idea Interest Over Time Analysis.docx"
Data source: "Idea Interest Over Time Data for Elizabeth.xlsx"
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class CEIdeaAnalyzer:
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.sheets = {}
        self.processed_data = None
        self.cohorts = ['H125', 'H224', 'H124', 'H223', 'H123', '2022', '2021', '2020']
        
    def load_data(self):
        """Load all sheets from the Excel file"""
        print("Loading data from Excel file...")
        for cohort in self.cohorts:
            try:
                df = pd.read_excel(self.data_path, sheet_name=cohort)
                self.sheets[cohort] = df
                print(f"  Loaded {cohort}: {len(df)} rows")
            except Exception as e:
                print(f"  Warning: Could not load {cohort}: {e}")
        print(f"Successfully loaded {len(self.sheets)} cohort sheets\n")
    
    def process_data(self):
        """Process and standardize data across cohorts"""
        print("Processing and standardizing data...")
        processed_sheets = []
        
        for cohort, df in self.sheets.items():
            if df.empty:
                continue
                
            # Create a standardized format
            processed_df = self._standardize_cohort_data(df, cohort)
            if processed_df is not None:
                processed_sheets.append(processed_df)
        
        if processed_sheets:
            self.processed_data = pd.concat(processed_sheets, ignore_index=True)
            print(f"Combined data: {len(self.processed_data)} total responses")
        else:
            print("No data could be processed")
    
    def _standardize_cohort_data(self, df, cohort):
        """Standardize individual cohort data format"""
        try:
            # Identify participant column
            participant_col = None
            for col in df.columns:
                if 'name' in col.lower() or col == df.columns[0]:
                    participant_col = col
                    break
            
            if participant_col is None:
                print(f"  Warning: No participant column found in {cohort}")
                return None
            
            # Identify stage/week column
            stage_col = None
            for col in df.columns:
                if 'stage' in col.lower() or 'submission' in col.lower():
                    stage_col = col
                    break
            
            # Identify idea rating columns
            idea_cols = []
            for col in df.columns:
                if 'idea interest' in col.lower() or self._is_idea_column(col, cohort):
                    idea_cols.append(col)
            
            standardized_rows = []
            
            # Process each row
            for idx, row in df.iterrows():
                participant = row[participant_col]
                if pd.isna(participant):
                    continue
                
                # Extract week/stage info
                week = self._extract_week(row, stage_col, idx)
                if week is None:
                    continue
                
                # Process each idea column
                for idea_col in idea_cols:
                    if pd.notna(row[idea_col]):
                        raw_rating = row[idea_col]
                        
                        # Convert to standard 1-7 scale
                        standardized_rating = self._convert_to_standard_scale(raw_rating, cohort, idea_col)
                        
                        if standardized_rating is not None:
                            idea_name = self._extract_idea_name_from_column(idea_col)
                            
                            standardized_rows.append({
                                'cohort': cohort,
                                'participant': participant,
                                'week': week,
                                'rating': standardized_rating,
                                'idea': idea_name,
                                'raw_rating': raw_rating
                            })
            
            if standardized_rows:
                return pd.DataFrame(standardized_rows)
            return None
            
        except Exception as e:
            print(f"  Error processing {cohort}: {e}")
            return None
    
    def _is_idea_column(self, col, cohort):
        """Check if column contains idea ratings"""
        indicators = ['reducing', 'platform', 'research', 'policy', 'intervention', 'program']
        return any(indicator in col.lower() for indicator in indicators)
    
    def _extract_week(self, row, stage_col, idx):
        """Extract week information from stage column or row position"""
        if stage_col and pd.notna(row[stage_col]):
            stage_text = str(row[stage_col]).lower()
            if 'week' in stage_text:
                # Extract week number
                import re
                match = re.search(r'week\s*(\d+)', stage_text)
                if match:
                    return f"Week {match.group(1)}"
        
        # Fallback: assume sequential weeks
        week_num = (idx % 5) + 1
        return f"Week {week_num}"
    
    def _convert_to_standard_scale(self, value, cohort, column):
        """Convert different rating scales to standard 1-7 scale"""
        try:
            if pd.isna(value):
                return None
            
            # Handle different cohort scales
            if cohort in ['2020', '2021']:
                # Ranking system: 1st choice = 7, 2nd = 6, etc.
                if isinstance(value, (int, float)) and 1 <= value <= 7:
                    return 8 - int(value)
            elif cohort == 'H123':
                # -3 to +3 scale
                if isinstance(value, (int, float)) and -3 <= value <= 3:
                    return value + 4
            else:
                # Standard 1-7 scale
                if isinstance(value, (int, float)) and 1 <= value <= 7:
                    return float(value)
            
            return None
            
        except (ValueError, TypeError):
            return None
    
    def _extract_idea_name_from_column(self, col):
        """Extract clean idea name from column header"""
        # Remove common prefixes and clean up
        cleaned = col.replace('1. Idea Interest [', '').replace(']', '')
        cleaned = cleaned.replace('Idea Interest ', '').strip()
        
        # Extract key terms
        if 'reducing keel bone' in cleaned.lower():
            return 'Keel Bone Fractures (KBF)'
        elif 'labor migration' in cleaned.lower():
            return 'Labor Migration Platform (LMP)'
        elif 'policy' in cleaned.lower():
            return 'Policy Research'
        elif 'research' in cleaned.lower():
            return 'Research Initiative'
        else:
            # Return first 30 characters as fallback
            return cleaned[:30].strip()
    
    def analyze_sentiment_responses(self):
        """P2: Analyze sentiment from qualitative responses"""
        print("4. SENTIMENT ANALYSIS (P2 Requirement)")
        print("=" * 40)
        
        # Load qualitative data from Excel
        qualitative_data = []
        for cohort, df in self.sheets.items():
            # Look for text response columns
            text_cols = [col for col in df.columns if any(word in col.lower() 
                        for word in ['comment', 'feedback', 'uncertain', 'change'])]
            
            if text_cols:
                for col in text_cols:
                    responses = df[col].dropna()
                    for response in responses:
                        if isinstance(response, str) and len(response.strip()) > 10:
                            qualitative_data.append({
                                'cohort': cohort,
                                'response': response.strip(),
                                'column': col
                            })
        
        if qualitative_data:
            print(f"Found {len(qualitative_data)} qualitative responses for analysis")
            
            # Simple sentiment categorization
            positive_keywords = ['better', 'clearer', 'more interested', 'understand', 'convinced']
            negative_keywords = ['less interested', 'confused', 'uncertain', 'difficult', 'complex']
            
            sentiments = []
            for item in qualitative_data:
                text = item['response'].lower()
                if any(word in text for word in positive_keywords):
                    sentiment = 'positive_change'
                elif any(word in text for word in negative_keywords):
                    sentiment = 'negative_change'
                else:
                    sentiment = 'neutral'
                
                sentiments.append(sentiment)
            
            # Summary
            sentiment_counts = pd.Series(sentiments).value_counts()
            print("\nSentiment Analysis Results:")
            for sentiment, count in sentiment_counts.items():
                print(f"  - {sentiment.replace('_', ' ').title()}: {count} ({count/len(sentiments):.1%})")
        else:
            print("No qualitative response data found for sentiment analysis")
        
        print("\n" + "="*50 + "\n")
    
    def analyze_founder_journeys(self):
        """P1: Analyze founder trajectories for launched ideas"""
        print("5. FOUNDER JOURNEY ANALYSIS (P1 Requirement)")
        print("=" * 40)
        
        # This would need actual founding outcome data
        # For now, provide framework for when that data is available
        
        print("Framework for founder analysis:")
        print("  1. Identify participants who founded ideas")
        print("  2. Track their rating trajectories for founded ideas")
        print("  3. Compare program vs non-program founded ideas")
        print("  4. Analyze pre-program vs program idea founding rates")
        
        print("\nNote: Requires founding outcome data to be added to dataset")
        print("\n" + "="*50 + "\n")
    
    def analyze_cause_area_convergence(self):
        """P1: Analyze cause area convergence and agnostic founders"""
        if self.processed_data is None:
            return
        
        print("6. CAUSE AREA CONVERGENCE ANALYSIS (P1 Requirement)")
        print("=" * 40)
        
        # Group ideas by cause areas (would need cause area mapping)
        cause_areas = {
            'Animal Welfare': ['keel bone', 'animal'],
            'Global Development': ['migration', 'labor'],
            'Policy': ['policy'],
            'Research': ['research']
        }
        
        # Analyze convergence patterns
        participant_preferences = self.processed_data.groupby(['cohort', 'participant', 'idea']).agg({
            'rating': ['first', 'last', 'mean']
        })
        
        print("Cause area convergence analysis framework created")
        print("Note: Requires detailed cause area mapping for full analysis")
        
        print("\n" + "="*50 + "\n")
    
    def analyze_preference_changes(self):
        """Analyze how preferences change over time"""
        if self.processed_data is None:
            print("No processed data available for analysis")
            return
        
        print("=== PREFERENCE CHANGE ANALYSIS ===\n")
        
        # Group by participant and idea to track changes
        participant_trajectories = []
        
        for (cohort, participant, idea), group in self.processed_data.groupby(['cohort', 'participant', 'idea']):
            if len(group) >= 2:  # Need at least 2 data points
                group_sorted = group.sort_values('week')
                first_rating = group_sorted.iloc[0]['rating']
                last_rating = group_sorted.iloc[-1]['rating']
                
                trajectory = {
                    'cohort': cohort,
                    'participant': participant,
                    'idea': idea,
                    'first_rating': first_rating,
                    'last_rating': last_rating,
                    'change': last_rating - first_rating,
                    'weeks_tracked': len(group_sorted)
                }
                participant_trajectories.append(trajectory)
        
        trajectories_df = pd.DataFrame(participant_trajectories)
        
        if trajectories_df.empty:
            print("No trajectory data available")
            return
        
        print(f"Analyzed {len(trajectories_df)} participant-idea trajectories\n")
        
        # P1: Analyze transition probabilities
        self._analyze_transition_probabilities(trajectories_df)
        
        # Analyze overall change patterns
        self._analyze_change_patterns(trajectories_df)
        
        return trajectories_df
    
    def _analyze_transition_probabilities(self, trajectories_df):
        """P1: Analyze negative-to-positive and positive-to-negative transitions"""
        print("1. TRANSITION PROBABILITY ANALYSIS")
        print("=" * 40)
        
        # Define categories
        negative_ratings = trajectories_df['first_rating'] <= 3
        positive_ratings = trajectories_df['first_rating'] >= 5
        neutral_ratings = trajectories_df['first_rating'] == 4
        
        became_positive = trajectories_df['last_rating'] >= 5
        became_negative = trajectories_df['last_rating'] <= 3
        
        # Negative to positive transitions
        neg_to_pos_candidates = trajectories_df[negative_ratings]
        if len(neg_to_pos_candidates) > 0:
            neg_to_pos_success = neg_to_pos_candidates[became_positive]
            neg_to_pos_prob = len(neg_to_pos_success) / len(neg_to_pos_candidates)
            
            print(f"Negative→Positive Transitions:")
            print(f"  - Participants starting negative (1-3): {len(neg_to_pos_candidates)}")
            print(f"  - Became positive (5-7): {len(neg_to_pos_success)}")
            print(f"  - Probability: {neg_to_pos_prob:.2%}")
            
            if len(neg_to_pos_success) > 0:
                avg_change = neg_to_pos_success['change'].mean()
                print(f"  - Average rating change: +{avg_change:.2f}")
        
        print()
        
        # Positive to negative transitions
        pos_to_neg_candidates = trajectories_df[positive_ratings]
        if len(pos_to_neg_candidates) > 0:
            pos_to_neg_success = pos_to_neg_candidates[became_negative]
            pos_to_neg_prob = len(pos_to_neg_success) / len(pos_to_neg_candidates)
            
            print(f"Positive→Negative Transitions:")
            print(f"  - Participants starting positive (5-7): {len(pos_to_neg_candidates)}")
            print(f"  - Became negative (1-3): {len(pos_to_neg_success)}")
            print(f"  - Probability: {pos_to_neg_prob:.2%}")
            
            if len(pos_to_neg_success) > 0:
                avg_change = pos_to_neg_success['change'].mean()
                print(f"  - Average rating change: {avg_change:.2f}")
        
        print("\n" + "="*50 + "\n")
    
    def _analyze_change_patterns(self, trajectories_df):
        """Analyze overall patterns of change"""
        print("2. OVERALL CHANGE PATTERNS")
        print("=" * 40)
        
        # Overall statistics
        avg_change = trajectories_df['change'].mean()
        std_change = trajectories_df['change'].std()
        
        print(f"Average rating change: {avg_change:+.2f}")
        print(f"Standard deviation: {std_change:.2f}")
        print(f"% with positive change: {(trajectories_df['change'] > 0).mean():.1%}")
        print(f"% with negative change: {(trajectories_df['change'] < 0).mean():.1%}")
        print(f"% with no change: {(trajectories_df['change'] == 0).mean():.1%}")
        
        # Change magnitude analysis
        large_positive = (trajectories_df['change'] >= 2).sum()
        large_negative = (trajectories_df['change'] <= -2).sum()
        
        print(f"\nLarge changes (±2 or more):")
        print(f"  - Large positive changes (≥+2): {large_positive} ({large_positive/len(trajectories_df):.1%})")
        print(f"  - Large negative changes (≤-2): {large_negative} ({large_negative/len(trajectories_df):.1%})")
        
        print("\n" + "="*50 + "\n")
    
    def analyze_by_cohort(self):
        """Analyze patterns by cohort"""
        if self.processed_data is None:
            return
        
        print("3. COHORT-WISE ANALYSIS")
        print("=" * 40)
        
        cohort_stats = []
        for cohort in self.processed_data['cohort'].unique():
            cohort_data = self.processed_data[self.processed_data['cohort'] == cohort]
            
            # Calculate basic stats
            stats_dict = {
                'cohort': cohort,
                'participants': cohort_data['participant'].nunique(),
                'total_responses': len(cohort_data),
                'avg_rating': cohort_data['rating'].mean(),
                'rating_std': cohort_data['rating'].std()
            }
            cohort_stats.append(stats_dict)
        
        cohort_df = pd.DataFrame(cohort_stats)
        
        print("Cohort Summary:")
        for _, row in cohort_df.iterrows():
            print(f"  {row['cohort']}: {row['participants']} participants, "
                  f"{row['total_responses']} responses, "
                  f"avg rating: {row['avg_rating']:.2f} (±{row['rating_std']:.2f})")
        
        print("\n" + "="*50 + "\n")
        
        return cohort_df
    
    def create_visualizations(self):
        """Create key visualizations"""
        if self.processed_data is None:
            print("No data available for visualization")
            return
        
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('CE Idea Interest Over Time Analysis', fontsize=16, fontweight='bold')
        
        # 1. Rating distribution by cohort
        ax1 = axes[0, 0]
        cohorts_to_plot = list(self.processed_data['cohort'].unique())[:6]  # Limit for readability
        data_subset = self.processed_data[self.processed_data['cohort'].isin(cohorts_to_plot)]
        
        sns.boxplot(data=data_subset, x='cohort', y='rating', ax=ax1)
        ax1.set_title('Rating Distribution by Cohort')
        ax1.set_ylabel('Interest Rating (1-7)')
        ax1.set_xlabel('Cohort')
        plt.setp(ax1.get_xticklabels(), rotation=45)
        
        # 2. Average ratings over time (if week data is numeric)
        ax2 = axes[0, 1]
        # This would need more sophisticated week parsing
        ax2.hist(self.processed_data['rating'], bins=7, range=(0.5, 7.5), alpha=0.7, edgecolor='black')
        ax2.set_title('Overall Rating Distribution')
        ax2.set_xlabel('Interest Rating')
        ax2.set_ylabel('Frequency')
        ax2.set_xticks(range(1, 8))
        
        # 3. Participant engagement (responses per participant)
        ax3 = axes[1, 0]
        participant_counts = self.processed_data.groupby('participant').size()
        ax3.hist(participant_counts, bins=10, alpha=0.7, edgecolor='black')
        ax3.set_title('Participant Engagement\n(Responses per Participant)')
        ax3.set_xlabel('Number of Responses')
        ax3.set_ylabel('Number of Participants')
        
        # 4. Rating trends by cohort
        ax4 = axes[1, 1]
        cohort_means = self.processed_data.groupby('cohort')['rating'].mean().sort_index()
        ax4.plot(range(len(cohort_means)), cohort_means.values, 'o-', linewidth=2, markersize=8)
        ax4.set_title('Average Rating by Cohort')
        ax4.set_xlabel('Cohort (chronological order)')
        ax4.set_ylabel('Average Interest Rating')
        ax4.set_xticks(range(len(cohort_means)))
        ax4.set_xticklabels(cohort_means.index, rotation=45)
        
        plt.tight_layout()
        
        # Save the visualization
        output_path = self.data_path.parent / 'ce_idea_analysis_visualizations.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Visualizations saved to: {output_path}")
        
        plt.show()
    
    def generate_summary_report(self, trajectories_df=None):
        """Generate a summary report of findings"""
        report_path = self.data_path.parent / 'ce_idea_analysis_summary.txt'
        
        with open(report_path, 'w') as f:
            f.write("CE IDEA INTEREST OVER TIME ANALYSIS - SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write(f"Data Source: {self.data_path.name}\n\n")
            
            if self.processed_data is not None:
                f.write("DATA OVERVIEW:\n")
                f.write(f"  - Total responses: {len(self.processed_data)}\n")
                f.write(f"  - Unique participants: {self.processed_data['participant'].nunique()}\n")
                f.write(f"  - Cohorts analyzed: {', '.join(sorted(self.processed_data['cohort'].unique()))}\n")
                f.write(f"  - Average rating: {self.processed_data['rating'].mean():.2f}\n")
                f.write(f"  - Rating range: {self.processed_data['rating'].min():.1f} - {self.processed_data['rating'].max():.1f}\n\n")
            
            if trajectories_df is not None and not trajectories_df.empty:
                f.write("KEY FINDINGS:\n")
                
                # Transition probabilities
                negative_start = trajectories_df['first_rating'] <= 3
                positive_end = trajectories_df['last_rating'] >= 5
                
                neg_to_pos_candidates = trajectories_df[negative_start]
                if len(neg_to_pos_candidates) > 0:
                    neg_to_pos_success = neg_to_pos_candidates[positive_end]
                    prob = len(neg_to_pos_success) / len(neg_to_pos_candidates)
                    f.write(f"  - Negative→Positive transition probability: {prob:.1%}\n")
                
                positive_start = trajectories_df['first_rating'] >= 5
                negative_end = trajectories_df['last_rating'] <= 3
                
                pos_to_neg_candidates = trajectories_df[positive_start]
                if len(pos_to_neg_candidates) > 0:
                    pos_to_neg_success = pos_to_neg_candidates[negative_end]
                    prob = len(pos_to_neg_success) / len(pos_to_neg_candidates)
                    f.write(f"  - Positive→Negative transition probability: {prob:.1%}\n")
                
                f.write(f"  - Average rating change: {trajectories_df['change'].mean():+.2f}\n")
                f.write(f"  - % with positive change: {(trajectories_df['change'] > 0).mean():.1%}\n")
                f.write(f"  - % with negative change: {(trajectories_df['change'] < 0).mean():.1%}\n")
                
            f.write("\nRECOMMendations for next steps:\n")
            f.write("  1. Conduct detailed sentiment analysis of qualitative feedback\n")
            f.write("  2. Analyze founder journeys for launched ideas specifically\n")
            f.write("  3. Investigate idea-specific patterns and popularity effects\n")
            f.write("  4. Validate findings with department stakeholders\n")
        
        print(f"Summary report saved to: {report_path}")
        return report_path

def main():
    """Main analysis workflow"""
    data_path = "/Users/hugo/Documents/AIM/Data Analysis/Idea Interest Over Time Data for Elizabeth.xlsx"
    
    print("CE IDEA INTEREST OVER TIME ANALYSIS")
    print("=" * 50)
    print(f"Analyzing data from: {data_path}\n")
    
    # Initialize analyzer
    analyzer = CEIdeaAnalyzer(data_path)
    
    # Load and process data
    analyzer.load_data()
    analyzer.process_data()
    
    if analyzer.processed_data is None:
        print("Error: Could not process data. Please check the file format.")
        return
    
    # Run analyses
    trajectories = analyzer.analyze_preference_changes()
    analyzer.analyze_by_cohort()
    analyzer.analyze_sentiment_responses()
    analyzer.analyze_founder_journeys()
    analyzer.analyze_cause_area_convergence()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Generate summary report
    analyzer.generate_summary_report(trajectories)
    
    print("\nAnalysis complete! Check the generated files for results.")

if __name__ == "__main__":
    main()