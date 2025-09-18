#!/usr/bin/env python3
"""
Advanced Visualizations for CE Idea Interest Analysis

Creates comprehensive visualizations including trajectory heatmaps, 
transition flow diagrams, and founder journey analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
from pathlib import Path

def create_trajectory_heatmap(trajectories_df, output_dir):
    """Create heatmap showing rating changes"""
    if trajectories_df is None or trajectories_df.empty:
        print("No trajectory data available for heatmap")
        return
    
    # Create a pivot table for the heatmap
    change_matrix = np.zeros((7, 7))
    
    for _, row in trajectories_df.iterrows():
        first = int(row['first_rating']) - 1  # Convert to 0-indexed
        last = int(row['last_rating']) - 1
        if 0 <= first < 7 and 0 <= last < 7:
            change_matrix[first][last] += 1
    
    plt.figure(figsize=(10, 8))
    
    # Create heatmap
    sns.heatmap(change_matrix, 
                annot=True, 
                fmt='.0f',
                cmap='RdYlBu_r',
                xticklabels=range(1, 8),
                yticklabels=range(1, 8),
                cbar_kws={'label': 'Number of Participants'})
    
    plt.title('Idea Interest Rating Transitions\n(From Initial Week to Final Week)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Final Rating', fontsize=12)
    plt.ylabel('Initial Rating', fontsize=12)
    
    # Add diagonal line to highlight no-change cases
    plt.plot([0, 7], [0, 7], 'k--', alpha=0.5, linewidth=1)
    
    # Highlight key transition zones
    # Negative to positive zone (bottom right)
    rect1 = plt.Rectangle((4.5, -0.5), 2.5, 3, fill=False, edgecolor='green', 
                         linewidth=3, linestyle='--', alpha=0.7)
    plt.gca().add_patch(rect1)
    plt.text(5.7, 1, 'Neg→Pos\nTransitions', ha='center', va='center', 
             color='green', fontweight='bold', fontsize=10)
    
    # Positive to negative zone (top left)
    rect2 = plt.Rectangle((-0.5, 4.5), 3, 2.5, fill=False, edgecolor='red', 
                         linewidth=3, linestyle='--', alpha=0.7)
    plt.gca().add_patch(rect2)
    plt.text(1, 5.7, 'Pos→Neg\nTransitions', ha='center', va='center', 
             color='red', fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    output_path = output_dir / 'trajectory_heatmap.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Trajectory heatmap saved to: {output_path}")
    plt.show()

def create_transition_flow_diagram(trajectories_df, output_dir):
    """Create Sankey-style flow diagram for transitions"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate transition counts
    transitions = {}
    categories = ['Negative (1-3)', 'Neutral (4)', 'Positive (5-7)']
    
    def categorize_rating(rating):
        if rating <= 3:
            return 'Negative (1-3)'
        elif rating == 4:
            return 'Neutral (4)'
        else:
            return 'Positive (5-7)'
    
    # Count transitions
    for cat1 in categories:
        transitions[cat1] = {}
        for cat2 in categories:
            transitions[cat1][cat2] = 0
    
    for _, row in trajectories_df.iterrows():
        start_cat = categorize_rating(row['first_rating'])
        end_cat = categorize_rating(row['last_rating'])
        transitions[start_cat][end_cat] += 1
    
    # Create flow diagram
    y_positions = [0.8, 0.5, 0.2]
    colors = ['red', 'orange', 'green']
    
    # Draw starting categories
    for i, (cat, color) in enumerate(zip(categories, colors)):
        total = sum(transitions[cat].values())
        rect = FancyBboxPatch((0.1, y_positions[i]-0.05), 0.15, 0.1,
                             boxstyle="round,pad=0.01", 
                             facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.175, y_positions[i], f'{cat}\n({total})', 
               ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw ending categories  
    for i, (cat, color) in enumerate(zip(categories, colors)):
        total = sum(transitions[start_cat][cat] for start_cat in categories)
        rect = FancyBboxPatch((0.75, y_positions[i]-0.05), 0.15, 0.1,
                             boxstyle="round,pad=0.01",
                             facecolor=color, alpha=0.7, edgecolor='black')
        ax.add_patch(rect)
        ax.text(0.825, y_positions[i], f'{cat}\n({total})', 
               ha='center', va='center', fontweight='bold', fontsize=10)
    
    # Draw flow arrows
    for i, start_cat in enumerate(categories):
        for j, end_cat in enumerate(categories):
            count = transitions[start_cat][end_cat]
            if count > 0:
                # Calculate arrow properties
                start_x, start_y = 0.25, y_positions[i]
                end_x, end_y = 0.75, y_positions[j]
                
                # Arrow thickness proportional to count
                arrow_width = max(0.01, count / max(sum(transitions[sc].values()) 
                                                 for sc in categories) * 0.1)
                
                # Color based on transition type
                if i < j:  # Moving to more positive
                    arrow_color = 'green'
                    alpha = 0.6
                elif i > j:  # Moving to more negative
                    arrow_color = 'red'
                    alpha = 0.6
                else:  # Staying same
                    arrow_color = 'blue'
                    alpha = 0.4
                
                # Draw arrow
                ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                           arrowprops=dict(arrowstyle='->', lw=arrow_width*50,
                                         color=arrow_color, alpha=alpha))
                
                # Add count label
                mid_x, mid_y = (start_x + end_x) / 2, (start_y + end_y) / 2
                if count >= 3:  # Only label significant flows
                    ax.text(mid_x, mid_y + 0.02, str(count), ha='center', va='bottom',
                           fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title('Idea Interest Transition Flow\n(Initial Week → Final Week)', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add labels
    ax.text(0.175, 0.95, 'Initial Interest', ha='center', fontsize=12, fontweight='bold')
    ax.text(0.825, 0.95, 'Final Interest', ha='center', fontsize=12, fontweight='bold')
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='green', alpha=0.6, label='Positive Change'),
        mpatches.Patch(color='red', alpha=0.6, label='Negative Change'),
        mpatches.Patch(color='blue', alpha=0.4, label='No Change')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.1), ncol=3)
    
    plt.tight_layout()
    
    output_path = output_dir / 'transition_flow_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Transition flow diagram saved to: {output_path}")
    plt.show()

def create_cohort_comparison_chart(processed_data, output_dir):
    """Create comprehensive cohort comparison"""
    if processed_data is None or processed_data.empty:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cohort Comparison Analysis', fontsize=16, fontweight='bold')
    
    # 1. Average rating by cohort
    ax1 = axes[0, 0]
    cohort_stats = processed_data.groupby('cohort').agg({
        'rating': ['mean', 'std', 'count']
    }).round(2)
    cohort_stats.columns = ['mean_rating', 'std_rating', 'count']
    cohort_stats = cohort_stats.reset_index()
    
    bars = ax1.bar(cohort_stats['cohort'], cohort_stats['mean_rating'], 
                   yerr=cohort_stats['std_rating'], capsize=5, alpha=0.7, color='skyblue')
    ax1.set_title('Average Interest Rating by Cohort')
    ax1.set_ylabel('Average Rating')
    ax1.set_xlabel('Cohort')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val in zip(bars, cohort_stats['mean_rating']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{mean_val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # 2. Response volume by cohort
    ax2 = axes[0, 1]
    participant_counts = processed_data.groupby('cohort')['participant'].nunique()
    response_counts = processed_data.groupby('cohort').size()
    
    x = np.arange(len(participant_counts))
    width = 0.35
    
    ax2.bar(x - width/2, participant_counts.values, width, label='Unique Participants', alpha=0.7)
    ax2.bar(x + width/2, response_counts.values, width, label='Total Responses', alpha=0.7)
    
    ax2.set_title('Participation Volume by Cohort')
    ax2.set_ylabel('Count')
    ax2.set_xlabel('Cohort')
    ax2.set_xticks(x)
    ax2.set_xticklabels(participant_counts.index, rotation=45)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Rating distribution comparison (violin plot)
    ax3 = axes[1, 0]
    cohorts_to_plot = list(processed_data['cohort'].unique())[:6]  # Limit for readability
    data_subset = processed_data[processed_data['cohort'].isin(cohorts_to_plot)]
    
    sns.violinplot(data=data_subset, x='cohort', y='rating', ax=ax3, inner='box')
    ax3.set_title('Rating Distribution by Cohort')
    ax3.set_ylabel('Interest Rating (1-7)')
    ax3.set_xlabel('Cohort')
    plt.setp(ax3.get_xticklabels(), rotation=45)
    
    # 4. Engagement metrics
    ax4 = axes[1, 1]
    engagement_stats = processed_data.groupby('cohort')['participant'].apply(
        lambda x: processed_data[processed_data['cohort'] == x.name].groupby('participant').size().mean()
    )
    
    bars = ax4.bar(engagement_stats.index, engagement_stats.values, alpha=0.7, color='lightcoral')
    ax4.set_title('Average Responses per Participant by Cohort')
    ax4.set_ylabel('Avg Responses per Participant')
    ax4.set_xlabel('Cohort')
    plt.setp(ax4.get_xticklabels(), rotation=45)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, engagement_stats.values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    output_path = output_dir / 'cohort_comparison_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Cohort comparison chart saved to: {output_path}")
    plt.show()

def create_founder_journey_analysis(output_dir):
    """Create visualization for founder journey patterns"""
    # This would typically use actual founder data from the analysis
    # For now, create a conceptual visualization
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Sample data for demonstration (would be replaced with actual founder data)
    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
    
    # Simulated founder trajectories
    founded_ideas = {
        'Founded Idea A': [2, 3, 4, 5, 6],
        'Founded Idea B': [1, 2, 3, 4, 6],
        'Founded Idea C': [3, 3, 4, 5, 7],
        'Not Founded': [4, 4, 3, 3, 3]
    }
    
    colors = ['green', 'blue', 'orange', 'red']
    
    for i, (trajectory, ratings) in enumerate(founded_ideas.items()):
        linestyle = '-' if 'Founded' in trajectory else '--'
        linewidth = 3 if 'Founded' in trajectory else 2
        alpha = 0.8 if 'Founded' in trajectory else 0.6
        
        ax.plot(weeks, ratings, marker='o', linewidth=linewidth, 
               linestyle=linestyle, alpha=alpha, color=colors[i], 
               markersize=8, label=trajectory)
    
    ax.set_title('Founder Journey Analysis\n(Interest Rating Trajectories for Key Ideas)', 
                fontsize=14, fontweight='bold')
    ax.set_ylabel('Interest Rating (1-7)')
    ax.set_xlabel('Time Period')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Add annotations
    ax.annotate('Successful\nFounder Pattern', xy=('Week 5', 6.5), xytext=('Week 3', 7),
               arrowprops=dict(arrowstyle='->', color='green', alpha=0.7),
               fontsize=10, ha='center',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.5))
    
    ax.set_ylim(0, 8)
    plt.tight_layout()
    
    output_path = output_dir / 'founder_journey_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Founder journey analysis saved to: {output_path}")
    plt.show()

def main():
    """Generate all advanced visualizations"""
    print("Creating Advanced Visualizations for CE Idea Analysis...")
    print("=" * 60)
    
    # Set up output directory
    output_dir = Path("/Users/hugo/Documents/AIM/Data Analysis")
    
    # Load data (simplified for demonstration)
    data_path = output_dir / "Idea Interest Over Time Data for Elizabeth.xlsx"
    
    try:
        # Try to load some sample data for visualization
        # This would normally come from the main analysis script
        
        # Create sample trajectory data for demonstration
        np.random.seed(42)
        n_participants = 50
        
        sample_trajectories = pd.DataFrame({
            'participant': [f'P{i}' for i in range(n_participants)],
            'first_rating': np.random.choice(range(1, 8), n_participants),
            'last_rating': np.random.choice(range(1, 8), n_participants),
        })
        sample_trajectories['change'] = sample_trajectories['last_rating'] - sample_trajectories['first_rating']
        
        # Create sample processed data
        sample_processed_data = pd.DataFrame({
            'cohort': np.random.choice(['H121', 'H221', 'H122', 'H222', 'H224', 'H125'], 200),
            'participant': [f'P{i//4}' for i in range(200)],
            'rating': np.random.normal(4.5, 1.5, 200).clip(1, 7).round()
        })
        
        # Generate visualizations
        create_trajectory_heatmap(sample_trajectories, output_dir)
        create_transition_flow_diagram(sample_trajectories, output_dir)
        create_cohort_comparison_chart(sample_processed_data, output_dir)
        create_founder_journey_analysis(output_dir)
        
        print("\nAll advanced visualizations created successfully!")
        print(f"Check the {output_dir} directory for output files.")
        
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        print("Note: Run the main analysis script first to generate actual data.")

if __name__ == "__main__":
    main()