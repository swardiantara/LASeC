import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

class LogEmbeddingAnalyzer:
    """
    Analyze log messages by computing cosine distances between 
    log content and their corresponding templates using various embeddings.
    """
    
    def __init__(self, csv_path):
        """
        Initialize with log data from CSV file.
        
        Args:
            csv_path: Path to CSV file with columns: Source, Content, EventId, EventTemplate
        """
        self.df = pd.read_csv(csv_path)
        self.results = {}
        
    def compute_cosine_distance(self, embeddings1, embeddings2):
        """
        Compute cosine distance (1 - cosine_similarity) between two sets of embeddings.
        
        Args:
            embeddings1: numpy array of shape (n_samples, embedding_dim)
            embeddings2: numpy array of shape (n_samples, embedding_dim)
            
        Returns:
            numpy array of cosine distances
        """
        # Compute cosine similarity for each pair
        cos_sim = np.array([
            cosine_similarity([emb1], [emb2])[0][0] 
            for emb1, emb2 in zip(embeddings1, embeddings2)
        ])
        # Convert to cosine distance
        cos_distance = 1 - cos_sim
        return cos_distance
    
    def analyze_embedding(self, model_name_or_path, k_value, strategy=None):
        """
        Analyze logs using a specific embedding model.
        
        Args:
            model_name_or_path: HuggingFace model name or path to fine-tuned model
            k_value: The k hyperparameter (0, 1, 2, 3, 5, 10, or 'pretrained')
            strategy: Sampling strategy ('random', 'informed', or None for pretrained/k=0)
            
        Returns:
            numpy array of cosine distances
        """
        print(f"Loading model: {model_name_or_path} (k={k_value}, strategy={strategy})")
        model = SentenceTransformer(model_name_or_path)
        
        # Encode log contents and templates
        print("Encoding log contents...")
        content_embeddings = model.encode(
            self.df['Content'].tolist(), 
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print("Encoding templates...")
        template_embeddings = model.encode(
            self.df['EventTemplate'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Compute cosine distances
        print("Computing cosine distances...")
        distances = self.compute_cosine_distance(content_embeddings, template_embeddings)
        
        # Store results with structured key
        result_key = (k_value, strategy)
        self.results[result_key] = distances
        
        print(f"Statistics for k={k_value}, strategy={strategy}:")
        print(f"  Mean distance: {np.mean(distances):.4f}")
        print(f"  Median distance: {np.median(distances):.4f}")
        print(f"  Std distance: {np.std(distances):.4f}")
        print(f"  Min distance: {np.min(distances):.4f}")
        print(f"  Max distance: {np.max(distances):.4f}\n")
        
        return distances
    
    def save_results(self, output_path):
        """
        Save computed distances to CSV file in long format for easy plotting.
        
        Args:
            output_path: Path to save CSV file
        """
        rows = []
        for (k_value, strategy), distances in self.results.items():
            for i, dist in enumerate(distances):
                rows.append({
                    'log_index': i,
                    'k_value': k_value,
                    'strategy': strategy if strategy else 'none',
                    'cosine_distance': dist,
                    'Source': self.df.iloc[i]['Source'],
                    'EventId': self.df.iloc[i]['EventId']
                })
        
        results_df = pd.DataFrame(rows)
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")
        return results_df
        
    def plot_grouped_boxplot(self, figsize=(6, 4), save_path=None):
        """
        Create grouped boxplot with k values on x-axis and strategies side-by-side.
        
        Structure:
        - X-axis: Pre-trained | k=0 | k=1 | k=2 | k=3 | k=5 | k=10
        - For k=1 to k=10: two boxplots (random and informed) side-by-side
        - Pre-trained and k=0: single boxplot centered
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        if not self.results:
            print("No results to plot. Run analyze_embedding() first.")
            return
        
        # Prepare data organized by k_value and strategy
        data_dict = {}
        for (k_value, strategy), distances in self.results.items():
            if k_value == 'pretrained':
                k_label = 'Pre-trained'
            elif k_value == 0:
                k_label = 'k=0'
            else:
                k_label = f'k={k_value}'
            
            if k_label not in data_dict:
                data_dict[k_label] = {}
            
            strategy_key = strategy if strategy else 'none'
            data_dict[k_label][strategy_key] = distances
        
        # Define order for x-axis
        k_order = ['Pre-trained', 'k=0', 'k=1', 'k=2', 'k=3', 'k=5', 'k=10']
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Define colors
        colors = {'none': 'gray', 'random': 'skyblue', 'informed': 'salmon'}
        
        # Define box width and spacing
        box_width = 0.3
        group_spacing = 0.5
        
        # Manual positioning
        positions = []
        box_data = []
        box_colors = []
        x_ticks = []
        x_labels = []
        
        for i, k_label in enumerate(k_order):
            if k_label not in data_dict:
                continue
            
            strategies_data = data_dict[k_label]
            
            if k_label in ['Pre-trained', 'k=0']:
                # Single centered boxplot
                pos = i * group_spacing
                positions.append(pos)
                box_data.append(strategies_data['none'])
                box_colors.append(colors['none'])
                x_ticks.append(pos)
                x_labels.append(k_label)
            else:
                # Two side-by-side boxplots
                center_pos = i * group_spacing
                
                # Random on the left
                if 'random' in strategies_data:
                    pos_random = center_pos - box_width/2
                    positions.append(pos_random)
                    box_data.append(strategies_data['random'])
                    box_colors.append(colors['random'])
                
                # Informed on the right
                if 'informed' in strategies_data:
                    pos_informed = center_pos + box_width/2
                    positions.append(pos_informed)
                    box_data.append(strategies_data['informed'])
                    box_colors.append(colors['informed'])
                
                x_ticks.append(center_pos)
                x_labels.append(k_label)
        
        # Create boxplots
        bp = ax.boxplot(box_data, positions=positions, widths=box_width,
                        patch_artist=True, showfliers=True,
                        boxprops=dict(linewidth=1.5),
                        medianprops=dict(linewidth=2, color='darkred'),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5))
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], box_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Set x-axis
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        
        # Customize plot
        ax.set_xlabel('Sampling Configuration', fontsize=11, fontweight='bold')
        ax.set_ylabel('Cosine Distance', fontsize=11, fontweight='bold')
        # ax.set_title('Distribution of Cosine Distances: Log Content vs Template\nAcross Different Sampling Strategies and k Values', 
                    #  fontsize=14, fontweight='bold', pad=20)
        
        # Create custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='gray', alpha=0.7, label='No Strategy'),
            Patch(facecolor='skyblue', alpha=0.7, label='Random Sampling'),
            Patch(facecolor='salmon', alpha=0.7, label='Informed Sampling')
        ]
        ax.legend(handles=legend_elements, title='Strategy', 
                 title_fontsize=11, fontsize=10, loc='best')
        
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_statistics_table(self, save_path=None):
        """
        Create a summary statistics table for all scenarios.
        
        Args:
            save_path: Optional path to save the table as CSV
        """
        stats_data = []
        
        for (k_value, strategy), distances in self.results.items():
            stats_data.append({
                'k_value': k_value,
                'strategy': strategy if strategy else 'none',
                'mean': np.mean(distances),
                'median': np.median(distances),
                'std': np.std(distances),
                'min': np.min(distances),
                'max': np.max(distances),
                'q25': np.percentile(distances, 25),
                'q75': np.percentile(distances, 75)
            })
        
        stats_df = pd.DataFrame(stats_data)
        stats_df = stats_df.sort_values(['k_value', 'strategy'])
        
        print("\n" + "="*80)
        print("SUMMARY STATISTICS")
        print("="*80)
        print(stats_df.to_string(index=False))
        print("="*80 + "\n")
        
        if save_path:
            stats_df.to_csv(save_path, index=False)
            print(f"Statistics table saved to {save_path}")
        
        return stats_df


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with your log data
    analyzer = LogEmbeddingAnalyzer(os.path.join('dataset', 'MultiUnique_2k.log_structured.csv'))
    out_dir = 'analysis_results'
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, 'cosine_distances_all_scenarios.csv')

    if os.path.exists(result_path):
        print(f"Loading existing results from {result_path}")
        results_df = pd.read_csv(result_path)
        # Reconstruct results dictionary
        for _, row in results_df.iterrows():
            key = (row['k_value'], row['strategy'] if row['strategy'] != 'none' else None)
            if key not in analyzer.results:
                analyzer.results[key] = []
            analyzer.results[key].append(row['cosine_distance'])
        # Convert lists to numpy arrays
        for key in analyzer.results:
            analyzer.results[key] = np.array(analyzer.results[key])
    else:   
        # Scenario 1: Pre-trained baseline (no fine-tuning)
        analyzer.analyze_embedding(
            'sentence-transformers/all-MiniLM-L6-v2',
            k_value='pretrained',
            strategy=None
        )
        
        # Scenario 2: k=0 (fine-tuned without sampling strategy)
        analyzer.analyze_embedding(
            'swardiantara/MultiSource-full-crk0-m0.5-e5-b128-L6',
            k_value=0,
            strategy=None
        )
        
        # Scenarios 3-12: k=1,2,3,5,10 with random and informed strategies
        k_values = [1, 2, 3, 5, 10]
        strategies = ['random', 'informed']
        
        for k in k_values:
            for strategy in strategies:
                strategy_suffix = 'r' if strategy == 'random' else 'd'
                model_name = f'MultiSource-full-c{strategy_suffix}k{k}-m0.5-e5-b128-L6'
                analyzer.analyze_embedding(
                    f'swardiantara/{model_name}',
                    k_value=k,
                    strategy=strategy
                )

        # Save detailed results
        analyzer.save_results(result_path)
    
    # Create grouped boxplot visualization
    analyzer.plot_grouped_boxplot(save_path=os.path.join(out_dir, 'grouped_boxplot_comparison.pdf'))
    
    # Print and save summary statistics
    analyzer.plot_statistics_table(save_path=os.path.join(out_dir, 'summary_statistics.csv'))