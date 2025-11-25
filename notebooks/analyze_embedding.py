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
        
    def plot_grouped_boxplot(self, figsize=(16, 6), save_path=None):
        """
        Create grouped boxplot with k values on x-axis and strategies side-by-side.
        
        Structure:
        - X-axis: Pre-trained | k=0 | k=1 | k=2 | k=3 | k=5 | k=10
        - For k=1 to k=10: two boxplots (random and informed) side-by-side
        - Pre-trained and k=0: single boxplot
        
        Args:
            figsize: Figure size as (width, height)
            save_path: Optional path to save the figure
        """
        if not self.results:
            print("No results to plot. Run analyze_embedding() first.")
            return
        
        # Prepare data for plotting
        plot_data = []
        for (k_value, strategy), distances in self.results.items():
            for dist in distances:
                # Create x-axis label
                if k_value == 'pretrained':
                    x_label = 'Pre-trained'
                    strategy_label = 'none'
                elif k_value == 0:
                    x_label = 'k=0'
                    strategy_label = 'none'
                else:
                    x_label = f'k={k_value}'
                    strategy_label = strategy if strategy else 'none'
                
                plot_data.append({
                    'k_value': x_label,
                    'strategy': strategy_label,
                    'cosine_distance': dist
                })
        
        plot_df = pd.DataFrame(plot_data)
        
        # Define custom order for x-axis
        k_order = ['Pre-trained', 'k=0', 'k=1', 'k=2', 'k=3', 'k=5', 'k=10']
        plot_df['k_value'] = pd.Categorical(plot_df['k_value'], categories=k_order, ordered=True)
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create grouped boxplot
        # We'll use hue for strategy and handle positioning manually
        sns.boxplot(
            data=plot_df, 
            x='k_value', 
            y='cosine_distance',
            hue='strategy',
            hue_order=['none', 'random', 'informed'],
            ax=ax,
            palette=['gray', 'skyblue', 'salmon']
        )
        
        # Customize plot
        ax.set_xlabel('Sampling Configuration', fontsize=13, fontweight='bold')
        ax.set_ylabel('Cosine Distance', fontsize=13, fontweight='bold')
        ax.set_title('Distribution of Cosine Distances: Log Content vs Template\nAcross Different Sampling Strategies and k Values', 
                     fontsize=14, fontweight='bold', pad=20)
        
        # Customize legend
        handles, labels = ax.get_legend_handles_labels()
        legend_labels = []
        for label in labels:
            if label == 'none':
                legend_labels.append('No Strategy')
            elif label == 'random':
                legend_labels.append('Random Sampling')
            elif label == 'informed':
                legend_labels.append('Informed Sampling')
        ax.legend(handles, legend_labels, title='Strategy', title_fontsize=11, fontsize=10)
        
        # Rotate x-axis labels if needed
        plt.xticks(rotation=0, ha='center')
        
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
    
    # Scenario 1: Pre-trained baseline (no fine-tuning)
    # analyzer.analyze_embedding(
    #     'sentence-transformers/all-MiniLM-L6-v2',
    #     k_value='pretrained',
    #     strategy=None
    # )
    
    # # Scenario 2: k=0 (fine-tuned without sampling strategy)
    # analyzer.analyze_embedding(
    #     'swardiantara/MultiSource-full-crk0-m0.5-e5-b128-L6',
    #     k_value=0,
    #     strategy=None
    # )
    
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
    out_dir = 'analysis_results'
    os.makedirs(out_dir, exist_ok=True)
    # Save detailed results
    analyzer.save_results(os.path.join(out_dir, 'log_embedding_cosine_distances.csv'))
    
    # Create grouped boxplot visualization
    analyzer.plot_grouped_boxplot(save_path=os.path.join(out_dir, 'grouped_boxplot_comparison.pdf'))
    
    # Print and save summary statistics
    analyzer.plot_statistics_table(save_path=os.path.join(out_dir, 'summary_statistics.csv'))