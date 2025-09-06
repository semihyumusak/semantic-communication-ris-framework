import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats
from typing import Dict, List, Tuple
import os
from tqdm import tqdm

class MultiUserSemanticAllocation:
    """
    Experiment to validate semantic resource allocation and interference management
    for multi-user scenarios. Addresses Reviewer #1's concern about missing multi-user analysis.
    """
    
    def __init__(self, output_dir="outputs/multi_user_validation"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.n_trials = 500
        self.num_bits = 4000
        
        # Define user types with different semantic priorities
        self.user_types = {
            'critical_medical': {
                'priority_weight': 1.0,
                'min_accuracy': 0.95,
                'description': 'Critical medical data',
                'latency_requirement': 'ultra_low'
            },
            'important_video': {
                'priority_weight': 0.6,
                'min_accuracy': 0.85,
                'description': 'Important video streaming',
                'latency_requirement': 'low'
            },
            'background_data': {
                'priority_weight': 0.2,
                'min_accuracy': 0.70,
                'description': 'Background file transfer',
                'latency_requirement': 'tolerant'
            }
        }
    
    def run_multi_user_experiment(self):
        """
        Compare semantic-aware vs equal allocation across multiple users
        """
        results = []
        snr_points = [-10, 0, 10]
        
        allocation_strategies = {
            'equal_allocation': self._equal_resource_allocation,
            'semantic_aware': self._semantic_aware_allocation,
            'priority_weighted': self._priority_weighted_allocation
        }
        
        for snr_db in snr_points:
            print(f"Testing multi-user allocation at {snr_db} dB...")
            
            for strategy_name, strategy_func in allocation_strategies.items():
                trial_results = []
                
                for trial in tqdm(range(self.n_trials), desc=f"{strategy_name} @ {snr_db}dB"):
                    np.random.seed(trial)
                    torch.manual_seed(trial)
                    
                    # Simulate multi-user scenario
                    user_results = self._simulate_multi_user_scenario(
                        snr_db, strategy_func, trial
                    )
                    
                    # Calculate aggregate metrics
                    aggregate_metrics = self._calculate_aggregate_metrics(user_results)
                    aggregate_metrics.update({
                        'strategy': strategy_name,
                        'snr_db': snr_db,
                        'trial': trial
                    })
                    
                    trial_results.append(aggregate_metrics)
                
                # Statistical analysis for this strategy
                df_trials = pd.DataFrame(trial_results)
                summary_stats = self._calculate_summary_stats(df_trials)
                summary_stats.update({
                    'strategy': strategy_name,
                    'snr_db': snr_db
                })
                
                results.append(summary_stats)
        
        # Save results and create analysis
        results_df = pd.DataFrame(results)
        results_df.to_csv(f"{self.output_dir}/multi_user_allocation_results.csv", index=False)
        
        self._create_multi_user_plots(results_df)
        self._analyze_fairness_efficiency_tradeoff(results_df)
        
        return results_df
    
    def _simulate_multi_user_scenario(self, snr_db: float, allocation_func, trial: int):
        """
        Simulate a single multi-user transmission scenario
        """
        # Create users with different semantic priorities
        users = {}
        total_bandwidth = 100.0  # MHz (example)
        total_power = 1.0  # Normalized
        
        for user_id, user_type in enumerate(['critical_medical', 'important_video', 'background_data']):
            users[f'user_{user_id}'] = {
                'type': user_type,
                'data': torch.randint(0, 2, (self.num_bits,)),
                'priority': self.user_types[user_type]['priority_weight'],
                'min_accuracy': self.user_types[user_type]['min_accuracy']
            }
        
        # Apply resource allocation strategy
        resource_allocation = allocation_func(users, total_bandwidth, total_power)
        
        # Simulate transmission for each user
        user_results = {}
        for user_id, user_data in users.items():
            allocated_bw = resource_allocation[user_id]['bandwidth']
            allocated_power = resource_allocation[user_id]['power']
            
            # Calculate effective SNR based on allocation
            effective_snr = snr_db + 10 * np.log10(allocated_power) + 10 * np.log10(allocated_bw / 20.0)
            
            # Add interference from other users
            interference_power = sum(
                resource_allocation[other_user]['power'] 
                for other_user in users.keys() 
                if other_user != user_id
            ) * 0.1  # 10% interference coupling
            
            sinr = effective_snr - 10 * np.log10(1 + interference_power)
            
            # Simulate semantic communication performance
            accuracy = self._semantic_performance_model(sinr, user_data['type'])
            
            user_results[user_id] = {
                'user_type': user_data['type'],
                'priority': user_data['priority'],
                'allocated_bandwidth': allocated_bw,
                'allocated_power': allocated_power,
                'effective_sinr': sinr,
                'semantic_accuracy': accuracy,
                'meets_requirement': accuracy >= user_data['min_accuracy'],
                'semantic_utility': self._calculate_semantic_utility(accuracy, user_data['priority'])
            }
        
        return user_results
    
    def _equal_resource_allocation(self, users: Dict, total_bw: float, total_power: float):
        """Equal resource allocation (baseline)"""
        n_users = len(users)
        allocation = {}
        
        for user_id in users.keys():
            allocation[user_id] = {
                'bandwidth': total_bw / n_users,
                'power': total_power / n_users
            }
        
        return allocation
    
    def _semantic_aware_allocation(self, users: Dict, total_bw: float, total_power: float):
        """Semantic-aware allocation based on content importance and requirements"""
        allocation = {}
        
        # Calculate priority-weighted allocation
        total_priority = sum(users[uid]['priority'] for uid in users.keys())
        
        for user_id, user_data in users.items():
            priority_ratio = user_data['priority'] / total_priority
            
            # Allocate more resources to higher priority users
            base_bw = total_bw * priority_ratio
            base_power = total_power * priority_ratio
            
            # Adjust based on minimum accuracy requirements
            min_acc = user_data['min_accuracy']
            if min_acc > 0.9:  # Critical users get extra resources
                boost_factor = 1.3
            elif min_acc > 0.8:  # Important users get moderate boost
                boost_factor = 1.1
            else:  # Background users use base allocation
                boost_factor = 0.8
            
            allocation[user_id] = {
                'bandwidth': base_bw * boost_factor,
                'power': base_power * boost_factor
            }
        
        # Normalize to ensure total doesn't exceed limits
        total_alloc_bw = sum(alloc['bandwidth'] for alloc in allocation.values())
        total_alloc_power = sum(alloc['power'] for alloc in allocation.values())
        
        for user_id in allocation.keys():
            allocation[user_id]['bandwidth'] *= total_bw / total_alloc_bw
            allocation[user_id]['power'] *= total_power / total_alloc_power
        
        return allocation
    
    def _priority_weighted_allocation(self, users: Dict, total_bw: float, total_power: float):
        """Pure priority-weighted allocation"""
        allocation = {}
        total_priority = sum(users[uid]['priority'] for uid in users.keys())
        
        for user_id, user_data in users.items():
            priority_ratio = user_data['priority'] / total_priority
            allocation[user_id] = {
                'bandwidth': total_bw * priority_ratio,
                'power': total_power * priority_ratio
            }
        
        return allocation
    
    def _semantic_performance_model(self, sinr_db: float, user_type: str):
        """Model semantic communication performance based on SINR and user type"""
        # Base performance from SINR
        base_acc = 1 / (1 + np.exp(-(sinr_db + 5) / 3))
        
        # Semantic processing provides different benefits for different content types
        semantic_benefits = {
            'critical_medical': 0.15,    # High redundancy for critical data
            'important_video': 0.10,     # Moderate semantic benefits
            'background_data': 0.05      # Lower semantic processing
        }
        
        semantic_boost = semantic_benefits.get(user_type, 0.05) * (1 - base_acc)
        final_acc = base_acc + semantic_boost
        
        # Add realistic noise
        final_acc += np.random.normal(0, 0.02)
        return np.clip(final_acc, 0.0, 1.0)
    
    def _calculate_semantic_utility(self, accuracy: float, priority: float):
        """Calculate semantic utility combining accuracy and priority"""
        # Utility function: weighted accuracy with priority scaling
        base_utility = accuracy
        priority_scaling = 1 + (priority - 0.5) * 0.5  # Scale by priority
        return base_utility * priority_scaling
    
    def _calculate_aggregate_metrics(self, user_results: Dict):
        """Calculate system-wide metrics from individual user results"""
        accuracies = [result['semantic_accuracy'] for result in user_results.values()]
        utilities = [result['semantic_utility'] for result in user_results.values()]
        requirements_met = [result['meets_requirement'] for result in user_results.values()]
        
        # Aggregate metrics
        metrics = {
            'avg_semantic_accuracy': np.mean(accuracies),
            'min_semantic_accuracy': np.min(accuracies),
            'total_semantic_utility': np.sum(utilities),
            'fairness_index': self._calculate_fairness_index(utilities),
            'requirements_satisfaction': np.mean(requirements_met),
            'critical_user_accuracy': user_results['user_0']['semantic_accuracy'],  # Medical user
            'system_efficiency': np.sum(utilities) / len(utilities)
        }
        
        return metrics
    
    def _calculate_fairness_index(self, utilities: List[float]):
        """Jain's fairness index for resource allocation"""
        if not utilities or len(utilities) == 0:
            return 0.0
        
        sum_util = sum(utilities)
        sum_squared = sum(u**2 for u in utilities)
        n = len(utilities)
        
        if sum_squared == 0:
            return 1.0
        
        return (sum_util**2) / (n * sum_squared)
    
    def _calculate_summary_stats(self, df_trials: pd.DataFrame):
        """Calculate summary statistics for multiple trials"""
        metrics = ['avg_semantic_accuracy', 'total_semantic_utility', 'fairness_index', 
                  'requirements_satisfaction', 'critical_user_accuracy']
        
        summary = {}
        for metric in metrics:
            values = df_trials[metric].values
            mean_val = np.mean(values)
            std_val = np.std(values, ddof=1)
            ci = stats.t.interval(0.95, len(values)-1, loc=mean_val, scale=stats.sem(values))
            
            summary.update({
                f'{metric}_mean': mean_val,
                f'{metric}_std': std_val,
                f'{metric}_ci_lower': ci[0],
                f'{metric}_ci_upper': ci[1]
            })
        
        return summary
    
    def _create_multi_user_plots(self, results_df: pd.DataFrame):
        """Create visualization plots for multi-user results"""
        
        # Plot 1: Semantic Utility vs SNR for different strategies
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        strategies = results_df['strategy'].unique()
        snr_points = sorted(results_df['snr_db'].unique())
        colors = {'equal_allocation': 'blue', 'semantic_aware': 'red', 'priority_weighted': 'green'}
        
        for strategy in strategies:
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for snr in snr_points:
                snr_data = strategy_data[strategy_data['snr_db'] == snr]
                if len(snr_data) > 0:
                    means.append(snr_data['total_semantic_utility_mean'].values[0])
                    ci_lowers.append(snr_data['total_semantic_utility_ci_lower'].values[0])
                    ci_uppers.append(snr_data['total_semantic_utility_ci_upper'].values[0])
            
            ax1.errorbar(snr_points, means, 
                        yerr=[np.array(means) - np.array(ci_lowers), 
                              np.array(ci_uppers) - np.array(means)],
                        label=strategy.replace('_', ' ').title(),
                        marker='o', capsize=5, color=colors.get(strategy, 'black'))
        
        ax1.set_xlabel('SNR (dB)')
        ax1.set_ylabel('Total Semantic Utility')
        ax1.set_title('Multi-User Semantic Utility vs SNR')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fairness Index comparison
        for strategy in strategies:
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for snr in snr_points:
                snr_data = strategy_data[strategy_data['snr_db'] == snr]
                if len(snr_data) > 0:
                    means.append(snr_data['fairness_index_mean'].values[0])
                    ci_lowers.append(snr_data['fairness_index_ci_lower'].values[0])
                    ci_uppers.append(snr_data['fairness_index_ci_upper'].values[0])
            
            ax2.errorbar(snr_points, means,
                        yerr=[np.array(means) - np.array(ci_lowers), 
                              np.array(ci_uppers) - np.array(means)],
                        label=strategy.replace('_', ' ').title(),
                        marker='s', capsize=5, color=colors.get(strategy, 'black'))
        
        ax2.set_xlabel('SNR (dB)')
        ax2.set_ylabel('Fairness Index')
        ax2.set_title('Resource Allocation Fairness')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/multi_user_allocation_comparison.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 3: Critical User Performance
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for strategy in strategies:
            strategy_data = results_df[results_df['strategy'] == strategy]
            
            means = []
            ci_lowers = []
            ci_uppers = []
            
            for snr in snr_points:
                snr_data = strategy_data[strategy_data['snr_db'] == snr]
                if len(snr_data) > 0:
                    means.append(snr_data['critical_user_accuracy_mean'].values[0])
                    ci_lowers.append(snr_data['critical_user_accuracy_ci_lower'].values[0])
                    ci_uppers.append(snr_data['critical_user_accuracy_ci_upper'].values[0])
            
            ax.errorbar(snr_points, [m*100 for m in means],
                       yerr=[(np.array(means) - np.array(ci_lowers))*100, 
                             (np.array(ci_uppers) - np.array(means))*100],
                       label=strategy.replace('_', ' ').title(),
                       marker='o', capsize=5, color=colors.get(strategy, 'black'))
        
        ax.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical Requirement (95%)')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Critical User Accuracy (%)')
        ax.set_title('Critical User Performance by Allocation Strategy')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/critical_user_performance.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_fairness_efficiency_tradeoff(self, results_df: pd.DataFrame):
        """Analyze the trade-off between fairness and efficiency"""
        
        tradeoff_analysis = []
        
        for _, row in results_df.iterrows():
            tradeoff_analysis.append({
                'strategy': row['strategy'],
                'snr_db': row['snr_db'],
                'efficiency': row['total_semantic_utility_mean'],
                'fairness': row['fairness_index_mean'],
                'critical_satisfaction': row['critical_user_accuracy_mean'] >= 0.95
            })
        
        tradeoff_df = pd.DataFrame(tradeoff_analysis)
        tradeoff_df.to_csv(f"{self.output_dir}/fairness_efficiency_tradeoff.csv", index=False)
        
        # Generate summary report
        with open(f"{self.output_dir}/multi_user_analysis_report.md", 'w') as f:
            f.write("# Multi-User Semantic Resource Allocation Analysis\n\n")
            f.write("## Key Findings\n\n")
            
            # Find best performing strategy
            best_efficiency = tradeoff_df.loc[tradeoff_df['efficiency'].idxmax()]
            best_fairness = tradeoff_df.loc[tradeoff_df['fairness'].idxmax()]
            
            f.write(f"- **Best Efficiency**: {best_efficiency['strategy']} at {best_efficiency['snr_db']} dB "
                   f"(Utility: {best_efficiency['efficiency']:.3f})\n")
            f.write(f"- **Best Fairness**: {best_fairness['strategy']} at {best_fairness['snr_db']} dB "
                   f"(Fairness: {best_fairness['fairness']:.3f})\n")
            
            # Critical user analysis
            critical_success = tradeoff_df.groupby('strategy')['critical_satisfaction'].mean()
            f.write(f"\n## Critical User Requirement Satisfaction\n\n")
            for strategy, success_rate in critical_success.items():
                f.write(f"- **{strategy.replace('_', ' ').title()}**: {success_rate*100:.1f}% success rate\n")
        
        print(f"Multi-user analysis complete. Results saved to {self.output_dir}")
        return tradeoff_df

def main():
    """Run multi-user semantic allocation experiment"""
    experiment = MultiUserSemanticAllocation()
    results = experiment.run_multi_user_experiment()
    
    print("\nMulti-User Experiment Summary:")
    print(f"- Tested {len(results)} allocation strategy/SNR combinations")
    print(f"- Results saved to {experiment.output_dir}")
    print("\nKey findings available in multi_user_analysis_report.md")

if __name__ == "__main__":
    main()
