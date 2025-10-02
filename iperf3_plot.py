#!/usr/bin/env python3
"""
Plot iperf3 data with improved performance and flexibility.

Enhanced by: Mohamed Amine Nasr  
Original Author: hchiuzhuo (https://github.com/hchiuzhuo/iperfPlotter/blob/master/iperf3_plot.py)

This module provides tools for parsing iperf3 JSON output files and 
generating publication-quality bandwidth plots with customizable styling.
"""

import json
import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class IperfPlotter:
    """
    Handles all plotting operations for iperf3 data
    
    This class provides methods to create various types of bandwidth plots
    from iperf3 data, including individual stream plots, aggregate throughput
    plots, and combined visualizations.
    
    Attributes:
        TABLEAU20 (list): RGB color palette for consistent plot styling
        bounds (list): List of boundary tuples (upper, lower, tag) for reference lines
        colors (list): Normalized RGB colors for matplotlib
    """
    
    TABLEAU20 = [
        (31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
        (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
        (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
        (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
        (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)
    ]
    
    def __init__(self, bounds=None):
        """
        Initialize the plotter with optional boundary lines
        
        Args:
            bounds (list, optional): List of tuples (upper, lower, tag) defining
                reference lines to display on plots. Defaults to None.
        """
        sns.set_style("white")
        self.bounds = bounds or []
        self.colors = [(r/255., g/255., b/255.) for r, g, b in self.TABLEAU20]
    
    def _setup_plot_style(self, ax, y_min, y_max, x_min, x_max, y_level, unit):
        """
        Configure common plot styling including axes, labels, and grid
        
        Args:
            ax (matplotlib.axes.Axes): The axes object to style
            y_min (float): Minimum y-axis value
            y_max (float): Maximum y-axis value
            x_min (float): Minimum x-axis value
            x_max (float): Maximum x-axis value
            y_level (float): Spacing between y-axis tick marks
            unit (str): Bandwidth unit label for y-axis
        """
        for spine in ['top', 'bottom', 'right', 'left']:
            ax.spines[spine].set_visible(False)
        
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)
        plt.yticks(np.arange(y_min, y_max, y_level),
                   [str(round(x, 1)) for x in np.arange(y_min, y_max, y_level)], 
                   fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel('Time (seconds)', fontsize=12)
        plt.ylabel(f'Bandwidth ({unit})', fontsize=12)
        
        # Grid lines
        for y in np.arange(y_min, y_max, y_level):
            plt.plot([x_min, x_max], [y, y], "--", lw=0.5, color="gray", alpha=0.3)
        
        plt.tick_params(axis="both", which="both", bottom=False, top=False,
                       labelbottom=True, left=False, right=False, labelleft=True)
    
    def _calculate_plot_dimensions(self, dataframe):
        """
        Calculate optimal plot dimensions from a DataFrame
        
        Determines appropriate axis ranges and tick spacing based on the
        data distribution to ensure readable and aesthetically pleasing plots.
        
        Args:
            dataframe (pd.DataFrame or pd.Series): Input data to analyze
            
        Returns:
            tuple: (y_min, y_max, x_min, x_max, y_level) where:
                - y_min (float): Minimum y-axis value
                - y_max (float): Maximum y-axis value
                - x_min (float): Minimum x-axis value (time)
                - x_max (float): Maximum x-axis value (time)
                - y_level (float): Spacing between y-axis ticks
        """
        values = dataframe.values if hasattr(dataframe, 'values') else dataframe
        
        y_level = (values.max() - values.min()) / 20
        if y_level == 0:
            y_level = values.max() * 0.05  # 5% of max if no variation
        
        y_max = values.max() + y_level
        y_min = max(0, values.min() - y_level/2)
        x_max = dataframe.index[-1]
        x_min = dataframe.index[0]
        
        return y_min, y_max, x_min, x_max, y_level
    
    def _add_bounds(self, x_max, y_level):
        """
        Add boundary reference lines to the current plot
        
        Draws horizontal lines representing expected upper and lower performance
        thresholds with labels.
        
        Args:
            x_max (float): Maximum x-axis value for label positioning
            y_level (float): Y-axis spacing for label offset calculation
        """
        for j, (upper, lower, tag) in enumerate(self.bounds):
            color = self.colors[j % len(self.colors)]
            if upper > 0:
                plt.axhline(y=upper, color=color, linestyle='--', linewidth=2, alpha=0.8)
                plt.text(x_max * 0.02, upper + y_level/4, 
                        f'{tag} (Upper: {upper})', fontsize=10, color=color)
            if lower > 0:
                plt.axhline(y=lower, color=color, linestyle='--', linewidth=2, alpha=0.8)
                plt.text(x_max * 0.02, lower - y_level/4, 
                        f'{tag} (Lower: {lower})', fontsize=10, color=color)
    
    def plot_streams_only(self, dataset, filename, desc, title, unit='Mbps'):
        """
        Create a plot showing only individual stream bandwidths
        
        Generates a line plot with one line per iperf3 stream, excluding
        the total aggregate throughput.
        
        Args:
            dataset (pd.DataFrame): DataFrame containing stream bandwidth data
            filename (str): Output file path for the plot
            desc (str): Description text to display on the plot
            title (str): Plot title
            unit (str, optional): Bandwidth unit (Mbps or MBps). Defaults to 'Mbps'.
        """
        stream_cols = [col for col in dataset.columns if col != 'Total_Sum']
        stream_data = dataset[stream_cols]
        
        y_min, y_max, x_min, x_max, y_level = self._calculate_plot_dimensions(stream_data)
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        self._setup_plot_style(ax, y_min, y_max, x_min, x_max, y_level, f'{unit} per Stream')
        
        # Plot streams
        for i, col in enumerate(stream_cols):
            plt.plot(stream_data.index, stream_data[col].values, 
                    lw=2, color=self.colors[i % len(self.colors)], label=col)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)
        self._add_bounds(x_max, y_level)
        
        plt.title(f"{title} - Individual Streams", fontsize=16, pad=20)
        plt.text(x_max * 0.98, y_max * 0.95, f"{desc} - Individual streams", 
                fontsize=10, ha='right', va='top', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
    
    def plot_sum_only(self, dataset, filename, desc, title, unit='Mbps'):
        """
        Create a plot showing only total aggregate throughput
        
        Generates a line plot displaying the sum of all stream bandwidths
        over time, without individual stream details.
        
        Args:
            dataset (pd.DataFrame): DataFrame containing Total_Sum column
            filename (str): Output file path for the plot
            desc (str): Description text to display on the plot
            title (str): Plot title
            unit (str, optional): Bandwidth unit (Mbps or MBps). Defaults to 'Mbps'.
            
        Raises:
            ValueError: If Total_Sum column is not present in dataset
        """
        if 'Total_Sum' not in dataset.columns:
            raise ValueError("No Total_Sum column found in dataset")
        
        sum_data = dataset[['Total_Sum']]
        y_min, y_max, x_min, x_max, y_level = self._calculate_plot_dimensions(sum_data)
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        self._setup_plot_style(ax, y_min, y_max, x_min, x_max, y_level, f'Total {unit}')
        
        plt.plot(sum_data.index, sum_data['Total_Sum'].values, 
                lw=4, color='darkblue', label='Total Sum')
        
        plt.legend(loc='upper right', fontsize=12)
        self._add_bounds(x_max, y_level)
        
        plt.title(f"{title} - Total", fontsize=16, pad=20)
        plt.text(x_max * 0.98, y_max * 0.95, f"{desc} - Aggregate throughput", 
                fontsize=10, ha='right', va='top', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
    
    def plot_streams_with_sum(self, dataset, filename, desc, title, unit='Mbps'):
        """
        Create a combined plot showing both individual streams and total
        
        Generates a line plot with individual stream bandwidths (lighter lines)
        and the aggregate total (bold black line) on the same axes.
        
        Args:
            dataset (pd.DataFrame): DataFrame containing stream and Total_Sum data
            filename (str): Output file path for the plot
            desc (str): Description text to display on the plot
            title (str): Plot title
            unit (str, optional): Bandwidth unit (Mbps or MBps). Defaults to 'Mbps'.
        """
        y_min, y_max, x_min, x_max, y_level = self._calculate_plot_dimensions(dataset)
        
        plt.figure(figsize=(14, 8))
        ax = plt.subplot(111)
        self._setup_plot_style(ax, y_min, y_max, x_min, x_max, y_level, unit)
        
        stream_cols = [col for col in dataset.columns if col != 'Total_Sum']
        
        # Plot streams
        for i, col in enumerate(stream_cols):
            plt.plot(dataset.index, dataset[col].values, 
                    lw=1.5, color=self.colors[i % len(self.colors)], alpha=0.7, label=col)
        
        # Plot sum
        if 'Total_Sum' in dataset.columns:
            plt.plot(dataset.index, dataset['Total_Sum'].values, 
                    lw=3, color='black', label='Total Sum')
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        self._add_bounds(x_max, y_level)
        
        plt.title(title, fontsize=16, pad=20)
        plt.text(x_max * 0.98, y_max * 0.95, desc, fontsize=10, 
                ha='right', va='top', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()


class IperfDataParser:
    """
    Handles parsing iperf3 JSON data and command-line options
    
    This class manages the configuration, file discovery, and data extraction
    from iperf3 JSON output files. It provides methods to parse command-line
    arguments and convert raw JSON data into plottable DataFrames.
    
    Attributes:
        foldername (str): Path to folder containing iperf3 JSON files
        plot_files (list): List of file paths to plot
        no_plot_files (list): List of files to exclude from plotting
        output (str): Output filename for main plot
        sum_output (str): Output filename for sum plot
        bounds (list): List of boundary tuples for reference lines
        unit (str): Bandwidth unit (Mbps or MBps)
        plot_style (str): Plot style (separate or combined)
        title (str): Custom plot title
    """
    
    def __init__(self):
        """Initialize the parser with default configuration values"""
        self.foldername = None
        self.plot_files = []
        self.no_plot_files = []
        self.output = "iperf.png"
        self.sum_output = ""
        self.bounds = []
        self.unit = 'Mbps'
        self.plot_style = 'separate'
        self.title = None
    
    def parse_options(self, args):
        """
        Parse command-line options and update configuration
        
        Processes command-line arguments to configure the plotter's behavior,
        including input files, output files, boundaries, and styling options.
        
        Args:
            args (list): List of command-line argument strings (typically sys.argv[1:])
            
        Returns:
            IperfDataParser: Self reference for method chaining
            
        Raises:
            SystemExit: If required arguments are missing or invalid
        """
        parser = OptionParser(usage='%prog -f FOLDER [options]')
        
        parser.add_option('-f', '--folder', dest='foldername',
                         help='Input folder absolute path [required]')
        parser.add_option('-o', '--output', dest='output', default="iperf.png",
                         help='Output plot filename (default: iperf.png)')
        parser.add_option('-s', '--sum-output', dest='sumOutput', default="",
                         help='Sum plot filename (optional, auto-generated if not specified)')
        parser.add_option('-p', '--plotfiles', dest='plotFiles', default="",
                         help='Comma-separated list of files to plot (default: all JSON files)')
        parser.add_option('-n', '--noPlotFiles', dest='noPlotFiles', default="",
                         help='Comma-separated list of files to exclude from plotting')
        parser.add_option('-u', '--upperLimit', type='float', dest='upperLimit', default=0,
                         help='Expected upper boundary')
        parser.add_option('-l', '--lowerLimit', type='float', dest='lowerLimit', default=0,
                         help='Expected lower boundary')
        parser.add_option('-b', '--bound', dest='bound', default="",
                         help='Multiple bounds: [upper,lower,tag][upper,lower,tag]...')
        parser.add_option('--unit', dest='unit', default='Mbps',
                         help='Bandwidth unit: Mbps or MBps (default: Mbps)')
        parser.add_option('--plot-style', dest='plotStyle', default='separate',
                         help='Plot style: separate, combined (default: separate)')
        parser.add_option('-t', '--title', dest='title', default="",
                         help='Custom plot title (default: auto-generated)')
        parser.add_option('-v', '--verbose', dest='verbose', action='store_true',
                         help='Verbose output')
        
        options, _ = parser.parse_args(args)
        
        if not options.foldername:
            parser.error('Folder path (-f) is required')
        
        self.foldername = options.foldername
        self.plot_files = [f.strip() for f in options.plotFiles.split(',') if f.strip()]
        self.no_plot_files = [f.strip() for f in options.noPlotFiles.split(',') if f.strip()]
        self.output = options.output
        self.sum_output = options.sumOutput
        self.unit = options.unit
        self.plot_style = options.plotStyle
        self.title = options.title if options.title else None
        
        # Parse bounds
        if options.bound:
            for bound_str in options.bound.replace('[', '').split(']'):
                parts = [x.strip() for x in bound_str.split(',') if x.strip()]
                if len(parts) == 3:
                    self.bounds.append([float(parts[0]), float(parts[1]), parts[2]])
        
        # Add simple bounds if specified
        if options.upperLimit > 0 or options.lowerLimit > 0:
            self.bounds.append([options.upperLimit, options.lowerLimit, 'Threshold'])
        
        return self
    
    def get_plot_files(self):
        """
        Get list of files to plot based on configuration
        
        Discovers JSON files in the configured folder, applies include/exclude
        filters, and returns the final list of files to process.
        
        Returns:
            list: List of absolute file paths to plot
            
        Raises:
            ValueError: If folder path is not set
        """
        if not self.foldername:
            raise ValueError("Folder path not set")
        
        if not self.plot_files:
            files = []
            for root, _, filenames in os.walk(self.foldername):
                files.extend([os.path.join(root, f) for f in filenames if f.endswith('.json')])
            self.plot_files = files
        else:
            self.plot_files = [
                f if os.path.isabs(f) else os.path.join(self.foldername, f) 
                for f in self.plot_files
            ]
        
        if self.no_plot_files:
            self.plot_files = [
                f for f in self.plot_files 
                if os.path.basename(f) not in self.no_plot_files
            ]
        
        return self.plot_files
    
    def _generate_bandwidth(self, iperf_data, unit='Mbps'):
        """
        Generate bandwidth time series from iperf3 JSON data
        
        Extracts bandwidth measurements from iperf3 JSON output and converts
        them into pandas Series objects organized by stream.
        
        Args:
            iperf_data (dict): Parsed iperf3 JSON data
            unit (str, optional): Target bandwidth unit (Mbps or MBps). Defaults to 'Mbps'.
            
        Returns:
            dict: Dictionary mapping stream names to pandas Series objects.
                Keys include 'Stream_1', 'Stream_2', etc., and 'Total_Sum'.
        """
        duration = iperf_data.get('start', {}).get('test_start', {}).get('duration', 0)
        divisor = 1024 * 1024 * 8 if unit == 'MBps' else 1024 * 1024
        
        streams_data = {}
        sum_data = {}
        
        for interval in iperf_data.get('intervals', []):
            streams = interval.get('streams', [])
            if not streams:
                continue
            
            time_point = round(float(streams[0].get('start', 0)), 0)
            if time_point > duration:
                continue
            
            interval_sum = 0
            for idx, stream in enumerate(streams):
                stream_name = f"Stream_{idx + 1}"
                bw_value = round(float(stream.get('bits_per_second', 0)) / divisor, 3)
                
                if stream_name not in streams_data:
                    streams_data[stream_name] = {'idx': [], 'values': []}
                
                streams_data[stream_name]['idx'].append(time_point)
                streams_data[stream_name]['values'].append(bw_value)
                interval_sum += bw_value
            
            sum_data[time_point] = interval_sum
        
        # Convert to pandas Series
        result = {
            name: pd.Series(data['values'], index=data['idx'])
            for name, data in streams_data.items()
        }
        
        if sum_data:
            result['Total_Sum'] = pd.Series(list(sum_data.values()), index=list(sum_data.keys()))
        
        return result
    
    def get_dataset(self):
        """
        Load and combine bandwidth data from all configured files
        
        Reads all JSON files, extracts bandwidth data, aligns time series,
        and combines them into a single DataFrame ready for plotting.
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns for each stream
                and Total_Sum. Index represents time in seconds.
                
        Raises:
            ValueError: If no valid data is found in any files
        """
        all_dataframes = []
        
        for filepath in self.get_plot_files():
            if not os.path.exists(filepath):
                print(f"Warning: File not found: {filepath}", file=sys.stderr)
                continue
            
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                
                streams_dict = self._generate_bandwidth(data, self.unit)
                if not streams_dict:
                    continue
                
                # Get all unique time points
                all_indices = sorted(set().union(*[s.index for s in streams_dict.values()]))
                
                # Align all series
                filename = os.path.splitext(os.path.basename(filepath))[0]
                aligned = {
                    f"{filename}_{name}" if len(self.plot_files) > 1 else name: 
                    series.reindex(all_indices).ffill()
                    for name, series in streams_dict.items()
                }
                
                all_dataframes.append(pd.DataFrame(aligned))
                
            except (json.JSONDecodeError, Exception) as e:
                print(f"Error processing {filepath}: {e}", file=sys.stderr)
                continue
        
        if not all_dataframes:
            raise ValueError(f"No valid data found in folder: {self.foldername}")
        
        # Combine all DataFrames
        dataset = pd.concat(all_dataframes, axis=1) if len(all_dataframes) > 1 else all_dataframes[0]
        return dataset.ffill()


def main():
    """
    Main execution function for the iperf3 plotter
    
    Orchestrates the complete workflow: parsing command-line arguments,
    loading data, and generating plots according to the specified configuration.
    
    Exits with status code 1 if errors occur during execution.
    """
    parser = IperfDataParser()
    config = parser.parse_options(sys.argv[1:])
    
    try:
        dataset = config.get_dataset()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    
    plotter = IperfPlotter(config.bounds)
    
    # Generate title and description
    title = config.title or f"iperf3 Results ({config.unit})"
    desc = f"iperf3 throughput ({config.unit})"
    
    # Generate plots based on style
    if config.plot_style == 'separate':
        plotter.plot_streams_only(dataset, config.output, desc, title, config.unit)
        
        sum_file = config.sum_output or f"{os.path.splitext(config.output)[0]}_sum.png"
        plotter.plot_sum_only(dataset, sum_file, desc, title, config.unit)
        
        print(f"Streams plot: {config.output}")
        print(f"Sum plot: {sum_file}")
    
    elif config.plot_style == 'combined':
        plotter.plot_streams_with_sum(dataset, config.output, desc, title, config.unit)
        print(f"Combined plot: {config.output}")
    
    else:
        print(f"Unknown plot style: {config.plot_style}", file=sys.stderr)
        print("Available styles: separate, combined", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()