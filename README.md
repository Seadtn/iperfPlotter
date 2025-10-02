# iperf3 Plotter

A Python tool for visualizing iperf3 network performance test results with publication-quality plots.

## Overview

This tool parses iperf3 JSON output files and generates customizable bandwidth plots showing individual stream performance, aggregate throughput, or both. It's designed for network performance analysis, benchmarking, and reporting.

## Features

- **Multiple Plot Styles**: Generate separate plots for streams and totals, or combine them
- **Flexible Data Sources**: Process single or multiple iperf3 JSON files
- **Customizable Boundaries**: Add reference lines for expected performance thresholds
- **Unit Support**: Display bandwidth in Mbps or MBps
- **Professional Styling**: Clean, publication-ready plots with Tableau color palette
- **Batch Processing**: Automatically discover and plot all JSON files in a directory

## Requirements

```bash
python >= 3.6
numpy
pandas
matplotlib
seaborn
```

## Installation

1. Clone or download the script:

```bash
git clone <repository-url>
cd iperfPlotter
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Plot all JSON files in a folder:

```bash
./iperf3_plot.py -f /path/to/iperf/results
```

### Common Options

```bash
# Specify output filename
./iperf3_plot.py -f ./results -o bandwidth_plot.png

# Use MBps instead of Mbps
./iperf3_plot.py -f ./results --unit MBps

# Combined plot (streams + total on same graph)
./iperf3_plot.py -f ./results --plot-style combined

# Add custom title
./iperf3_plot.py -f ./results -t "My Network Performance Test"
```

### Advanced Options

#### Adding Performance Boundaries

Add upper and lower threshold lines:

```bash
# Simple threshold
./iperf3_plot.py -f ./results -u 1000 -l 800

# Multiple labeled boundaries
./iperf3_plot.py -f ./results -b "[1000,900,Expected][800,700,Minimum]"
```

#### Filtering Files

```bash
# Plot specific files only
./iperf3_plot.py -f ./results -p "test1.json,test2.json"

# Exclude specific files
./iperf3_plot.py -f ./results -n "failed_test.json,warmup.json"
```

#### Custom Output Names

```bash
# Specify both stream and sum plot filenames
./iperf3_plot.py -f ./results -o streams.png -s totals.png
```

## Command-Line Options

| Option                | Description                              | Default        |
| --------------------- | ---------------------------------------- | -------------- |
| `-f`, `--folder`      | Input folder path (required)             | -              |
| `-o`, `--output`      | Output plot filename                     | `iperf.png`    |
| `-s`, `--sum-output`  | Sum plot filename                        | Auto-generated |
| `-p`, `--plotfiles`   | Comma-separated list of files to plot    | All JSON files |
| `-n`, `--noPlotFiles` | Comma-separated list of files to exclude | None           |
| `-u`, `--upperLimit`  | Expected upper boundary                  | 0              |
| `-l`, `--lowerLimit`  | Expected lower boundary                  | 0              |
| `-b`, `--bound`       | Multiple bounds: `[upper,lower,tag]...`  | None           |
| `--unit`              | Bandwidth unit: `Mbps` or `MBps`         | `Mbps`         |
| `--plot-style`        | Plot style: `separate` or `combined`     | `separate`     |
| `-t`, `--title`       | Custom plot title                        | Auto-generated |
| `-v`, `--verbose`     | Verbose output                           | Off            |

## Examples

### Example 1: Basic Network Test

```bash
# Run iperf3 test
iperf3 -c server.example.com -t 60 -P 4 -J > results.json

# Create plots
./iperf3_plot.py -f . -p results.json -t "4-Stream Test to Server"
```

Output:

- `iperf.png` - Individual stream bandwidths
- `iperf_sum.png` - Total aggregate throughput

### Example 2: Comparing Multiple Tests

```bash
# Directory structure:
# /tests/
#   ├── test_10gbps.json
#   ├── test_1gbps.json
#   └── test_100mbps.json

./iperf3_plot.py -f /tests --plot-style combined -t "Interface Comparison"
```

### Example 3: Performance Validation

```bash
# Test with expected 10 Gbps link (9.5-10 Gbps acceptable)
./iperf3_plot.py -f ./results \
  -u 10000 -l 9500 \
  -t "10G Link Validation" \
  --unit Mbps
```

### Example 4: Long-term Monitoring

```bash
# Plot hourly tests, exclude failures
./iperf3_plot.py -f /var/log/network_tests \
  -n "failed.json,incomplete.json" \
  -b "[1000,950,Target][800,750,Acceptable]" \
  -o daily_performance.png
```

## Output Files

### Separate Mode (Default)

- **Streams Plot**: Shows individual stream performance over time

  - One line per iperf3 stream
  - Colored using Tableau20 palette
  - Legend positioned outside plot area

- **Sum Plot**: Shows aggregate throughput
  - Single bold line representing total bandwidth
  - Cleaner view for overall performance

### Combined Mode

- Single plot with all streams (lighter, transparent lines) and total (bold black line)
- Useful for understanding stream distribution and total performance together

## iperf3 JSON Format

This tool expects standard iperf3 JSON output. Generate compatible files:

```bash
# Single stream
iperf3 -c <server> -t <duration> -J > output.json

# Multiple streams (parallel)
iperf3 -c <server> -t <duration> -P <num_streams> -J > output.json

# UDP test
iperf3 -c <server> -u -b <bandwidth> -J > output.json
```

## Plot Customization

The tool uses professional styling by default:

- **Color Palette**: Tableau20 for distinct, colorblind-friendly colors
- **Grid**: Subtle horizontal gridlines for readability
- **Fonts**: 12pt labels, 16pt title
- **Resolution**: 300 DPI for publication quality
- **Format**: PNG with tight bounding box

## Troubleshooting

### No Data Found Error

```
Error: No valid data found in folder: /path/to/folder
```

**Solution**: Ensure the folder contains valid iperf3 JSON files (with `-J` flag)

### Missing Total_Sum Column

```
ValueError: No Total_Sum column found in dataset
```

**Solution**: This occurs with single-file plots in sum-only mode. Ensure JSON contains interval data.

### File Not Found Warning

```
Warning: File not found: /path/to/file.json
```

**Solution**: Verify file paths are correct. Use absolute paths or ensure files exist in specified folder.

## Architecture

### Classes

- **`IperfPlotter`**: Handles all plot generation

  - `plot_streams_only()`: Individual stream visualization
  - `plot_sum_only()`: Aggregate throughput visualization
  - `plot_streams_with_sum()`: Combined visualization

- **`IperfDataParser`**: Manages data parsing and configuration
  - `parse_options()`: Command-line argument processing
  - `get_dataset()`: JSON file loading and DataFrame creation
  - `_generate_bandwidth()`: Bandwidth extraction from JSON

## Contributing

Contributions are welcome! Areas for improvement:

- Additional plot types (box plots, histograms)
- Support for iperf2 format
- Interactive plots with Plotly
- CSV export functionality
- Statistics summary generation

## License

MIT License - Feel free to use and modify as needed.

## Author

Mohamed Amine Nasr

## Changelog

### Version 1.0

- Initial release
- Support for Mbps and MBps units
- Separate and combined plot modes
- Configurable boundaries
- Multi-file processing

## Support

For issues, questions, or feature requests, please open an issue on the project repository.

## See Also

- [iperf3 Main Code](https://github.com/hchiuzhuo/iperfPlotter/blob/master/iperf3_plot.py)
