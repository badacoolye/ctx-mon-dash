import pandas as pd
import glob
import logging
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MonitorDataAnalyzer:
    def __init__(self, input_dir="output", output_dir="output"):
        self.apps_df = None
        self.servers_df = None
        
        # Determine the latest input subfolder if available.
        if os.path.isdir(input_dir):
            subdirs = [
                os.path.join(input_dir, d)
                for d in os.listdir(input_dir)
                if os.path.isdir(os.path.join(input_dir, d))
            ]
            if subdirs:
                # Assuming subfolder names are dates in YYYYMMDD format
                latest_subdir = max(subdirs)
                self.input_dir = latest_subdir
                logger.info(f"Using latest input directory: {self.input_dir}")
            else:
                self.input_dir = input_dir
                logger.info(f"No subdirectories found in {input_dir}, using {input_dir} as input.")
        else:
            self.input_dir = input_dir

        # Create a dated subfolder for output (YYYYMMDD format).
        current_date = datetime.now().strftime("%Y%m%d")
        self.output_dir = os.path.join(output_dir, current_date)
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Output directory ready: {self.output_dir}")

    def load_latest_data(self):
        """Load the most recent CSV files for both applications and servers from the input directory."""
        app_files = glob.glob(os.path.join(self.input_dir, 'applications_*.csv'))
        server_files = glob.glob(os.path.join(self.input_dir, 'servers_*.csv'))

        if app_files:
            latest_app = max(app_files)  # Assumes filenames include sortable timestamps.
            self.apps_df = pd.read_csv(latest_app)
            self.apps_df['SummaryDate'] = pd.to_datetime(self.apps_df['SummaryDate'])
            logger.info(f"Loaded applications data from {latest_app}")
        else:
            logger.warning("No application CSV files found in the input directory.")

        if server_files:
            latest_server = max(server_files)
            self.servers_df = pd.read_csv(latest_server)
            self.servers_df['SummaryDate'] = pd.to_datetime(self.servers_df['SummaryDate'])
            logger.info(f"Loaded servers data from {latest_server}")
        else:
            logger.warning("No server CSV files found in the input directory.")

    def _aggregate_data(self, df, group_col):
        """Helper method to aggregate data given a DataFrame and a grouping column."""
        if df is None or df.empty:
            return None
        try:
            # First, compute daily stats.
            daily_stats = df.groupby([group_col, pd.Grouper(key='SummaryDate', freq='D')]).agg({
                'TotalLaunchesCount': 'sum',
                'TotalUsageDuration': 'max',
                'PeakConcurrentInstanceCount': 'max'
            }).reset_index()

            # Then, aggregate across all days.
            final_stats = daily_stats.groupby(group_col).agg({
                'TotalLaunchesCount': 'sum',
                'TotalUsageDuration': 'sum',
                'PeakConcurrentInstanceCount': 'max'
            }).reset_index()

            final_stats['TotalUsageDuration_Hours'] = final_stats['TotalUsageDuration'] / 3600
            final_stats = final_stats.sort_values('PeakConcurrentInstanceCount', ascending=False)
            return final_stats

        except Exception as e:
            logger.error(f"Error aggregating data for {group_col}: {e}")
            return None

    def get_aggregated_values_per_application(self):
        """Aggregate values per application."""
        return self._aggregate_data(self.apps_df, 'Application_Name')

    def get_aggregated_values_per_desktop(self):
        """Aggregate values per desktop group."""
        return self._aggregate_data(self.servers_df, 'DesktopGroup_Name')

    def create_pie_charts(self, stats_df, chart_type, metrics=None):
        """
        Create pie charts for the analyzed data and return a list of file paths.
        The charts are saved as PNG files.
        """
        chart_files = []
        if stats_df is None or stats_df.empty:
            logger.warning(f"No data available to create pie charts for {chart_type}")
            return chart_files

        if metrics is None:
            metrics = ['TotalLaunchesCount', 'TotalUsageDuration', 'PeakConcurrentInstanceCount']

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_column = 'Application_Name' if chart_type.lower() == 'applications' else 'DesktopGroup_Name'
        colors = plt.cm.Set3(np.linspace(0, 1, 11))  # Color palette

        for metric in metrics:
            # Adjust figure size for bigger charts.
            plt.figure(figsize=(10, 10))
            data = stats_df[[name_column, metric]].copy()
            data = data.sort_values(metric, ascending=False)

            # Group smaller segments into "Others" if more than 10 items.
            if len(data) > 10:
                top_10 = data.iloc[:10]
                others_value = data.iloc[10:][metric].sum()
                others = pd.DataFrame([{name_column: 'Others', metric: others_value}])
                data = pd.concat([top_10, others], ignore_index=True)

            total = data[metric].sum()
            legend_labels = []
            for name, value in zip(data[name_column], data[metric]):
                percentage = (value / total) * 100 if total else 0
                if metric == 'TotalUsageDuration':
                    hours = value / 3600
                    legend_labels.append(f'{name} ({percentage:.1f}%) - {hours:.1f} hours')
                else:
                    legend_labels.append(f'{name} ({percentage:.1f}%) - {value:,.0f}')

            patches, _ = plt.pie(data[metric], colors=colors, labels=None)
            plt.legend(patches, legend_labels,
                       title=f"{metric.replace('Total', 'Total ').replace('Count', ' Count')} Distribution",
                       loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
            plt.title(f"{chart_type} by {metric.replace('Total', 'Total ').replace('Count', ' Count')}",
                      pad=20, fontsize=14)
            plt.tight_layout()

            output_file = os.path.join(self.output_dir, f"pie_chart_{chart_type.lower()}_{metric}_{timestamp}.png")
            plt.savefig(output_file, bbox_inches='tight', dpi=300)
            plt.close()
            logger.info(f"Saved pie chart to {output_file}")
            chart_files.append(output_file)
        return chart_files

    def generate_html_report(self, app_csv=None, server_csv=None, app_chart_files=None, server_chart_files=None):
        """
        Generate an HTML report that includes links to CSV files, renders the data as tables,
        and embeds the generated pie chart images side by side with improved styling.
        """
        report_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        html_file = os.path.join(self.output_dir, f"analysis_report_{report_timestamp}.html")
        html_content = f"""
        <html>
        <head>
            <title>Analysis Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .container {{
                    width: 90%;
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: #fff;
                    padding: 20px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2, h3 {{
                    color: #333;
                }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin-bottom: 20px;
                }}
                table, th, td {{
                    border: 1px solid #aaa;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                }}
                .chart-container {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                    margin-bottom: 20px;
                }}
                .chart-container > div {{
                    flex: 1 1 calc(33.333% - 20px);
                }}
                .chart-container img {{
                    width: 100%;
                    height: auto;
                    border: 1px solid #ccc;
                }}
                a {{
                    color: #0066cc;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Analysis Report</h1>
        """

        if app_csv:
            html_content += f"<h2>Applications Analysis</h2>\n"
            html_content += f'<p><a href="{os.path.basename(app_csv)}">Download Applications CSV</a></p>\n'
            try:
                df_app = pd.read_csv(app_csv)
                html_table = df_app.to_html(index=False)
                html_content += html_table
            except Exception as e:
                html_content += f"<p>Error reading applications CSV: {e}</p>"
            if app_chart_files:
                html_content += "<h3>Applications Pie Charts</h3>\n"
                html_content += '<div class="chart-container">'
                for chart in app_chart_files:
                    html_content += f'<div><img src="{os.path.basename(chart)}" alt="Pie Chart"></div>\n'
                html_content += '</div>'

        if server_csv:
            html_content += f"<h2>Desktop Groups Analysis</h2>\n"
            html_content += f'<p><a href="{os.path.basename(server_csv)}">Download Desktop Groups CSV</a></p>\n'
            try:
                df_server = pd.read_csv(server_csv)
                html_table = df_server.to_html(index=False)
                html_content += html_table
            except Exception as e:
                html_content += f"<p>Error reading desktop groups CSV: {e}</p>"
            if server_chart_files:
                html_content += "<h3>Desktop Groups Pie Charts</h3>\n"
                html_content += '<div class="chart-container">'
                for chart in server_chart_files:
                    html_content += f'<div><img src="{os.path.basename(chart)}" alt="Pie Chart"></div>\n'
                html_content += '</div>'

        html_content += """
            </div>
        </body>
        </html>
        """
        with open(html_file, "w") as f:
            f.write(html_content)
        logger.info(f"Saved HTML report to {html_file}")

    def save_analysis_results(self, output_prefix="analysis"):
        """
        Save analysis results to CSV files, create visualizations, and generate an HTML report.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        app_csv, server_csv = None, None
        app_chart_files, server_chart_files = [], []

        # Process application data.
        if self.apps_df is not None:
            app_stats = self.get_aggregated_values_per_application()
            if app_stats is not None:
                app_csv = os.path.join(self.output_dir, f"{output_prefix}_applications_{timestamp}.csv")
                app_stats.to_csv(app_csv, index=False)
                logger.info(f"Saved application analysis to {app_csv}")
                print("\nAll Applications (sorted by Peak Concurrent Instances):")
                pd.set_option('display.max_rows', None)
                pd.set_option('display.max_columns', None)
                pd.set_option('display.width', None)
                print(app_stats.to_string())
                app_chart_files = self.create_pie_charts(app_stats, "Applications")
        
        # Process server/desktop group data.
        if self.servers_df is not None:
            server_stats = self.get_aggregated_values_per_desktop()
            if server_stats is not None:
                server_csv = os.path.join(self.output_dir, f"{output_prefix}_servers_{timestamp}.csv")
                server_stats.to_csv(server_csv, index=False)
                logger.info(f"Saved server analysis to {server_csv}")
                print("\nAll Desktop Groups (sorted by Peak Concurrent Instances):")
                print(server_stats.to_string())
                server_chart_files = self.create_pie_charts(server_stats, "Desktop_Groups")
        
        # Generate HTML report combining CSV data and pie charts.
        self.generate_html_report(app_csv=app_csv, server_csv=server_csv,
                                  app_chart_files=app_chart_files, server_chart_files=server_chart_files)

def main():
    analyzer = MonitorDataAnalyzer()
    try:
        analyzer.load_latest_data()
        analyzer.save_analysis_results()
    except Exception as e:
        logger.error(f"An error occurred during analysis: {e}")
        raise

if __name__ == "__main__":
    main()