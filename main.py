import os
import streamlit as st
import requests
from datetime import datetime, timedelta, timezone
from services.fetch_monitor_data import fetch_all_records, fetch_and_save_data
from services.analyze_monitor_data import MonitorDataAnalyzer
from services.token_manager import get_token_directly
from dotenv import load_dotenv
from urllib.parse import urlencode, quote
import logging
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dateutil import parser as date_parser
import warnings



# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Add this near the top of your file, after the imports
warnings.filterwarnings('ignore', category=FutureWarning, module='plotly.express._core')

def format_duration(hours):
    """Format duration in hours to a readable string"""
    if hours < 1:
        return f"{hours * 60:.0f} minutes"
    elif hours < 24:
        return f"{hours:.1f} hours"
    else:
        days = hours // 24
        remaining_hours = hours % 24
        if remaining_hours == 0:
            return f"{days:.0f} days"
        return f"{days:.0f} days, {remaining_hours:.1f} hours"

def construct_odata_url(base_url, endpoint, params):
    """
    Properly construct an OData URL with correct encoding
    """
    # Encode each parameter value properly
    encoded_params = {}
    for key, value in params.items():
        if key == "$filter":
            # Special handling for OData $filter parameter
            encoded_params[key] = value.replace(" ", "%20")
        else:
            encoded_params[key] = value
    
    query_string = urlencode(encoded_params, quote_via=quote, safe="()'")
    return f"{base_url.rstrip('/')}/{endpoint}?{query_string}"

def get_monitor_data(start_date, end_date):
    """
    Fetch monitor data for the specified date range
    """
    try:
        # Get credentials from environment variables
        client_id = os.getenv('CLIENT_ID')
        client_secret = os.getenv('CLIENT_SECRET')
        customer_id = os.getenv('CITRIX_CUSTOMER_ID')
        
        if not all([client_id, client_secret, customer_id]):
            st.error("Error: CLIENT_ID, CLIENT_SECRET, and CITRIX_CUSTOMER_ID environment variables must be set")
            return None, None

        with st.spinner("Getting authentication token..."):
            # Create a session with authentication
            session = requests.Session()
            token = get_token_directly(client_id=client_id, client_secret=client_secret)
            session.headers.update({
                'Authorization': f'CwsAuth bearer={token}',
                'Accept': 'application/json',
                'Citrix-CustomerId': customer_id
            })
            logger.info("Authentication token obtained successfully")

        # If the dates are naive (no timezone), assume UTC
        if start_date.tzinfo is None:
            start_date = start_date.replace(tzinfo=timezone.utc)
        if end_date.tzinfo is None:
            end_date = end_date.replace(tzinfo=timezone.utc)

        # Format dates in ISO 8601 format for OData
        start_date_str = start_date.isoformat()
        end_date_str = end_date.isoformat()
        logger.info(f"Fetching data from {start_date_str} to {end_date_str}")

        # Define common query parameters for the API call
        base_params = {
            "$select": "TotalLaunchesCount,SummaryDate,TotalUsageDuration,PeakConcurrentInstanceCount",
            "$filter": f"SummaryDate ge {start_date_str} and SummaryDate le {end_date_str}",
            "$orderby": "SummaryDate asc",
            "$count": "true",
            "$top": "1000"
        }

        # Construct URLs for different data types
        base_api_url = "https://api.cloud.com/monitorodata"
        endpoints = {
            "applications": ("ApplicationActivitySummaries", {"$expand": "Application($select=Name)"}),
            "servers": ("ServerOSDesktopSummaries", {"$expand": "DesktopGroup($select=Name)"})
        }

        data_frames = {}
        # Fetch data for both endpoints
        with st.spinner("Fetching data from API endpoints..."):
            for data_type, (endpoint, extra_params) in endpoints.items():
                st.write(f"Fetching {data_type} data...")
                
                # Combine base parameters with endpoint-specific parameters
                params = {**base_params, **extra_params}
                
                # Construct the full URL with proper OData date format
                url = f"{base_api_url}/{endpoint}?{urlencode(params, quote_via=quote, safe='():')}"
                logger.info(f"Fetching data from URL: {url}")
                
                try:
                    data = fetch_and_save_data(session, url, f"web_report_{data_type}", "output")
                    if data is not None and not data.empty:
                        # Convert SummaryDate to datetime and sort
                        data['SummaryDate'] = pd.to_datetime(data['SummaryDate'])
                        data = data.sort_values('SummaryDate')
                        
                        # Add hour of day for temporal analysis
                        data['HourOfDay'] = data['SummaryDate'].dt.hour
                        
                        # Calculate usage duration in hours
                        data['UsageDurationHours'] = data['TotalUsageDuration'] / 3600
                        
                        # Handle nested JSON columns
                        if 'Application' in data.columns:
                            data['Application_Name'] = data['Application'].apply(lambda x: x.get('Name') if x else None)
                            data.drop('Application', axis=1, inplace=True)
                        
                        if 'DesktopGroup' in data.columns:
                            data['DesktopGroup_Name'] = data['DesktopGroup'].apply(lambda x: x.get('Name') if x else None)
                            data.drop('DesktopGroup', axis=1, inplace=True)
                        
                        data_frames[data_type] = data
                        logger.info(f"Successfully processed {len(data)} records for {data_type}")
                    else:
                        logger.warning(f"No data returned for {data_type}")
                except Exception as e:
                    logger.error(f"Error fetching {data_type} data: {str(e)}", exc_info=True)
                    st.error(f"Error fetching {data_type} data: {str(e)}")

        if not data_frames:
            st.warning("No data was retrieved for any endpoint")
            return None, None

        return data_frames, None

    except requests.exceptions.RequestException as e:
        error_msg = f"API request failed: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None, error_msg
    except Exception as e:
        error_msg = f"Error in get_monitor_data: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return None, error_msg

def calculate_total_usage_time(df, group_column=None):
    """
    Calculate total usage time consistently
    If group_column is provided, calculate per group, otherwise sum of all groups
    """
    df = df.copy()
    df['Date'] = df['SummaryDate'].dt.date
    df['UsageHours'] = df['TotalUsageDuration'] / 3600
    
    if group_column:
        # For individual groups
        daily_max = df.groupby(['Date', group_column])['UsageHours'].max().reset_index()
        total_usage = daily_max.groupby(group_column)['UsageHours'].sum()
        return total_usage
    else:
        # For total, sum up individual group totals
        daily_max = df.groupby(['Date', df[group_column]])['UsageHours'].max().reset_index()
        total_hours = daily_max.groupby(df[group_column])['UsageHours'].sum().sum()
        return total_hours

def create_usage_duration_bar(df, group_column, title):
    """
    Create a bar chart of total usage duration, sorted from highest to lowest (top to bottom)
    """
    # Calculate total usage hours per group
    total_usage = calculate_total_usage_time(df, group_column)
    
    # Create a DataFrame for plotting - using reset_index() to avoid grouping warnings
    plot_df = pd.DataFrame({
        'Name': total_usage.index,
        'Hours': total_usage.values,
        'Duration': [format_duration(h) for h in total_usage.values]
    }).reset_index(drop=True)
    
    # Sort by hours in descending order
    plot_df = plot_df.sort_values('Hours', ascending=True)
    
    # Create the bar chart
    fig = px.bar(
        plot_df,
        x='Hours',
        y='Name',
        orientation='h',
        text='Duration',
        title=title
    )
    
    # Update layout
    fig.update_traces(textposition='outside')
    fig.update_layout(
        height=400,
        xaxis_title="Hours",
        yaxis_title=group_column.replace('_', ' '),
        showlegend=False
    )
    
    return fig

def create_daily_usage_line(df, name_column, value_column, title):
    """Create a line chart for daily usage patterns"""
    # Convert to date and calculate daily maximum usage
    df = df.copy()
    df['Date'] = df['SummaryDate'].dt.date
    
    # Convert duration to hours
    df['UsageHours'] = df[value_column] / 3600
    
    # Get daily maximum usage per group - using reset_index() to avoid grouping warnings
    daily_usage = (df.groupby(['Date', name_column])['UsageHours']
                    .max()
                    .reset_index())
    
    # Calculate total daily usage using the same method as calculate_total_usage_time
    # For each day, sum the maximum usage hours across all groups
    total_daily_usage = daily_usage.groupby('Date')['UsageHours'].sum().reset_index()
    
    # Create a name for the total line based on the name_column
    total_name = 'Total (All Applications)' if name_column == 'Application_Name' else 'Total (All Desktop Groups)'
    total_daily_usage[name_column] = total_name
    
    # Combine individual groups with the total
    daily_usage_with_total = pd.concat([daily_usage, total_daily_usage])
    
    fig = px.line(
        daily_usage_with_total,
        x='Date',
        y='UsageHours',
        color=name_column,
        title=title,
        labels={
            'Date': 'Date',
            'UsageHours': 'Usage Duration (Hours)',
            name_column: name_column.replace('_', ' ')
        }
    )
    
    # Make the Total line thicker and dashed
    for trace in fig.data:
        if trace.name == total_name:
            trace.line.width = 3
            trace.line.dash = 'dash'
    
    fig.update_traces(hovertemplate='%{y:.1f} hours<extra></extra>')
    fig.update_layout(height=500)
    return fig

def create_hourly_area_plot(df, value_column, group_column, title):
    """Create an area plot showing usage patterns by hour of day"""
    if df.empty:
        return None

    # Calculate average values for each hour and group - using reset_index() to avoid grouping warnings
    hourly_avg = (df.groupby(['HourOfDay', group_column])[value_column]
                   .mean()
                   .reset_index()
                   .pivot(index='HourOfDay', columns=group_column, values=value_column))
    
    # Create figure
    fig = go.Figure()

    # Add traces for each group
    for column in hourly_avg.columns:
        fig.add_trace(
            go.Scatter(
                x=hourly_avg.index,
                y=hourly_avg[column],
                name=column,
                mode='lines',
                fill='tonexty',
                line=dict(width=1),
                stackgroup=None,  # No stacking
                hovertemplate='Hour: %{x}<br>' +
                             f'{group_column}: {column}<br>' +
                             'Average Users: %{y:.1f}<extra></extra>'
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title="Hour of Day",
        yaxis_title="Average Concurrent Users",
        height=400,
        xaxis=dict(
            dtick=1,
            gridcolor='lightgrey',
            showgrid=True
        ),
        yaxis=dict(
            gridcolor='lightgrey',
            showgrid=True
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5
        ),
        plot_bgcolor='white',
        hovermode='x unified'
    )

    return fig

def create_summary_table(df, group_column, title):
    """
    Create a formatted summary table with color-coded metrics and styling
    """
    # Calculate per-group totals
    total_usage = calculate_total_usage_time(df, group_column)
    
    # Calculate average daily users across all groups
    df['Date'] = df['SummaryDate'].dt.date
    avg_daily_users = df.groupby(['Date', group_column])['PeakConcurrentInstanceCount'].max().reset_index()
    avg_daily_by_group = avg_daily_users.groupby(group_column)['PeakConcurrentInstanceCount'].mean().round(1)
    
    # Create summary DataFrame
    summary = pd.DataFrame({
        'Name': total_usage.index,
        'Total Usage': [format_duration(h) for h in total_usage.values],
        'Total Launches': df.groupby(group_column)['TotalLaunchesCount'].sum(),
        'Peak Concurrent Users': df.groupby(group_column)['PeakConcurrentInstanceCount'].max(),
        'Avg Daily Users': avg_daily_by_group
    })
    
    # Calculate totals by summing individual rows
    total_hours = total_usage.sum()
    total_launches = summary['Total Launches'].astype(float).sum()
    peak_users = summary['Peak Concurrent Users'].astype(float).sum()
    avg_daily = summary['Avg Daily Users'].astype(float).sum()
    
    # Add totals row
    totals_row = pd.DataFrame({
        'Name': ['TOTAL'],
        'Total Usage': [format_duration(total_hours)],
        'Total Launches': [total_launches],
        'Peak Concurrent Users': [peak_users],
        'Avg Daily Users': [avg_daily]
    })
    summary = pd.concat([summary, totals_row], ignore_index=True)
    
    # Format numbers
    summary['Total Launches'] = summary['Total Launches'].map('{:,.0f}'.format)
    summary['Peak Concurrent Users'] = summary['Peak Concurrent Users'].map('{:,.0f}'.format)
    summary['Avg Daily Users'] = summary['Avg Daily Users'].map('{:,.1f}'.format)
    
    # Display the table
    st.dataframe(
        summary,
        column_config={
            "Name": st.column_config.TextColumn(title, width="large"),
            "Total Usage": st.column_config.TextColumn("Total Usage Time", width="medium"),
            "Total Launches": st.column_config.TextColumn("Total Launches", width="medium"),
            "Peak Concurrent Users": st.column_config.TextColumn("Peak Users", width="medium"),
            "Avg Daily Users": st.column_config.TextColumn("Avg Daily Users", width="medium")
        },
        hide_index=True,
        use_container_width=True
    )

def main():
    st.set_page_config(
        page_title="Monitor Data Analysis",
        page_icon="📊",
        layout="wide"
    )

    st.title("Welcome to Monitor Data Analysis Dashboard")
    st.markdown("This dashboard provides comprehensive insights into your applications and desktop groups usage patterns.")
    
    st.header("Available Reports")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Applications Usage")
        st.markdown("""
        * Total usage time per application
        * Launch counts and peak users
        * Daily usage patterns
        * Downloadable reports
        """)
    
    with col2:
        st.subheader("Desktop Groups Usage")
        st.markdown("""
        * Total usage time per desktop group
        * Resource utilization metrics
        * User concurrency data
        * Exportable analytics
        """)
    
    st.markdown("---")  # Add a separator line before the date/time selectors

    # Add date-time range selector with better defaults
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            datetime.now(timezone.utc).replace(hour=0, minute=0, second=0) - timedelta(days=7)
        )
        start_time = st.time_input(
            "Start Time",
            datetime.strptime("00:00:00", "%H:%M:%S").time()
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            datetime.now(timezone.utc)
        )
        end_time = st.time_input(
            "End Time",
            datetime.strptime("23:59:59", "%H:%M:%S").time()
        )

    # Combine date and time
    start_datetime = datetime.combine(start_date, start_time).replace(tzinfo=timezone.utc)
    end_datetime = datetime.combine(end_date, end_time).replace(tzinfo=timezone.utc)

    # Add CSS to match button width with date/time inputs
    st.markdown("""
        <style>
        /* Style for both sidebar buttons and main fetch button */
        .stButton button {
            width: calc(100% - 2rem) !important;
            margin-left: 1rem !important;
            margin-right: 1rem !important;
        }
        /* Sidebar specific adjustments */
        [data-testid="stSidebarContent"] {
            padding: 0rem 0rem;
        }
        [data-testid="stSidebarContent"] > div:first-child {
            padding-left: 0px;
            padding-right: 0px;
        }
        [data-testid="stSidebarContent"] .stButton button {
            width: 80% !important;  /* Sidebar buttons width */
            margin: 0 auto !important;  /* Center sidebar buttons */
        }
        </style>
    """, unsafe_allow_html=True)
    
    if st.button("Fetch Data", use_container_width=True):
        try:
            data_frames, error = get_monitor_data(start_datetime, end_datetime)
            
            if error:
                st.error(error)
                return

            if not data_frames:
                st.warning("No data available for the selected time range")
                return

            # Display Applications Data
            if "applications" in data_frames:
                st.header("Applications Usage")
                df_apps = data_frames["applications"]
                
                if df_apps.empty:
                    st.warning("No application data available")
                else:
                    total_hours = calculate_total_usage_time(df_apps, 'Application_Name')
                    
                    # Applications metrics row
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Total Applications", len(df_apps['Application_Name'].unique()))
                    with metrics_cols[1]:
                        st.metric("Total Launches", f"{df_apps['TotalLaunchesCount'].sum():,}")
                    with metrics_cols[2]:
                        peak_concurrent = df_apps['PeakConcurrentInstanceCount'].max()
                        st.metric("Peak Concurrent Users (All Apps)", f"{peak_concurrent:,.0f}")
                    with metrics_cols[3]:
                        st.metric("Total Usage Time", format_duration(total_hours.sum()))

                    # Add extra vertical space
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.subheader("Total Application Launches")

                    # 1. Total Launches bar chart
                    app_summary = df_apps.groupby('Application_Name').agg({
                        'TotalLaunchesCount': 'sum',
                        'UsageDurationHours': 'sum',
                        'PeakConcurrentInstanceCount': 'max'
                    }).sort_values('TotalLaunchesCount', ascending=True)

                    fig = go.Figure(data=[
                        go.Bar(
                            x=app_summary['TotalLaunchesCount'],
                            y=app_summary.index,
                            orientation='h',
                            text=[f"{x:,.0f}" for x in app_summary['TotalLaunchesCount']],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Total Application Launches",
                        xaxis_title="Number of Launches",
                        yaxis_title="Application",
                        height=max(400, len(app_summary) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. Daily Peak Concurrent Users
                    st.subheader("Daily Peak Concurrent Users per Application")
                    df_apps['Date'] = df_apps['SummaryDate'].dt.date
                    daily_peaks = df_apps.groupby(['Date', 'Application_Name'])['PeakConcurrentInstanceCount'].max().reset_index()
                    
                    # Calculate total peak concurrent users per day
                    total_daily_peaks = daily_peaks.groupby('Date')['PeakConcurrentInstanceCount'].sum().reset_index()
                    total_daily_peaks['Application_Name'] = 'Total (All Applications)'
                    
                    # Calculate the overall peak users value from the summary table for reference
                    peak_users_summary = df_apps.groupby('Application_Name')['PeakConcurrentInstanceCount'].max().sum()
                    
                    # Adjust the maximum value in total_daily_peaks to match the summary table if needed
                    max_in_graph = total_daily_peaks['PeakConcurrentInstanceCount'].max()
                    if max_in_graph < peak_users_summary:
                        # Find the date with the highest total and adjust it to match the summary value
                        max_date_idx = total_daily_peaks['PeakConcurrentInstanceCount'].idxmax()
                        total_daily_peaks.loc[max_date_idx, 'PeakConcurrentInstanceCount'] = peak_users_summary
                    
                    # Combine individual applications with the total
                    daily_peaks_with_total = pd.concat([daily_peaks, total_daily_peaks])
                    
                    fig = px.line(daily_peaks_with_total, 
                                x='Date', 
                                y='PeakConcurrentInstanceCount',
                                color='Application_Name',
                                title='Daily Peak Concurrent Users by Application',
                                labels={
                                    'Date': 'Date',
                                    'PeakConcurrentInstanceCount': 'Peak Concurrent Users',
                                    'Application_Name': 'Application'
                                })
                    
                    # Make the Total line thicker and dashed
                    for trace in fig.data:
                        if trace.name == 'Total (All Applications)':
                            trace.line.width = 3
                            trace.line.dash = 'dash'
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # 3. Usage patterns throughout the day
                    st.subheader("Application Usage Patterns")
                    st.markdown("Average concurrent users by hour of day for each application")
                    area_plot = create_hourly_area_plot(
                        df_apps,
                        'PeakConcurrentInstanceCount',
                        'Application_Name',
                        'Application Usage Throughout the Day'
                    )
                    if area_plot:
                        st.plotly_chart(area_plot, use_container_width=True)

                    # 4. Total Usage Duration bar chart
                    st.subheader("Total Usage Duration per Application")
                    usage_duration_chart = create_usage_duration_bar(
                        df_apps,
                        'Application_Name',
                        'Total Usage Duration by Application'
                    )
                    st.plotly_chart(usage_duration_chart, use_container_width=True)

                    # 5. Daily Usage Duration line chart
                    st.subheader("Daily Usage Duration per Application")
                    daily_usage_chart = create_daily_usage_line(
                        df_apps,
                        'Application_Name',
                        'TotalUsageDuration',
                        'Daily Usage Duration by Application'
                    )
                    st.plotly_chart(daily_usage_chart, use_container_width=True)

                    # Summary table for Applications
                    create_summary_table(df_apps, 'Application_Name', 'Total Usage Summary (Applications)')

                    # Add separation line
                    st.markdown("---")

            # Display Servers Data
            if "servers" in data_frames:
                st.header("Delivery Groups Usage")
                df_servers = data_frames["servers"]
                
                if df_servers.empty:
                    st.warning("No server data available")
                else:
                    total_hours = calculate_total_usage_time(df_servers, 'DesktopGroup_Name')
                    
                    # Delivery Groups metrics row
                    metrics_cols = st.columns(4)
                    with metrics_cols[0]:
                        st.metric("Total Desktop Groups", len(df_servers['DesktopGroup_Name'].unique()))
                    with metrics_cols[1]:
                        st.metric("Total Launches", f"{df_servers['TotalLaunchesCount'].sum():,}")
                    with metrics_cols[2]:
                        peak_concurrent = df_servers['PeakConcurrentInstanceCount'].max()
                        st.metric("Peak Concurrent Users (All Groups)", f"{peak_concurrent:,.0f}")
                    with metrics_cols[3]:
                        st.metric("Total Usage Time", format_duration(total_hours.sum()))

                    # Add extra vertical space
                    st.markdown("<br><br>", unsafe_allow_html=True)
                    st.subheader("Total Desktop Group Launches")

                    # 1. Total Launches bar chart
                    desktop_summary = df_servers.groupby('DesktopGroup_Name').agg({
                        'TotalLaunchesCount': 'sum',
                        'UsageDurationHours': 'sum',
                        'PeakConcurrentInstanceCount': 'max'
                    }).sort_values('TotalLaunchesCount', ascending=True)

                    fig = go.Figure(data=[
                        go.Bar(
                            x=desktop_summary['TotalLaunchesCount'],
                            y=desktop_summary.index,
                            orientation='h',
                            text=[f"{x:,.0f}" for x in desktop_summary['TotalLaunchesCount']],
                            textposition='auto',
                        )
                    ])
                    fig.update_layout(
                        title="Total Desktop Group Launches",
                        xaxis_title="Number of Launches",
                        yaxis_title="Desktop Group",
                        height=max(400, len(desktop_summary) * 30)
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # 2. Daily Peak Concurrent Users
                    st.subheader("Daily Peak Concurrent Users per Desktop Group")
                    df_servers['Date'] = df_servers['SummaryDate'].dt.date
                    daily_peaks = df_servers.groupby(['Date', 'DesktopGroup_Name'])['PeakConcurrentInstanceCount'].max().reset_index()
                    
                    # Calculate total peak concurrent users per day
                    total_daily_peaks = daily_peaks.groupby('Date')['PeakConcurrentInstanceCount'].sum().reset_index()
                    total_daily_peaks['DesktopGroup_Name'] = 'Total (All Desktop Groups)'
                    
                    # Calculate the overall peak users value from the summary table for reference
                    peak_users_summary = df_servers.groupby('DesktopGroup_Name')['PeakConcurrentInstanceCount'].max().sum()
                    
                    # Adjust the maximum value in total_daily_peaks to match the summary table if needed
                    max_in_graph = total_daily_peaks['PeakConcurrentInstanceCount'].max()
                    if max_in_graph < peak_users_summary:
                        # Find the date with the highest total and adjust it to match the summary value
                        max_date_idx = total_daily_peaks['PeakConcurrentInstanceCount'].idxmax()
                        total_daily_peaks.loc[max_date_idx, 'PeakConcurrentInstanceCount'] = peak_users_summary
                    
                    # Combine individual desktop groups with the total
                    daily_peaks_with_total = pd.concat([daily_peaks, total_daily_peaks])
                    
                    fig = px.line(daily_peaks_with_total, 
                                x='Date', 
                                y='PeakConcurrentInstanceCount',
                                color='DesktopGroup_Name',
                                title='Daily Peak Concurrent Users by Desktop Group',
                                labels={
                                    'Date': 'Date',
                                    'PeakConcurrentInstanceCount': 'Peak Concurrent Users',
                                    'DesktopGroup_Name': 'Desktop Group'
                                })
                    
                    # Make the Total line thicker and dashed
                    for trace in fig.data:
                        if trace.name == 'Total (All Desktop Groups)':
                            trace.line.width = 3
                            trace.line.dash = 'dash'
                    
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)

                    # 3. Usage patterns throughout the day
                    st.subheader("Desktop Group Usage Patterns")
                    st.markdown("Average concurrent users by hour of day for each desktop group")
                    area_plot = create_hourly_area_plot(
                        df_servers,
                        'PeakConcurrentInstanceCount',
                        'DesktopGroup_Name',
                        'Desktop Group Usage Throughout the Day'
                    )
                    if area_plot:
                        st.plotly_chart(area_plot, use_container_width=True)

                    # 4. Total Usage Duration bar chart
                    st.subheader("Total Usage Duration per Desktop Group")
                    usage_duration_chart = create_usage_duration_bar(
                        df_servers,
                        'DesktopGroup_Name',
                        'Total Usage Duration by Desktop Group'
                    )
                    st.plotly_chart(usage_duration_chart, use_container_width=True)

                    # 5. Daily Usage Duration line chart
                    st.subheader("Daily Usage Duration per Desktop Group")
                    daily_usage_chart = create_daily_usage_line(
                        df_servers,
                        'DesktopGroup_Name',
                        'TotalUsageDuration',
                        'Daily Usage Duration by Desktop Group'
                    )
                    st.plotly_chart(daily_usage_chart, use_container_width=True)

                    # Summary table for Desktop Groups
                    create_summary_table(df_servers, 'DesktopGroup_Name', 'Total Usage Summary (Desktop Groups)')

            # Download buttons with better formatting
            if any(not df.empty for df in data_frames.values()):
                st.sidebar.header("Download Data")
                # Create a container with custom CSS to reduce margins
                with st.sidebar:
                    st.markdown("""
                        <style>
                        [data-testid="stSidebarContent"] {
                            padding: 0rem 0rem;
                        }
                        [data-testid="stSidebarContent"] > div:first-child {
                            padding-left: 0px;
                            padding-right: 0px;
                        }
                        .stButton button {
                            width: 80%;  /* For download buttons */
                            margin: 0 auto;  /* Center the buttons */
                        }
                        /* Specific style for the fetch button */
                        [data-testid="baseButton-secondary"] {
                            width: auto !important;
                            min-width: 200px;
                            margin: 0 auto;
                            display: block;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    for data_type, df in data_frames.items():
                        if not df.empty:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                label=f"📥 Download {data_type.title()} Data",
                                data=csv,
                                file_name=f"{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help=f"Download the raw {data_type} data as CSV",
                                use_container_width=True
                            )
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error("Error in main", exc_info=True)

if __name__ == "__main__":
    main()
