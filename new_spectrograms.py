## Author: Prof. Meredith Franklin
## Contributor: Yang Xiang
## Date: 2025-05-07-13-40
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz

# Assuming the file is in the current working directory.
# If not, provide the full path to the file.
try:
    df = pd.read_csv(r"C:\Users\franc\PycharmProjects\Summer_research\Codes\sound_pollutant_met_data.csv")
    print(df.head())
except FileNotFoundError:
    print("Error: 'sound_pollutant_met_data.csv' not found. Please make sure the file exists in the current directory or provide the correct path.")

# Renaming columns by replacing spaces with underscores
df.rename(columns=lambda col: col.replace(" ", "_"), inplace=True)
df['time_utc'] = pd.to_datetime(df['time_utc'], errors='coerce', utc = True)

# convert time zone from utc to mountain
mountain_tz = pytz.timezone('America/Denver')
df['time_mountain'] = df['time_utc'].dt.tz_convert(mountain_tz)
print(f"The time zone of the new dataset is at {df["time_mountain"].dt.tz}")


# Define the list of sound frequencies we want to plot (32 in total)
frequency_reduced = [
    "LZeq_12.5Hz", "LZeq_16Hz",
    "LZeq_20Hz", "LZeq_25Hz", "LZeq_31.5Hz", "LZeq_40Hz", "LZeq_50Hz",
    "LZeq_63Hz", "LZeq_80Hz", "LZeq_100Hz", "LZeq_125Hz", "LZeq_160Hz",
    "LZeq_200Hz", "LZeq_250Hz", "LZeq_315Hz", "LZeq_400Hz", "LZeq_500Hz",
    "LZeq_630Hz", "LZeq_800Hz", "LZeq_1kHz", "LZeq_1.25kHz", "LZeq_1.6kHz",
    "LZeq_2kHz", "LZeq_2.5kHz", "LZeq_3.15kHz", "LZeq_4kHz", "LZeq_5kHz",
    "LZeq_6.3kHz", "LZeq_8kHz", "LZeq_10kHz", "LZeq_12.5kHz", "LZeq_16kHz"
]   ## Yang: the frequency list is verified to be containing all frequencies we need, as we did in the summary bar plot.


# Define Function to plot spectrogram

def plot_spectrogram_with_dates(df, time_col, freq_cols, plot_case=""):
    """
    Plots a spectrogram with y-axis as sound frequencies and x-axis as time (formatted as dates).
    """
    # Ensure time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])

    # Convert the time column to seconds since the start for plotting purposes
    time_deltas = (df[time_col] - df[time_col].min()).dt.total_seconds()
    time_array = df[time_col].to_numpy()  # Use datetime directly for x-axis labels

    # Extract frequency data and transpose it for plotting
    sound_matrix = df[freq_cols].to_numpy().T  # Transpose for correct orientation
     # Check for non-finite values in the sound_matrix and replace them
    sound_matrix = np.nan_to_num(sound_matrix)
    # Create the plot
    plt.figure(figsize=(12, 8))
    plt.pcolormesh(
        time_array,  # Time axis as datetime objects
        np.arange(len(freq_cols)),  # Frequency levels (index)
        sound_matrix,  # Data matrix
        shading='auto',
        cmap='viridis'
    )
    plt.colorbar(label='Amplitude')
    plt.xlabel('Time')
    plt.ylabel('Frequencies')
    #plt.title('Spectrogram of Sound Frequencies Over Time')

    ax = plt.gca()  # Get current axes

    # Format x-axis as dates based on plot_case
    if plot_case == "A":  # 4 months (July-Oct 2023)
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=1, interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif plot_case == "B":  # 1 month (September 2023)
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))  # Every Monday
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif plot_case == "C":  # 1 week (Sep 18-25, 2023)
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))  # Every day
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif plot_case == "D":  # 1 day (Sep 20, 2023)
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))  # Every 3 hours, adjust as needed
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # Show Hour:Minute
        # To add the date to the x-axis label for the day plot:
        # date_str = df_plot[time_col].dt.date.min().strftime('%Y-%m-%d')
        # plt.xlabel(f'Time ({date_str})')
    else:  # Default formatter if no case matches
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.xticks(rotation=45)

    # Replace frequency level indices with actual frequency names for y-axis
    plt.yticks(ticks=np.arange(len(freq_cols)), labels=freq_cols, fontsize=8)

    # Add plot label (A, B, C, D)
    if plot_case:
        # Adjust x, y position and font size as needed to match Figure 35
        ax.text(0.95, 0.90, plot_case, transform=ax.transAxes,
                fontsize=20, fontweight='bold', va='top', ha='right', color='white')

    plt.tight_layout()
    # plt.show() # remove for automatically save images each turn

## Spectrogram ⅓ octave bands measured at LNM (A) 4 months, July through October.
start_date = "2023-07-01"
end_date = "2023-11-01"
# Robust date filtering
start_dt_A = pd.Timestamp(start_date, tz=mountain_tz)
end_dt_A = pd.Timestamp(end_date, tz=mountain_tz)
df_filtered_A = df[(df['time_mountain'] >= start_dt_A) & (df['time_mountain'] < end_dt_A)].copy() # Use .copy()

plot_spectrogram_with_dates(df_filtered_A, 'time_mountain', frequency_reduced, plot_case="A") # Pass plot_case
plot_filename = "spectrogram_4months_A.png" # Added _A to filename
try:
    plt.savefig(plot_filename)
    print(f"\nPlot file {plot_filename} successfully saved.")
except Exception as e:
    print(f"\nError saving plot: {e}")
plt.show()
plt.close()

## Spectrogram ⅓ octave bands measured at LNM (B) 1 month, September of 2023.
start_date = "2023-09-01"
end_date = "2023-10-01"
start_dt_B = pd.Timestamp(start_date, tz=mountain_tz)
end_dt_B = pd.Timestamp(end_date, tz=mountain_tz)
df_filtered_B = df[(df['time_mountain'] >= start_dt_B) & (df['time_mountain'] < end_dt_B)].copy()

plot_spectrogram_with_dates(df_filtered_B, 'time_mountain', frequency_reduced, plot_case="B") # Pass plot_case
plot_filename = "spectrogram_month_B.png" # Added _B to filename
try:
    plt.savefig(plot_filename)
    print(f"\nPlot file {plot_filename} successfully saved.")
except Exception as e:
    print(f"\nError saving plot: {e}")
plt.show()
plt.close()

## Spectrogram ⅓ octave bands measured at LNM (C) 1 week, September 18–25, 2023
# Note: Figure 35 caption says (C) 1 week, September 18–25, 2023.
# Your end_date "2023-09-26" is correct to include the 25th (up to midnight).
start_date = "2023-09-18"
end_date = "2023-09-26" # Includes up to 2023-09-25 23:59:59...
start_dt_C = pd.Timestamp(start_date, tz=mountain_tz)
end_dt_C = pd.Timestamp(end_date, tz=mountain_tz)
df_filtered_C = df[(df['time_mountain'] >= start_dt_C) & (df['time_mountain'] < end_dt_C)].copy()

plot_spectrogram_with_dates(df_filtered_C, 'time_mountain', frequency_reduced, plot_case="C") # Pass plot_case
plot_filename = "spectrogram_week_C.png" # Added _C to filename
try:
    plt.savefig(plot_filename)
    print(f"\nPlot file {plot_filename} successfully saved.")
except Exception as e:
    print(f"\nError saving plot: {e}")
plt.show()
plt.close()

## Spectrogram ⅓ octave bands measured at LNM (D) 1 day, September 20, 2023.
start_date = "2023-09-20"
end_date = "2023-09-21" # Includes up to 2023-09-20 23:59:59...
start_dt_D = pd.Timestamp(start_date, tz=mountain_tz)
end_dt_D = pd.Timestamp(end_date, tz=mountain_tz)
df_filtered_D = df[(df['time_mountain'] >= start_dt_D) & (df['time_mountain'] < end_dt_D)].copy()

plot_spectrogram_with_dates(df_filtered_D, 'time_mountain', frequency_reduced, plot_case="D") # Pass plot_case
plot_filename = "spectrogram_day_D.png" # Added _D to filename
try:
    plt.savefig(plot_filename)
    print(f"\nPlot file {plot_filename} successfully saved.")
except Exception as e:
    print(f"\nError saving plot: {e}")
plt.show()
plt.close()