import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
import scipy.stats as stats


class dendro:
    """
    A class for processing high-frequency dendrometer measurements of tree stems.
    This code accompanies the paper:
    
    "Tree Growth, Contraction, and Recovery: Disentangling Soil and Atmospheric Drought Effects"
    Erez Feuer, Yakir Preisler, Eyal Rotenberg, Dan Yakir and Yair Mau
    Plant, Cell & Environment (2025)

    The methods section gives a detailed description of the algorithms used in this class.
    
    This class standardizes raw input data to millimeter-scale diameter measurements,
    resampled at a consistent temporal resolution, and computes a suite of metrics. 

    Parameters:
    - data: pd.Series with a datetime index representing diameter-related measurements
    - units: the physical unit of the input data ('m', 'cm', 'mm', or 'um')
    - metric: the type of measurement ('diameter', 'circumference', or 'radius')
    - initial_diameter_in_mm: optional offset to add to values (in mm)
    - points_per_day: number of data points per day, for example 48 if the measurement frequency is once every half hour
    """
    def __init__(self, data, units='mm', metric='diameter', initial_diameter_in_mm=0, points_per_day=48):
        # store the number of data points expected per day
        self.points_per_day = points_per_day

        # preserve the name of the input series
        self.name = data.name

        # resample to a regular interval based on points per day
        self.data = self._resample_data(data)

        # convert from radius or circumference to diameter (if needed), then to mm
        self.data = self._convert_to_diameter(self.data, metric, initial_diameter_in_mm)
        self.data = self._convert_units_to_mm(self.data, units)

        # backup the clean (converted) series
        self.original_data = self.data

        # mark NA positions before filling
        self.na_flags = self.data.isna()

        # fill missing values to produce a continuous time series
        self.data = self.data.ffill().bfill()

        # mark the start and end of the original valid data
        self.start = data.first_valid_index()
        self.end = data.last_valid_index()

    def _resample_data(self, data):
        """Resample the input series to a uniform interval based on the defined number of points per day.
        Default is 48 points per day, which corresponds to half-hourly measurements.
        """
        # compute number of minutes between samples
        freq_minutes = int(24 * 60 / self.points_per_day)
        # resample to evenly spaced time steps using mean aggregation
        return data.resample(f'{freq_minutes}min').mean()

    def _convert_to_diameter(self, data, metric, initial_diameter_in_mm):
        """Convert the input data to diameter in mm, based on the specified measurement type."""
        # verify that the metric is supported
        if metric not in ['diameter', 'circumference', 'radius']:
            raise ValueError("metric must be one of 'diameter', 'circumference', or 'radius'")

        # convert to diameter depending on input type
        if metric == 'diameter':
            return data + initial_diameter_in_mm  # already in diameter
        elif metric == 'circumference':
            return data / np.pi + initial_diameter_in_mm  # convert circumference to diameter
        elif metric == 'radius':
            return data * 2 + initial_diameter_in_mm  # convert radius to diameter

    def _convert_units_to_mm(self, data, units):
        """Convert the input values to millimeters regardless of their original physical units."""
        # verify that the unit is supported
        if units not in ['m', 'cm', 'mm', 'um']:
            raise ValueError("units must be one of 'm', 'cm', 'mm', or 'um'")

        # convert all units to mm
        if units == 'm':
            return data * 1000  # meters to mm
        elif units == 'cm':
            return data * 10    # centimeters to mm
        elif units == 'mm':
            return data         # already in mm
        elif units == 'um':
            return data / 1000  # micrometers to mm

    def trend(self):
        """
        Compute the underlying growth trend by removing daily variation
        using a 1-day centered rolling mean.

        Returns in units of: mm
        """
        # Calculate the trend using a rolling window
        trend = self.data.rolling(window='1D', center=True, min_periods=1).mean()
        return trend

    def GRO(self, trend=True):
        """
        Compute the growth-related signal GRO: the expanding max
        of the diameter or trend signal.

        Returns in units of: mm
        """
        # create GRO (Accumulated growth assuming zerow growth model)
        if trend:
            return self.trend().expanding().max() 
        else:
            return self.data.expanding().max()

    def TWD(self, trend=True):
        """
        Compute Tree Water Deficit (TWD) as the difference between GRO
        and the trend (or raw data).

        Returns in units of: mm
        """
        # Tree water deficit
        if trend:
            return self.GRO(trend=True) - self.trend()
        else:
            return self.GRO() - self.data

    def GRO_rate(self, days_in_window=45, trend=True):
        """
        Computes the instantaneous rate of growth by computing the first
        derivative of the GRO signal using a Savitzky-Golay filter.

        The window size defines the time span used to estimate the slope.

        Returns in units of: mm per day
        """
        # returns in units of: mm per day
        points_per_window = int(self.points_per_day * days_in_window)
        index = self.data.index
        savgol_derivative = savgol_filter(x=self.GRO(trend=trend),
                                          window_length=points_per_window,
                                          polyorder=1,
                                          deriv=1) * self.points_per_day
        return pd.Series(savgol_derivative, index=index, name='GRO_rate')
    
    def TREX(self):
        """
        Compute the Tree REcovery indeX (TREX). This algorithm is the same as
        the one presented in the paper by Feuer et al. (2025).

        Returns in units of: day
        """
        # returns in units of: day
        TWD = self.TWD(trend=True).resample('D').first().ffill()

        TREX = pd.Series([0] * len(TWD), index=TWD.index)

        def calc_TREX(TREX_index, TWD_index):
            if TWD.iloc[TREX_index] == 0 or TWD_index < 0:
                return 0
            if TWD.iloc[TREX_index] > TWD.iloc[TWD_index]:
                return TREX.iloc[TWD_index] + 1
            return calc_TREX(TREX_index, TWD_index - 1)

        for TREX_index in range(1, len(TREX)):
            TREX.iloc[TREX_index] = calc_TREX(TREX_index, TREX_index - 1)

        return TREX
    
    def fraction_of_expanding_days(self, data_segment):
        """
        Compute the omega as the fraction of days with expansion 
        out of the total days in the given dbh measurement window.

        Returns in units of: ratio, dimensionless
        """
        # count how many non-NaN values exist in the window
        total_days = data_segment.count()  # Total valid days in the window
        if total_days == 0:
            return np.nan  # avoid division by zero if window is empty

        # calculate the day-to-day difference
        daily_diff = data_segment.diff()

        # count how many days show expansion (positive change)
        expansion_days = (daily_diff > 0).sum()

        # return the fraction of expanding days out of total
        return expansion_days / total_days
    
    def omega(self, days_in_window=45):
        """
        Apply the omega metric using a rolling window over daily data.
        Omega quantifies the proportion of expanding days within each window.

        Returns in units of: ratio, dimensionless
        """
        # resample to daily resolution and forward-fill missing values
        daily = self.data.resample('1D').first().ffill()
        # apply the omega function in a rolling window centered around each date
        return daily.rolling(window=f'{int(days_in_window)}D', center=True).apply(self.fraction_of_expanding_days)

    def slope(self, data_segment):
        """
        Compute the linear slope of the input time series segment using ordinary least squares.

        Returns in units of: same as data_segment units per time step
        """
        # create a time axis for regression (0, 1, 2, ...)
        x = np.arange(len(data_segment))
        # extract values from the input Series
        y = data_segment.values
        # perform linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # return only the slope
        return slope

    def rolling_dbh_slope(self, days_in_window=45):
        """
        Calculate the rolling slope of the stem diameter signal using a centered window.
        Scales the result to mm per day based on window size and measurement frequency.

        Returns in units of: mm per day
        """
        # apply slope calculation over rolling windows on self.data
        return self.data.rolling(window=f'{int(days_in_window)}D', center=True).apply(self.slope) * self.points_per_day

    def rolling_TREX_slope(self, days_in_window=45):
        """
        Calculate the rolling slope of the TREX signal using a centered window.

        Returns in units of: unitless
        """
        # compute TREX and apply slope function over rolling windows
        TREX = self.TREX()
        return TREX.rolling(f'{int(days_in_window)}D', center=True).apply(self.slope)

    def increasing_GRO(self, trex_threshold=7, look_ahead_threshold=2):
        """
        Detect periods of growth based on TREX dynamics:
        - growth is True if TREX < trex_threshold
        - growth is set to False trex_threshold days before any TREX ≥ trex_threshold
        - growth remains False until TREX ≤ look_ahead_threshold

        Parameters:
        - trex_threshold: upper threshold to define end of growth
        - look_ahead_threshold: TREX must fall below this to resume growth

        Returns in units of: boolean (growth flag)
        """
        TREX = self.TREX()
        # initialize growth as True when TREX is low
        growth = TREX < trex_threshold

        # find times where TREX crosses from growing to non-growing
        stops = TREX.index[(growth.shift(1) == True) & (growth == False)]

        # loop through each stop event
        for stop in stops:
            # set previous trex_threshold days to not growing
            start_time = stop - pd.Timedelta(days=trex_threshold)
            growth.loc[(growth.index >= start_time) & (growth.index < stop)] = False

            # find first time after stop where TREX drops low enough to resume growth
            lookahead_idx = TREX.loc[
                (TREX.index > stop) & (TREX <= look_ahead_threshold)
            ].index

            if not lookahead_idx.empty:
                reset_time = lookahead_idx[0]
                # keep growth=False between stop and reset
                growth.loc[(growth.index > stop) & (growth.index < reset_time)] = False

        return growth
    

    def df_all(self, days_in_window=45, trex_threshold=7, look_ahead_threshold=2):
        """
        Calculate and return all statistics and core signals in a single DataFrame,
        aligning daily-frequency variables to original sampling via forward-fill.

        Returns in units of: mixed (depends on individual signals)
        """
        # keep a copy of the original (possibly gappy) input data
        original_data = self.data
        # filled and standardized diameter signal
        data = self.data
        # boolean mask of originally missing values
        na_flags = self.na_flags

        # daily-scale signals from raw or trend
        trend = self.trend()
        GRO = self.GRO()
        TWD = self.TWD()
        GRO_rate = self.GRO_rate()
        rolling_dbh_slope = self.rolling_dbh_slope(days_in_window)
        increasing_GRO = self.increasing_GRO(trex_threshold=trex_threshold, look_ahead_threshold=look_ahead_threshold)

        # signals that are computed at daily frequency and need to be aligned
        TREX = self.TREX()
        omega = self.omega(days_in_window)
        TREX_slope = self.rolling_TREX_slope(days_in_window)
        

        # use the high-frequency index to align all signals
        target_index = data.index

        # construct final DataFrame, aligning daily signals via forward-fill
        return pd.DataFrame({
            'original_data': original_data,
            'data': data,
            'na_flags': na_flags,
            'trend': trend,
            'GRO': GRO,
            'TWD': TWD,
            'GRO_rate': GRO_rate,
            'rolling_dbh_slope': rolling_dbh_slope,
            'TREX': TREX.reindex(target_index).ffill(),
            'omega': omega.reindex(target_index).ffill(),
            'TREX_slope': TREX_slope.reindex(target_index).ffill(),
            'increasing_GRO': increasing_GRO.reindex(target_index).ffill(),
        }, index=target_index)




