"""Load Predictor for v0.5-α Predictive Controller.

Uses time-series analysis to forecast arrival rate:
- Simple Moving Average (SMA) for baseline prediction
- Trend detection via arrival_velocity (rate of change)
- Surge prediction when velocity exceeds threshold

Source: SageServe ARIMA-based forecasting approach for LLM serving.
"""

from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple
import time


@dataclass
class PredictionResult:
    """Result of a load prediction."""
    predicted_rps: float           # Forecasted requests per second
    arrival_velocity: float        # Rate of change (rps/sec)
    confidence: float              # 0.0-1.0 confidence in prediction
    surge_detected: bool           # True if velocity > surge threshold
    horizon_sec: float             # Prediction horizon used


class LoadPredictor:
    """Forecasts arrival rate using time-series analysis.
    
    Implements the SageServe-style prediction logic:
    1. Track rolling history of arrival rates
    2. Calculate Simple Moving Average (SMA) as baseline
    3. Detect trend via arrival_velocity
    4. Predict surge when velocity > threshold
    """
    
    def __init__(self, 
                 window_size: int = 30,
                 surge_velocity_threshold: float = 2.0):
        """
        Args:
            window_size: Number of samples to track (at 1 sample/sec = 30 sec window)
            surge_velocity_threshold: Velocity (rps/sec) above which surge is detected
        """
        # Rolling history of (timestamp, arrival_rate) pairs
        self.history = deque(maxlen=window_size)
        self.window_size = window_size
        self.surge_threshold = surge_velocity_threshold
        
        # Tracking for forecast accuracy
        self.predictions = deque(maxlen=100)  # (predicted, actual) pairs
    
    def record(self, arrival_rate: float, timestamp: float = None) -> None:
        """Record a new arrival rate observation.
        
        Args:
            arrival_rate: Current requests per second
            timestamp: When this was observed (defaults to now)
        """
        if timestamp is None:
            timestamp = time.time()
        self.history.append((timestamp, arrival_rate))
    
    def predict(self, horizon_sec: float = 10.0) -> PredictionResult:
        """Forecast arrival rate for t+horizon.
        
        Uses:
        1. SMA for baseline
        2. Linear extrapolation using velocity
        
        Args:
            horizon_sec: How far ahead to predict (seconds)
            
        Returns:
            PredictionResult with forecast and confidence
        """
        if len(self.history) < 2:
            # Not enough data - return current or zero
            current = self.history[-1][1] if self.history else 0.0
            return PredictionResult(
                predicted_rps=current,
                arrival_velocity=0.0,
                confidence=0.1,
                surge_detected=False,
                horizon_sec=horizon_sec
            )
        
        # Calculate Simple Moving Average (baseline)
        rates = [r for _, r in self.history]
        sma = sum(rates) / len(rates)
        
        # Calculate arrival_velocity (trend)
        # Use first and last quartile to smooth noise
        n = len(self.history)
        if n >= 4:
            early_samples = list(self.history)[:n//4]
            late_samples = list(self.history)[-(n//4):]
            
            early_avg = sum(r for _, r in early_samples) / len(early_samples)
            late_avg = sum(r for _, r in late_samples) / len(late_samples)
            
            time_span = late_samples[-1][0] - early_samples[0][0]
            velocity = (late_avg - early_avg) / max(1.0, time_span)
        else:
            # Simple two-point velocity
            oldest = self.history[0]
            newest = self.history[-1]
            time_span = newest[0] - oldest[0]
            velocity = (newest[1] - oldest[1]) / max(1.0, time_span)
        
        # Linear extrapolation: predicted = current + velocity * horizon
        current_rate = self.history[-1][1]
        predicted_rps = max(0.0, current_rate + velocity * horizon_sec)
        
        # Surge detection
        surge_detected = velocity > self.surge_threshold
        
        # Confidence based on data freshness and variance
        confidence = min(1.0, len(self.history) / self.window_size)
        if len(rates) > 2:
            # Lower confidence if high variance
            import statistics
            try:
                variance = statistics.variance(rates)
                if variance > sma * 2:  # High relative variance
                    confidence *= 0.5
            except:
                pass
        
        return PredictionResult(
            predicted_rps=predicted_rps,
            arrival_velocity=velocity,
            confidence=confidence,
            surge_detected=surge_detected,
            horizon_sec=horizon_sec
        )
    
    def record_actual(self, predicted: float, actual: float) -> None:
        """Record a prediction/actual pair for accuracy tracking."""
        self.predictions.append((predicted, actual))
    
    def get_forecast_accuracy(self) -> float:
        """Calculate Mean Absolute Percentage Error (MAPE).
        
        Returns:
            MAPE as percentage (lower is better)
        """
        if not self.predictions:
            return 0.0
        
        errors = []
        for predicted, actual in self.predictions:
            if actual > 0:
                error = abs(predicted - actual) / actual
                errors.append(error)
        
        return (sum(errors) / len(errors)) * 100 if errors else 0.0
    
    def get_stats(self) -> dict:
        """Return predictor statistics."""
        prediction = self.predict()
        return {
            "current_sma": sum(r for _, r in self.history) / len(self.history) if self.history else 0,
            "arrival_velocity": prediction.arrival_velocity,
            "surge_detected": prediction.surge_detected,
            "forecast_accuracy_mape": self.get_forecast_accuracy(),
            "sample_count": len(self.history),
        }
