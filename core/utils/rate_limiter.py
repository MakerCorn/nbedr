"""
Rate limiting utilities for managing API request rates to cloud-based AI services.

This module provides flexible rate limiting capabilities to handle various rate limit
constraints imposed by cloud-based AI services like OpenAI, Azure OpenAI, and others.
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, Callable
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class RateLimitStrategy(Enum):
    """Available rate limiting strategies."""
    FIXED_WINDOW = "fixed_window"       # Fixed time window
    SLIDING_WINDOW = "sliding_window"   # Sliding time window  
    TOKEN_BUCKET = "token_bucket"       # Token bucket algorithm
    ADAPTIVE = "adaptive"               # Adaptive based on response times

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    # Basic rate limiting
    enabled: bool = False
    strategy: RateLimitStrategy = RateLimitStrategy.SLIDING_WINDOW
    
    # Request rate limits
    requests_per_minute: Optional[int] = None
    requests_per_hour: Optional[int] = None
    requests_per_day: Optional[int] = None
    
    # Token-based rate limits (for AI services)
    tokens_per_minute: Optional[int] = None
    tokens_per_hour: Optional[int] = None
    tokens_per_day: Optional[int] = None
    
    # Burst handling
    max_burst_requests: Optional[int] = None
    burst_window_seconds: float = 60.0
    
    # Adaptive rate limiting
    target_response_time: float = 2.0   # Target response time in seconds
    max_response_time: float = 10.0     # Max acceptable response time
    adaptation_factor: float = 0.1      # How quickly to adapt (0.0 to 1.0)
    
    # Retry configuration
    max_retries: int = 3
    base_retry_delay: float = 1.0       # Base delay in seconds
    max_retry_delay: float = 60.0       # Maximum retry delay
    exponential_backoff: bool = True
    jitter: bool = True                 # Add randomness to retry delays
    
    # Error handling
    retry_on_rate_limit: bool = True
    retry_on_server_error: bool = True
    fail_fast_on_auth_error: bool = True

class RateLimiter:
    """Flexible rate limiter supporting multiple strategies."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self._lock = threading.RLock()
        
        # Request tracking
        self._request_times = deque()
        self._token_usage = deque()
        
        # Token bucket state
        self._tokens = 0.0
        self._last_refill = time.time()
        
        # Adaptive rate limiting state
        self._response_times = deque(maxlen=100)
        self._current_rate_limit = config.requests_per_minute or 60
        self._last_adaptation = time.time()
        
        # Statistics
        self._total_requests = 0
        self._total_tokens = 0
        self._total_wait_time = 0.0
        self._rate_limit_hits = 0
        
        if config.enabled:
            logger.info(f"Rate limiter enabled with strategy: {config.strategy.value}")
    
    def acquire(self, estimated_tokens: Optional[int] = None) -> float:
        """
        Acquire permission to make a request.
        
        Args:
            estimated_tokens: Estimated tokens for this request
            
        Returns:
            Delay time in seconds that was applied
        """
        if not self.config.enabled:
            return 0.0
        
        with self._lock:
            delay = self._calculate_delay(estimated_tokens)
            if delay > 0:
                logger.debug(f"Rate limiting: sleeping for {delay:.2f}s")
                time.sleep(delay)
                self._total_wait_time += delay
                self._rate_limit_hits += 1
            
            # Record the request
            now = time.time()
            self._request_times.append(now)
            if estimated_tokens:
                self._token_usage.append((now, estimated_tokens))
            
            self._total_requests += 1
            if estimated_tokens:
                self._total_tokens += estimated_tokens
            
            return delay
    
    def record_response(self, response_time: float, actual_tokens: Optional[int] = None):
        """
        Record response information for adaptive rate limiting.
        
        Args:
            response_time: Response time in seconds
            actual_tokens: Actual tokens used (if different from estimate)
        """
        if not self.config.enabled:
            return
        
        with self._lock:
            self._response_times.append(response_time)
            
            if actual_tokens and self._token_usage:
                # Update the last token record with actual usage
                if self._token_usage:
                    last_time, estimated = self._token_usage[-1]
                    self._token_usage[-1] = (last_time, actual_tokens)
                    self._total_tokens += (actual_tokens - estimated)
            
            # Adaptive rate limiting adjustment
            if self.config.strategy == RateLimitStrategy.ADAPTIVE:
                self._adapt_rate_limit(response_time)
    
    def record_error(self, error_type: str, retry_after: Optional[float] = None):
        """
        Record an error for rate limiting adjustments.
        
        Args:
            error_type: Type of error (rate_limit, server_error, etc.)
            retry_after: Retry-After header value if available
        """
        if not self.config.enabled:
            return
        
        with self._lock:
            if error_type == "rate_limit":
                self._rate_limit_hits += 1
                if retry_after:
                    logger.info(f"Rate limit hit, server suggests waiting {retry_after}s")
                    time.sleep(retry_after)
                elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
                    # Reduce rate for adaptive strategy
                    self._current_rate_limit = max(1, self._current_rate_limit * 0.8)
                    logger.info(f"Reduced adaptive rate limit to {self._current_rate_limit}")
    
    def _calculate_delay(self, estimated_tokens: Optional[int] = None) -> float:
        """Calculate how long to wait before making the next request."""
        now = time.time()
        
        if self.config.strategy == RateLimitStrategy.FIXED_WINDOW:
            return self._fixed_window_delay(now)
        elif self.config.strategy == RateLimitStrategy.SLIDING_WINDOW:
            return self._sliding_window_delay(now, estimated_tokens)
        elif self.config.strategy == RateLimitStrategy.TOKEN_BUCKET:
            return self._token_bucket_delay(now, estimated_tokens)
        elif self.config.strategy == RateLimitStrategy.ADAPTIVE:
            return self._adaptive_delay(now, estimated_tokens)
        else:
            return 0.0
    
    def _fixed_window_delay(self, now: float) -> float:
        """Calculate delay for fixed window strategy."""
        if not self.config.requests_per_minute:
            return 0.0
        
        # Clean requests older than 1 minute
        minute_ago = now - 60
        while self._request_times and self._request_times[0] < minute_ago:
            self._request_times.popleft()
        
        if len(self._request_times) >= self.config.requests_per_minute:
            # Need to wait until the oldest request is more than 1 minute old
            oldest_time = self._request_times[0]
            return max(0, 60 - (now - oldest_time))
        
        return 0.0
    
    def _sliding_window_delay(self, now: float, estimated_tokens: Optional[int] = None) -> float:
        """Calculate delay for sliding window strategy."""
        max_delay = 0.0
        
        # Check request rate limits
        if self.config.requests_per_minute:
            minute_ago = now - 60
            while self._request_times and self._request_times[0] < minute_ago:
                self._request_times.popleft()
            
            if len(self._request_times) >= self.config.requests_per_minute:
                time_until_slot_free = 60 - (now - self._request_times[0])
                max_delay = max(max_delay, time_until_slot_free)
        
        # Check token rate limits
        if estimated_tokens and self.config.tokens_per_minute:
            minute_ago = now - 60
            while self._token_usage and self._token_usage[0][0] < minute_ago:
                self._token_usage.popleft()
            
            current_tokens = sum(tokens for _, tokens in self._token_usage)
            if current_tokens + estimated_tokens > self.config.tokens_per_minute:
                # Find when enough tokens will be available
                needed_tokens = (current_tokens + estimated_tokens) - self.config.tokens_per_minute
                for timestamp, tokens in self._token_usage:
                    needed_tokens -= tokens
                    if needed_tokens <= 0:
                        time_until_tokens_free = 60 - (now - timestamp)
                        max_delay = max(max_delay, time_until_tokens_free)
                        break
        
        # Check burst limits
        if self.config.max_burst_requests:
            burst_window_ago = now - self.config.burst_window_seconds
            burst_requests = sum(1 for t in self._request_times if t > burst_window_ago)
            
            if burst_requests >= self.config.max_burst_requests:
                oldest_in_window = next((t for t in self._request_times if t > burst_window_ago), now)
                time_until_burst_free = self.config.burst_window_seconds - (now - oldest_in_window)
                max_delay = max(max_delay, time_until_burst_free)
        
        return max_delay
    
    def _token_bucket_delay(self, now: float, estimated_tokens: Optional[int] = None) -> float:
        """Calculate delay for token bucket strategy."""
        if not self.config.requests_per_minute:
            return 0.0
        
        # Refill tokens
        time_passed = now - self._last_refill
        tokens_to_add = (self.config.requests_per_minute / 60.0) * time_passed
        self._tokens = min(self.config.requests_per_minute, self._tokens + tokens_to_add)
        self._last_refill = now
        
        # Check if we have enough tokens
        tokens_needed = 1
        if estimated_tokens and self.config.tokens_per_minute:
            # Scale token cost based on estimated token usage
            token_cost_ratio = estimated_tokens / (self.config.tokens_per_minute / 60.0)
            tokens_needed = max(1, token_cost_ratio)
        
        if self._tokens >= tokens_needed:
            self._tokens -= tokens_needed
            return 0.0
        else:
            # Calculate how long to wait for enough tokens
            tokens_needed_to_wait = tokens_needed - self._tokens
            wait_time = tokens_needed_to_wait / (self.config.requests_per_minute / 60.0)
            return wait_time
    
    def _adaptive_delay(self, now: float, estimated_tokens: Optional[int] = None) -> float:
        """Calculate delay for adaptive strategy."""
        # Use sliding window as base, but adjust rate based on response times
        delay = self._sliding_window_delay(now, estimated_tokens)
        
        # Adjust the current rate limit based on recent response times
        if now - self._last_adaptation > 30:  # Adapt every 30 seconds
            self._adapt_rate_limit()
            self._last_adaptation = now
        
        return delay
    
    def _adapt_rate_limit(self, latest_response_time: Optional[float] = None):
        """Adapt the rate limit based on response times."""
        if not self._response_times:
            return
        
        # Calculate average response time
        avg_response_time = sum(self._response_times) / len(self._response_times)
        
        if latest_response_time:
            # Weight recent response time more heavily
            avg_response_time = (avg_response_time * 0.7) + (latest_response_time * 0.3)
        
        # Adjust rate limit based on response time
        if avg_response_time > self.config.max_response_time:
            # Slow down significantly
            adjustment = -self.config.adaptation_factor * 2
        elif avg_response_time > self.config.target_response_time:
            # Slow down gradually
            adjustment = -self.config.adaptation_factor
        elif avg_response_time < self.config.target_response_time * 0.5:
            # Speed up
            adjustment = self.config.adaptation_factor
        else:
            # Maintain current rate
            adjustment = 0
        
        old_rate = self._current_rate_limit
        self._current_rate_limit = max(1, self._current_rate_limit * (1 + adjustment))
        
        if abs(old_rate - self._current_rate_limit) > 1:
            logger.debug(f"Adapted rate limit from {old_rate:.1f} to {self._current_rate_limit:.1f} RPM "
                        f"(avg response time: {avg_response_time:.2f}s)")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self._lock:
            recent_response_times = list(self._response_times)
            avg_response_time = sum(recent_response_times) / len(recent_response_times) if recent_response_times else 0
            
            return {
                'enabled': self.config.enabled,
                'strategy': self.config.strategy.value,
                'total_requests': self._total_requests,
                'total_tokens': self._total_tokens,
                'total_wait_time': self._total_wait_time,
                'rate_limit_hits': self._rate_limit_hits,
                'average_response_time': avg_response_time,
                'current_rate_limit': self._current_rate_limit if self.config.strategy == RateLimitStrategy.ADAPTIVE else self.config.requests_per_minute,
                'requests_in_last_minute': len([t for t in self._request_times if time.time() - t < 60]),
                'tokens_in_last_minute': sum(tokens for timestamp, tokens in self._token_usage if time.time() - timestamp < 60)
            }

def create_rate_limiter_from_config(
    enabled: bool = False,
    strategy: str = "sliding_window",
    requests_per_minute: Optional[int] = None,
    requests_per_hour: Optional[int] = None,
    tokens_per_minute: Optional[int] = None,
    tokens_per_hour: Optional[int] = None,
    max_burst_requests: Optional[int] = None,
    burst_window_seconds: float = 60.0,
    target_response_time: float = 2.0,
    max_response_time: float = 10.0,
    max_retries: int = 3,
    base_retry_delay: float = 1.0,
    **kwargs
) -> RateLimiter:
    """
    Create a rate limiter from configuration parameters.
    
    This is a convenience function for creating rate limiters with common configurations.
    """
    try:
        strategy_enum = RateLimitStrategy(strategy)
    except ValueError:
        logger.warning(f"Unknown rate limit strategy '{strategy}', using sliding_window")
        strategy_enum = RateLimitStrategy.SLIDING_WINDOW
    
    config = RateLimitConfig(
        enabled=enabled,
        strategy=strategy_enum,
        requests_per_minute=requests_per_minute,
        requests_per_hour=requests_per_hour,
        tokens_per_minute=tokens_per_minute,
        tokens_per_hour=tokens_per_hour,
        max_burst_requests=max_burst_requests,
        burst_window_seconds=burst_window_seconds,
        target_response_time=target_response_time,
        max_response_time=max_response_time,
        max_retries=max_retries,
        base_retry_delay=base_retry_delay,
        **kwargs
    )
    
    return RateLimiter(config)

def get_common_rate_limits() -> Dict[str, Dict[str, Any]]:
    """
    Get common rate limit configurations for popular AI services.
    
    Returns:
        Dictionary with service names as keys and rate limit configs as values
    """
    return {
        'openai_gpt4': {
            'requests_per_minute': 500,
            'tokens_per_minute': 10000,
            'strategy': 'sliding_window',
            'max_burst_requests': 50
        },
        'openai_gpt35_turbo': {
            'requests_per_minute': 3500,
            'tokens_per_minute': 90000,
            'strategy': 'sliding_window',
            'max_burst_requests': 100
        },
        'azure_openai_standard': {
            'requests_per_minute': 120,
            'tokens_per_minute': 6000,
            'strategy': 'sliding_window',
            'max_burst_requests': 20
        },
        'anthropic_claude': {
            'requests_per_minute': 1000,
            'tokens_per_minute': 100000,
            'strategy': 'sliding_window',
            'max_burst_requests': 50
        },
        'conservative': {
            'requests_per_minute': 60,
            'tokens_per_minute': 2000,
            'strategy': 'sliding_window',
            'max_burst_requests': 10
        },
        'aggressive': {
            'requests_per_minute': 1000,
            'tokens_per_minute': 50000,
            'strategy': 'adaptive',
            'max_burst_requests': 100
        }
    }