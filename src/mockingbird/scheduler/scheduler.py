from datetime import timedelta

from mockingbird.conversation.history import History
from mockingbird.scheduler.burst import Burst
from mockingbird.scheduler.busy import Busy
from mockingbird.scheduler.initiation import Initiation
from mockingbird.scheduler.response_delay import ResponseDelay
from mockingbird.scheduler.sleep import Sleep



class Scheduler:
    def __init__(
        self,
        history: History,
        busy_bin_size: timedelta = timedelta(minutes=15),
        conversation_window: timedelta = timedelta(minutes=10),
        busy_max_duration: timedelta = timedelta(hours=3),
        busy_randomness_interval: timedelta = timedelta(minutes=30),
        burst_window: timedelta = timedelta(minutes=10),
        burst_delay_bin_size: timedelta = timedelta(seconds=1),
        burst_smoothing_sigma: timedelta = timedelta(seconds=2),
        response_delay_window: timedelta = timedelta(minutes=10),
        response_delay_bin_size: timedelta = timedelta(seconds=1),
        response_delay_smoothing_sigma: timedelta = timedelta(seconds=2),
        initiation_bin_size: timedelta = timedelta(minutes=15),
        initiation_smoothing_sigma: timedelta = timedelta(minutes=15),
        sleep_bin_size: timedelta = timedelta(minutes=15),
        sleep_lower_percentile: float = 0.80,
        sleep_upper_percentile: float = 0.99,
        sleep_start_skew: tuple[float, float] = (2, 5),
        sleep_end_skew: tuple[float, float] = (2, 5),
        sleep_randomness_interval: timedelta = timedelta(minutes=30),
    ) -> None:
        self.available: bool = False
        self.in_conversation: bool = False

        self.busy = Busy(
            history,
            bin_size=busy_bin_size,
            conversation_window=conversation_window,
            max_duration=busy_max_duration,
            randomness_interval=busy_randomness_interval,
        )
        self.burst = Burst(
            history,
            window=burst_window,
            delay_bin_size=burst_delay_bin_size,
            smoothing_sigma=burst_smoothing_sigma,
        )
        self.response_delay = ResponseDelay(
            history,
            window=response_delay_window,
            bin_size=response_delay_bin_size,
            smoothing_sigma=response_delay_smoothing_sigma,
        )
        self.initiation = Initiation(
            history,
            bin_size=initiation_bin_size,
            conversation_window=conversation_window,
            smoothing_sigma=initiation_smoothing_sigma,
        )
        self.sleep = Sleep(
            self.initiation.count_distribution,
            bin_size=sleep_bin_size,
            lower_percentile=sleep_lower_percentile,
            upper_percentile=sleep_upper_percentile,
            start_skew=sleep_start_skew,
            end_skew=sleep_end_skew,
            randomness_interval=sleep_randomness_interval,
        )
