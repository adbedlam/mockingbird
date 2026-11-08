import heapq
from datetime import datetime, timedelta

from mockingbird.conversation.history import History
from mockingbird.event import Event, EventType
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
        self.busy = Busy(
            history,
            bin_size=busy_bin_size,
            conversation_window=conversation_window,
            max_busy_duration=busy_max_duration,
            randomness_interval=busy_randomness_interval,
        )
        self.burst = Burst(
            history,
            burst_window=burst_window,
            delay_bin_size=burst_delay_bin_size,
            smoothing_sigma=burst_smoothing_sigma,
        )
        self.response_delay = ResponseDelay(
            history,
            response_window=response_delay_window,
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

        self.queue: list[Event] = []

    def generate_schedule(self):
        self.queue.clear()

        signals: list[Event] = []

        today = datetime.now().date()
        weekday = today.weekday()

        busy_intervals = self.busy.generate(weekday)

        for start, end in busy_intervals:
            signals.append(
                Event(
                    timestamp=datetime.combine(today, start),
                    type=EventType.BECOME_BUSY,
                )
            )

            signals.append(
                Event(
                    timestamp=datetime.combine(today, end),
                    type=EventType.BECOME_AVAILABLE,
                )
            )

        wake_time, sleep_time = self.sleep.generate(weekday)

        signals.append(
            Event(timestamp=datetime.combine(today, wake_time), type=EventType.WAKE_UP)
        )

        signals.append(
            Event(
                timestamp=datetime.combine(today, sleep_time),
                type=EventType.GO_TO_SLEEP,
            )
        )

        initiations = self.initiation.generate(weekday)

        for initiation in initiations:
            message_time = datetime.combine(today, initiation)

            signals.append(
                Event(
                    timestamp=message_time,
                    type=EventType.SEND_MESSAGE,
                )
            )

            delays = self.burst.generate(weekday)

            for delay in delays:
                message_time += delay

                signals.append(
                    Event(
                        timestamp=message_time,
                        type=EventType.SEND_MESSAGE,
                    )
                )

        for signal in signals:
            heapq.heappush(self.queue, signal)
