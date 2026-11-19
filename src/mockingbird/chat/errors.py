class DeliveryBlocked(RuntimeError):
    """Delivery needs operator attention; never schedule a fresh automatic retry."""


class DeliverySuperseded(RuntimeError):
    """New Telegram messages made the draft stale immediately before sending."""


class DeliveryDeferred(RuntimeError):
    """A read failed before sending. Wait for transport recovery, not operator input."""
