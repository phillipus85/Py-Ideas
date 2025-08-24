# -*- coding: utf-8 -*-
"""
Configuration module for...

# FIXME: adjust documentation to match the actual implementation.
# TODO: Add description of the module.
# TODO: check Q-Model formula and consistency with the theory.
# TODO: rename some of the models properly (e.g., M/M/c, M/M/c/K, etc.)
# TODO check for better and shorter names in the classes, methods and variables

*IMPORTANT:* Based on the theory from:

    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""
# TODO add Queue class with all its variants

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional
# import numpy as np
import math


# @dataclass
# class QueueModel(ABC):
#     pass


# @dataclass
# class MM1(QueueModel):
#     """M/M/1 queue model"""
#     pass


# @dataclass
# class MM1K(QueueModel):
#     """M/M/1/K queue model"""
#     pass


# @dataclass
# class MM1L(QueueModel):
#     """M/M/1/L queue model"""
#     pass

# @dataclass
# class MM1K(QueueModel):
#     """M/M/1/K queue model"""
#     pass




@dataclass
class BaseQueueModel(ABC):
    """Abstract base class for queueing theory models"""

    # Arrival rate
    arrival_rate: float

    # Service rate per server
    service_rate: float

    # Number of servers
    num_servers: int = 1

    # Utilization factor
    utilization: float = field(default=0.0, init=False)

    # Maximum system capacity
    max_capacity: Optional[int] = None

    # Average number of requests in the system
    avg_req_in_sys: float = field(default=0.0, init=False)

    # Average number of requests in queue
    avg_req_in_queue: float = field(default=0.0, init=False)

    # Average time a request spends in the system
    avg_time_in_sys: float = field(default=0.0, init=False)

    # Average time a request spends waiting in queue
    avg_time_in_queue: float = field(default=0.0, init=False)

    # Metrics storage
    performance_metrics: Dict[int, float] = field(
        default_factory=dict, init=False)

    def __post_init__(self):
        """Initial validation and metric calculation"""
        self._validate_basic_parameters()
        self._validate_parameters()
        self._calculate_metrics()

    def _validate_basic_parameters(self) -> None:
        """Validates basic parameters common to all queueing models"""
        if self.arrival_rate <= 0:
            raise ValueError("Arrival rate must be positive")
        if self.service_rate <= 0:
            raise ValueError("Service rate must be positive")
        if self.num_servers <= 0:
            raise ValueError("Number of servers must be positive")

    @abstractmethod
    def _validate_parameters(self) -> None:
        """Validates parameters specific to each queueing model"""
        pass

    @abstractmethod
    def _calculate_metrics(self) -> None:
        """Calculates analytical metrics for the queueing model"""
        pass

    @abstractmethod
    def probability_n_req(self, n: int) -> float:
        """Calculates P(n) - probability of having n requests in the system"""
        pass

    @abstractmethod
    def is_stable(self) -> bool:
        """Checks if the queueing system is stable"""
        pass

    def get_metrics(self) -> Dict[str, Any]:
        """Returns a dictionary with the calculated metrics"""
        return {
            'Average number in system': self.avg_req_in_sys,
            'Average number in queue': self.avg_req_in_queue,
            'Average time in system': self.avg_time_in_sys,
            'Average time in queue': self.avg_time_in_queue,
            'Server utilization': self.utilization,
        }

    def print_summary(self) -> None:
        """Prints a summary of the queueing system"""
        print(f"\n=== {self.__class__.__name__} System Summary ===")
        print(f"λ = {self.arrival_rate}, μ = {self.service_rate}")
        print(f"Servers (c) = {self.num_servers}")
        print(f"Capacity (K) = {self.max_capacity}")
        print(f"System {'STABLE' if self.is_stable() else 'UNSTABLE'}")

        for key, value in self.get_metrics().items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")


@dataclass
class MM1(BaseQueueModel):
    """M/M/1 queue system (1 server, infinite capacity)"""

    def _validate_parameters(self) -> None:
        """Validations specific to M/M/1"""

        if self.num_servers != 1:
            raise ValueError("M/M/1 must have exactly 1 server")
        if self.max_capacity is not None:
            raise ValueError("M/M/1 assumes infinite capacity")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ μ")

    def is_stable(self) -> bool:
        """Checks if the system is stable"""

        return self.arrival_rate < self.service_rate

    def calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/1"""

        self.utilization = self.arrival_rate / self.service_rate
        self.avg_req_in_sys = self.utilization / (1 - self.utilization)
        self.avg_req_in_queue = self.utilization ** 2 / (1 - self.utilization)
        self.avg_time_in_sys = self.avg_req_in_sys / self.arrival_rate
        self.avg_time_in_queue = self.avg_req_in_queue / self.arrival_rate

    def probability_n_req(self, n: int) -> float:
        """Calculates P(n) - probability of having n requests in the system for M/M/1"""

        if not self.is_stable():
            raise ValueError(
                "System is unstable: utilization must be < 1 for M/M/1")

        if n < 0:
            return 0.0

        _p_0 = 1 - self.utilization
        return (self.utilization ** n) * _p_0


@dataclass
class MMC(BaseQueueModel):
    """M/M/c queue system (c servers, infinite capacity)"""

    def validate_parameters(self) -> None:
        """Validations specific to M/M/C"""
        if self.num_servers < 1:
            raise ValueError("M/M/c requires at least one server")
        if self.max_capacity is not None:
            raise ValueError("M/M/c assumes infinite capacity")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ c x μ)")

    def is_stable(self) -> bool:
        """Checks if the system is stable"""

        return self.arrival_rate < self.num_servers * self.service_rate

    def calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/C"""
        self.utilization = self.arrival_rate / \
            (self.num_servers * self.service_rate)
        _a = self.arrival_rate / self.service_rate
        p_0 = self._calculate_p0()
        self.avg_req_in_queue = (p_0 * (_a ** self.num_servers) * self.utilization) / (
            math.factorial(self.num_servers) * ((1 - self.utilization) ** 2))
        self.avg_req_in_sys = self.avg_req_in_queue + _a
        self.avg_time_in_queue = self.avg_req_in_queue / self.arrival_rate
        self.avg_time_in_sys = self.avg_time_in_queue + 1 / self.service_rate

    def calculate_p0(self) -> float:
        """Calculates P(0) for M/M/c"""
        _a = self.arrival_rate / self.service_rate
        _sum1 = sum((_a ** i) / math.factorial(i)
                    for i in range(self.num_servers))
        _sum2 = (_a ** self.num_servers) / \
            (math.factorial(self.num_servers) * (1 - self.utilization))
        return 1 / (_sum1 + _sum2)

    def probability_n_req(self, n: int) -> float:
        """Calculates P(n) - probability of having n requests in the system"""
        if n < 0:
            return 0.0

        _a = self.arrival_rate / self.service_rate
        _p_0 = self._calculate_p0()

        if n < self.num_servers:
            return ((_a ** n) / math.factorial(n)) * _p_0
        else:
            return ((_a ** n) / (math.factorial(self.num_servers) * (self.num_servers ** (n - self.num_servers)))) * _p_0


@dataclass
class MM1L(BaseQueueModel):
    """M/M/1/L queueing system: 1 server, finite capacity L"""

    def validate_parameters(self) -> None:
        """Validations specific to M/M/1/L"""
        if self.num_servers != 1:
            raise ValueError("M/M/1/L requires exactly 1 server")
        if self.max_capacity is None or self.max_capacity < 1:
            raise ValueError("M/M/1/L requires a positive finite capacity L")

    def is_stable(self) -> bool:
        """Checks if the system is stable - M/M/1/L is always stable due to finite capacity"""
        return self.arrival_rate > 0 and self.service_rate > 0

    def probability_n_req(self, n: int) -> float:
        """Calculate P(n) for n=0,...,L in M/M/1/L"""
        if n < 0 or n > self.max_capacity:
            return 0.0

        _rho = self.arrival_rate / self.service_rate

        if _rho == 1:
            return 1 / (self.max_capacity + 1)
        else:
            _p_0 = (1 - _rho) / (1 - _rho**(self.max_capacity + 1))
            return (_rho**n) * _p_0

    def calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/1/L"""
        _rho = self.arrival_rate / self.service_rate
        _p_max_capacity = self.probability_n_req(self.max_capacity)
        _lambda_eff = self.arrival_rate * (1 - _p_max_capacity)

        self.utilization = _rho * (1 - _p_max_capacity)
        self.avg_req_in_sys = sum(n * self.probability_n_req(n)
                                  for n in range(self.max_capacity + 1))
        self.avg_req_in_queue = sum(
            max(0, n - 1) * self.probability_n_req(n) for n in range(self.max_capacity + 1))

        if _lambda_eff > 0:
            self.avg_time_in_sys = self.avg_req_in_sys / _lambda_eff
            self.avg_time_in_queue = self.avg_req_in_queue / _lambda_eff
        else:
            self.avg_time_in_sys = 0
            self.avg_time_in_queue = 0


@dataclass
class MMCL(BaseQueueModel):
    """M/M/c/L queueing system: c servers, finite capacity L"""

    def validate_parameters(self) -> None:
        """Validations specific to M/M/c/L"""
        if self.num_servers < 1:
            raise ValueError("M/M/c/L requires at least one server")
        if self.max_capacity is None or self.max_capacity < self.num_servers:
            raise ValueError(
                "M/M/c/L requires finite capacity L >= c (number of servers)")

    def is_stable(self) -> bool:
        """Checks if the system is stable - M/M/c/L is always stable due to finite capacity"""
        return self.arrival_rate > 0 and self.service_rate > 0

    def calculate_p0(self) -> float:
        """Calculates P(0) for M/M/c/L"""
        _rho = self.arrival_rate / self.service_rate

        _sum1 = sum((_rho**n) / math.factorial(n)
                    for n in range(self.num_servers))

        if self.arrival_rate == self.num_servers * self.service_rate:
            _sum2 = ((_rho**self.num_servers) / math.factorial(self.num_servers)
                     ) * (self.max_capacity - self.num_servers + 1)
        else:
            utilization = _rho / self.num_servers
            _sum2 = ((_rho**self.num_servers) / math.factorial(self.num_servers)) * \
                ((1 - utilization**(self.max_capacity -
                 self.num_servers + 1)) / (1 - utilization))

        return 1 / (_sum1 + _sum2)

    def probability_n_req(self, n: int) -> float:
        """Calculate P(n) for n=0,...,L in M/M/c/L"""
        if n < 0 or n > self.max_capacity:
            return 0.0

        _rho = self.arrival_rate / self.service_rate
        _p_0 = self._calculate_p0()

        if n < self.num_servers:
            return ((_rho**n) / math.factorial(n)) * _p_0
        else:
            return ((_rho**n) / (math.factorial(self.num_servers) * (self.num_servers**(n - self.num_servers)))) * _p_0

    def calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/c/L"""
        _p_max_capacity = self.probability_n_req(self.max_capacity)
        _lambda_eff = self.arrival_rate * (1 - _p_max_capacity)
        self.avg_req_in_sys = sum(n * self.probability_n_req(n)
                                  for n in range(self.max_capacity + 1))
        self.avg_req_in_queue = sum(max(
            0, n - self.num_servers) * self.probability_n_req(n) for n in range(self.max_capacity + 1))
        server_busy_prob = sum(min(n, self.num_servers) * self.probability_n_req(n)
                               for n in range(self.max_capacity + 1))
        self.utilization = server_busy_prob / self.num_servers
        if _lambda_eff > 0:
            self.avg_time_in_sys = self.avg_req_in_sys / _lambda_eff
            self.avg_time_in_queue = self.avg_req_in_queue / _lambda_eff
        else:
            self.avg_time_in_sys = 0
            self.avg_time_in_queue = 0
