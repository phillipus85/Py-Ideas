# -*- coding: utf-8 -*-
"""
Module queues.py
===========================================

This module provides queuing theory models with a unified interface.

Usage:
    # Create a simple M/M/1 queue
    queue = QueueModel.create(arrival_rate=5, service_rate=10)

    # Create a multi-server M/M/c queue
    queue = QueueModel.create(arrival_rate=15, service_rate=5, num_servers=4)

    # Create a limited capacity queue M/M/1/K
    queue = QueueModel.create(arrival_rate=5, service_rate=10, max_capacity=20)

    # Create a multi-server limited capacity queue M/M/c/K
    queue = QueueModel.create(arrival_rate=15, service_rate=5, num_servers=4, max_capacity=30)

*IMPORTANT:* Based on the theory from:
    # H.Gorter, *Dimensionalanalyse: Eine Theoririe der physikalischen Dimensionen mit Anwendungen*
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, ClassVar
from enum import Enum
import math


class QueueType(Enum):
    """Enumeration of supported queue model types."""
    MM1 = "M/M/1"     # Single server, infinite capacity
    MMS = "M/M/s"     # Multiple servers, infinite capacity
    MM1K = "M/M/1/K"  # Single server, finite capacity
    MMSK = "M/M/s/K"  # Multiple servers, finite capacity


@dataclass
class BaseQueueModel(ABC):
    """Abstract base class for queueing theory models"""

    # Arrival rate (λ)
    arrival_rate: float

    # Service rate per server (μ)
    service_rate: float

    # Number of servers (s or c)
    num_servers: int = 1

    # Utilization factor (ρ)
    utilization: float = field(default=0.0, init=False)

    # Maximum system capacity (K)
    max_capacity: Optional[int] = None

    # Average number of requests in the system (L)
    avg_req_in_sys: float = field(default=0.0, init=False)

    # Average number of requests in queue (Lq)
    avg_req_in_queue: float = field(default=0.0, init=False)

    # Average time a request spends in the system (W)
    avg_time_in_sys: float = field(default=0.0, init=False)

    # Average time a request spends waiting in queue (Wq)
    avg_time_in_queue: float = field(default=0.0, init=False)

    # Queue type
    queue_type: ClassVar[QueueType]

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
        print(f"\n=== {self.queue_type.value} System Summary ===")
        print(f"λ = {self.arrival_rate}, μ = {self.service_rate}")
        print(f"Servers (c) = {self.num_servers}")
        if self.max_capacity is not None:
            print(f"Capacity (K) = {self.max_capacity}")
        print(f"System {'STABLE' if self.is_stable() else 'UNSTABLE'}")

        for key, value in self.get_metrics().items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")


@dataclass
class MM1Model(BaseQueueModel):
    """M/M/1 queue system (1 server, infinite capacity)"""

    queue_type: ClassVar[QueueType] = QueueType.MM1

    def _validate_parameters(self) -> None:
        """Validations specific to M/M/1"""
        if self.num_servers != 1:
            raise ValueError("M/M/1 must have exactly 1 server")
        if self.max_capacity is not None:
            raise ValueError("M/M/1 assumes infinite capacity")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ μ)")

    def is_stable(self) -> bool:
        """Checks if the system is stable"""
        return self.arrival_rate < self.service_rate

    def _calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/1"""
        self.utilization = self.arrival_rate / self.service_rate
        self.avg_req_in_sys = self.utilization / (1 - self.utilization)
        self.avg_req_in_queue = self.utilization ** 2 / (1 - self.utilization)
        self.avg_time_in_sys = 1 / (self.service_rate - self.arrival_rate)
        self.avg_time_in_queue = self.arrival_rate / (self.service_rate * (self.service_rate - self.arrival_rate))

    def probability_n_req(self, n: int) -> float:
        """Calculates P(n) - probability of having n requests in the system for M/M/1"""
        if not self.is_stable():
            raise ValueError("System is unstable: utilization must be < 1 for M/M/1")
        if n < 0:
            return 0.0
        _p_0 = 1 - self.utilization
        return (self.utilization ** n) * _p_0


@dataclass
class MMsModel(BaseQueueModel):
    """M/M/s queue system (s servers, infinite capacity)"""

    queue_type: ClassVar[QueueType] = QueueType.MMS

    def _validate_parameters(self) -> None:
        """Validations specific to M/M/s"""
        if self.num_servers < 1:
            raise ValueError("M/M/s requires at least one server")
        if self.max_capacity is not None:
            raise ValueError("M/M/s assumes infinite capacity")
        if not self.is_stable():
            raise ValueError("System is unstable (λ ≥ s × μ)")

    def is_stable(self) -> bool:
        """Checks if the system is stable"""
        return self.arrival_rate < self.num_servers * self.service_rate

    def _calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/s"""
        self.utilization = self.arrival_rate / (self.num_servers * self.service_rate)
        _a = self.arrival_rate / self.service_rate  # traffic intensity
        _p_0 = self._calculate_p0()

        # Erlang C formula for probability of waiting
        _p_wait = ((_a ** self.num_servers) / math.factorial(self.num_servers)) * (_p_0 / (1 - self.utilization))
        # Average queue length
        self.avg_req_in_queue = _p_wait * (_a * self.utilization) / (1 - self.utilization)

        # Average system length
        self.avg_req_in_sys = self.avg_req_in_queue + _a

        # Average waiting time
        self.avg_time_in_queue = self.avg_req_in_queue / self.arrival_rate

        # Average system time
        self.avg_time_in_sys = self.avg_time_in_queue + (1 / self.service_rate)

    def _calculate_p0(self) -> float:
        """Calculates P(0) for M/M/s"""
        _a = self.arrival_rate / self.service_rate
        _sum1 = sum((_a ** i) / math.factorial(i) for i in range(self.num_servers))
        _sum2 = (_a ** self.num_servers) / (math.factorial(self.num_servers) * (1 - self.utilization))
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
class MM1KModel(BaseQueueModel):
    """M/M/1/K queueing system: 1 server, finite capacity K"""

    queue_type: ClassVar[QueueType] = QueueType.MM1K

    def _validate_parameters(self) -> None:
        """Validations specific to M/M/1/K"""
        if self.num_servers != 1:
            raise ValueError("M/M/1/K requires exactly 1 server")
        if self.max_capacity is None or self.max_capacity < 1:
            raise ValueError("M/M/1/K requires a positive finite capacity K")

    def is_stable(self) -> bool:
        """Checks if the system is stable - M/M/1/K is always stable due to finite capacity"""
        return self.arrival_rate > 0 and self.service_rate > 0

    def probability_n_req(self, n: int) -> float:
        """Calculate P(n) for n=0,...,K in M/M/1/K"""
        if n < 0 or n > self.max_capacity:
            return 0.0

        _rho = self.arrival_rate / self.service_rate

        if _rho == 1:
            return 1 / (self.max_capacity + 1)
        else:
            _p_0 = (1 - _rho) / (1 - _rho**(self.max_capacity + 1))
            return (_rho**n) * _p_0

    def _calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/1/K"""
        _rho = self.arrival_rate / self.service_rate

        # Calculate P(K) - probability of system being full
        _p_k = self.probability_n_req(self.max_capacity)

        # Effective arrival rate (considering blocked customers)
        _lambda_eff = self.arrival_rate * (1 - _p_k)

        # Server utilization
        self.utilization = _lambda_eff / self.service_rate

        if _rho == 1:
            # Special case when rho = 1
            self.avg_req_in_sys = self.max_capacity / 2
            self.avg_req_in_queue = self.avg_req_in_sys - (1 - _p_k)
        else:
            # Calculate using finite sum formulas
            if _rho != 1:
                term1 = _rho * (1 - (self.max_capacity + 1) * _rho**self.max_capacity + self.max_capacity * _rho**(self.max_capacity + 1))
                term2 = (1 - _rho) * (1 - _rho**(self.max_capacity + 1))
                self.avg_req_in_sys = term1 / term2
                self.avg_req_in_queue = self.avg_req_in_sys - (1 - _p_k)

        # Calculate time metrics
        if _lambda_eff > 0:
            self.avg_time_in_sys = self.avg_req_in_sys / _lambda_eff
            self.avg_time_in_queue = self.avg_req_in_queue / _lambda_eff
        else:
            self.avg_time_in_sys = 0
            self.avg_time_in_queue = 0


@dataclass
class MMsKModel(BaseQueueModel):
    """M/M/s/K queueing system: s servers, finite capacity K"""

    queue_type: ClassVar[QueueType] = QueueType.MMSK

    def _validate_parameters(self) -> None:
        """Validations specific to M/M/s/K"""
        if self.num_servers < 1:
            raise ValueError("M/M/s/K requires at least one server")
        if self.max_capacity is None or self.max_capacity < self.num_servers:
            raise ValueError("M/M/s/K requires finite capacity K >= s (number of servers)")

    def is_stable(self) -> bool:
        """Checks if the system is stable - M/M/s/K is always stable due to finite capacity"""
        return self.arrival_rate > 0 and self.service_rate > 0

    def _calculate_p0(self) -> float:
        """Calculates P(0) for M/M/s/K"""
        _rho = self.arrival_rate / self.service_rate
        _s = self.num_servers
        _k = self.max_capacity

        # Sum for n from 0 to s-1
        _sum1 = sum((_rho**n) / math.factorial(n) for n in range(_s))

        # Term for n from s to K
        if self.arrival_rate == _s * self.service_rate:
            _sum2 = ((_rho**_s) / math.factorial(_s)) * (_k - _s + 1)
        else:
            _server_util = _rho / _s
            _sum2 = ((_rho**_s) / math.factorial(_s)) * \
                ((1 - _server_util**(_k - _s + 1)) / (1 - _server_util))

        return 1 / (_sum1 + _sum2)

    def probability_n_req(self, n: int) -> float:
        """Calculate P(n) for n=0,...,K in M/M/s/K"""
        if n < 0 or n > self.max_capacity:
            return 0.0

        _rho = self.arrival_rate / self.service_rate
        _p_0 = self._calculate_p0()

        if n < self.num_servers:
            return ((_rho**n) / math.factorial(n)) * _p_0
        else:
            return ((_rho**n) / (math.factorial(self.num_servers) * (self.num_servers**(n - self.num_servers)))) * _p_0

    def _calculate_metrics(self) -> None:
        """Calculates analytical metrics for M/M/s/K"""
        # Calculate P(K) - probability of system being full
        _p_k = self.probability_n_req(self.max_capacity)

        # Effective arrival rate
        _lambda_eff = self.arrival_rate * (1 - _p_k)

        # Calculate expected number in system and queue
        self.avg_req_in_sys = sum(n * self.probability_n_req(n) for n in range(self.max_capacity + 1))

        self.avg_req_in_queue = sum(max(0, n - self.num_servers) * self.probability_n_req(n) for n in range(self.max_capacity + 1))

        # Calculate server utilization
        server_busy_prob = sum(min(n, self.num_servers) * self.probability_n_req(n) for n in range(self.max_capacity + 1))
        self.utilization = server_busy_prob / self.num_servers

        # Calculate time metrics
        if _lambda_eff > 0:
            self.avg_time_in_sys = self.avg_req_in_sys / _lambda_eff
            self.avg_time_in_queue = self.avg_req_in_queue / _lambda_eff
        else:
            self.avg_time_in_sys = 0
            self.avg_time_in_queue = 0


class QueueModel:
    """Factory class for creating and working with various queue models.

    This class provides a unified interface to create and work with different
    queueing models (M/M/1, M/M/s, M/M/1/K, M/M/s/K) based on input parameters.

    Usage:
        # Create M/M/1 queue
        queue = QueueModel.create(arrival_rate=5, service_rate=10)

        # Create M/M/s queue with 3 servers
        queue = QueueModel.create(arrival_rate=10, service_rate=4, num_servers=3)

        # Create M/M/1/K queue with capacity 20
        queue = QueueModel.create(arrival_rate=5, service_rate=10, max_capacity=20)

        # Get queue type
        model_type = QueueModel.get_model_type(arrival_rate=5, service_rate=10)
    """

    @staticmethod
    def get_model_type(arrival_rate: float,
                       service_rate: float,
                       num_servers: int = 1,
                       max_capacity: Optional[int] = None) -> QueueType:
        """Determines the appropriate queue model type based on parameters.

        Args:
            arrival_rate: The arrival rate (λ)
            service_rate: The service rate per server (μ)
            num_servers: Number of servers (s or c), defaults to 1
            max_capacity: Maximum system capacity (K), defaults to None (infinite)

        Returns:
            QueueType: The appropriate queue model type
        """
        if max_capacity is None:
            # Infinite capacity cases
            return QueueType.MMS if num_servers > 1 else QueueType.MM1
        else:
            # Finite capacity cases
            return QueueType.MMSK if num_servers > 1 else QueueType.MM1K

    @staticmethod
    def create(arrival_rate: float,
               service_rate: float,
               num_servers: int = 1,
               max_capacity: Optional[int] = None,
               model_type: Optional[QueueType] = None) -> BaseQueueModel:
        """Creates and returns the appropriate queue model.

        Args:
            arrival_rate: The arrival rate (λ)
            service_rate: The service rate per server (μ)
            num_servers: Number of servers (s or c), defaults to 1
            max_capacity: Maximum system capacity (K), defaults to None (infinite)
            model_type: Explicitly specify model type (optional)

        Returns:
            BaseQueueModel: An instance of the appropriate queue model

        Raises:
            ValueError: If parameters are inconsistent with the specified model type
        """
        # If model type not explicitly provided, determine from parameters
        if model_type is None:
            model_type = QueueModel.get_model_type(arrival_rate,
                                                   service_rate,
                                                   num_servers,
                                                   max_capacity)

        # Create the appropriate model instance
        if model_type == QueueType.MM1:
            return MM1Model(arrival_rate,
                            service_rate)
        elif model_type == QueueType.MMS:
            return MMsModel(arrival_rate,
                            service_rate,
                            num_servers)
        elif model_type == QueueType.MM1K:
            return MM1KModel(arrival_rate,
                             service_rate,
                             1,
                             max_capacity)
        elif model_type == QueueType.MMSK:
            return MMsKModel(arrival_rate,
                             service_rate,
                             num_servers,
                             max_capacity)
        else:
            raise ValueError(f"Unsupported queue model type: {model_type}")

    @staticmethod
    def compare_models(arrival_rates: list[float],
                       service_rates: list[float],
                       num_servers_list: list[int] = None,
                       max_capacities: list[int] = None) -> Dict[str, BaseQueueModel]:
        """Creates multiple queue models for comparison.

        Args:
            arrival_rates: List of arrival rates to compare
            service_rates: List of service rates to compare
            num_servers_list: List of server counts to compare (optional)
            max_capacities: List of system capacities to compare (optional)

        Returns:
            Dict[str, BaseQueueModel]: Dictionary of queue models with descriptive keys
        """
        results = {}

        # Use default values if not provided
        if num_servers_list is None:
            num_servers_list = [1]
        if max_capacities is None:
            max_capacities = [None]

        # Create combinations
        for ar in arrival_rates:
            for sr in service_rates:
                for ns in num_servers_list:
                    for mc in max_capacities:
                        # Create a descriptive key
                        key = f"λ={ar},μ={sr}"
                        if ns > 1:
                            key += f",s={ns}"
                        if mc is not None:
                            key += f",K={mc}"

                        # Create the model
                        results[key] = QueueModel.create(ar, sr, ns, mc)

        return results


# Example usage:
if __name__ == "__main__":
    # Create an M/M/1 queue
    queue1 = QueueModel.create(arrival_rate=5,
                               service_rate=10)
    queue1.print_summary()

    # Create an M/M/3 queue
    queue2 = QueueModel.create(arrival_rate=15,
                               service_rate=6,
                               num_servers=3)
    queue2.print_summary()

    # Create an M/M/1/20 queue
    queue3 = QueueModel.create(arrival_rate=5,
                               service_rate=4,
                               max_capacity=20)
    queue3.print_summary()

    # Create an M/M/2/15 queue
    queue4 = QueueModel.create(arrival_rate=10,
                               service_rate=6,
                               num_servers=2,
                               max_capacity=15)
    queue4.print_summary()

    # Compare different M/M/1 queues with varying arrival rates
    comparison = QueueModel.compare_models(
        arrival_rates=[2, 4, 6, 8],
        service_rates=[10],
        num_servers_list=[1],
        max_capacities=[None]
    )

    print("\n=== Comparison of M/M/1 Queues with Different Arrival Rates ===")
    for key, model in comparison.items():
        print(f"\n{key}:")
        print(f"  Utilization: {model.utilization:.4f}")
        print(f"  Avg queue length: {model.avg_req_in_queue:.4f}")
        print(f"  Avg wait time: {model.avg_time_in_queue:.4f}")
