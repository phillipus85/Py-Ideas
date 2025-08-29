from pydasa import Queue


def print_metrics(results: dict) -> None:
    for key, value in results.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    # Example usage of pydasa Queue class
    # M/M/1 Queue
    print("\nM/M/1 Queue")
    mm1 = Queue(_lambda=2, miu=2.5)
    mm1.calculate_metrics()
    print_metrics(mm1.get_metrics())

    # M/M/s Queue
    print("\nM/M/s Queue")
    mms = Queue(_lambda=2.0, miu=1.5, n_servers=3)
    mms.calculate_metrics()
    print_metrics(mms.get_metrics())
    print(f"Probability of 0 jobs in system P(0): {mms.calculate_prob_zero():.4f}")
    print(f"Tau: {mms._lambda / mms.miu:.4f}")

    # M/M/1/K Queue
    print("\nM/M/1/K Queue")
    mm1k = Queue(_lambda=25.0, miu=40.0, n_servers=1, kapacity=12)
    mm1k.calculate_metrics()
    print_metrics(mm1k.get_metrics())
    print(mm1k)

    # M/M/s/K Queue
    print("\nM/M/s/K Queue")
    mmsk = Queue(_lambda=45.0, miu=15.0, n_servers=3, kapacity=12)
    mmsk.calculate_metrics()
    print_metrics(mmsk.get_metrics())
    print(mmsk)
    print(
        f"Probability of 0 jobs in system P(0): {mmsk.calculate_prob_zero():.8f}")
    print(
        f"Probability of 7 jobs in system P(7): {mmsk.calculate_prob_n(7):.8f}")
    print(f"Tau: {mmsk._lambda / mmsk.miu:.8f}")
