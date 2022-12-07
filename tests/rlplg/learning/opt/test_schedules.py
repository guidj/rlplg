import math

from rlplg.learning.opt import schedules


def test_epoch_schedule_call():
    def schedule(step: int):
        assert step >= 0
        return math.floor(step // 100)

    epoch_scheduler = schedules.EpochSchedule(schedule=schedule)

    output_1 = epoch_scheduler(step=1)
    output_2 = epoch_scheduler(step=2)
    output_3 = epoch_scheduler(step=100)
    output_4 = epoch_scheduler(step=202)
    output_5 = epoch_scheduler(step=304)

    assert output_1 == 0
    assert output_2 == 0
    assert output_3 == 1
    assert output_4 == 2
    assert output_5 == 3


def test_learning_rate_schedule_call():
    lrs = schedules.LearningRateSchedule(
        schedule=lambda _, learning_rate: learning_rate * 0.9
    )

    output_1 = lrs(step=1, learning_rate=1)
    output_2 = lrs(step=2, learning_rate=1)
    output_3 = lrs(step=2, learning_rate=100)

    assert output_1 == 0.9
    assert output_2 == 0.9  # epoch doesn't affect output
    assert output_3 == 90


def test_learning_rate_schedule_call_using_epoch_based_scheduler():
    def schedule(step: int, learning_rate: float):
        if step < 10:
            return learning_rate
        return learning_rate * 0.9

    lrs = schedules.LearningRateSchedule(schedule=schedule)

    output_1 = lrs(step=1, learning_rate=1)
    output_2 = lrs(step=9, learning_rate=1)
    output_3 = lrs(step=10, learning_rate=1)
    output_4 = lrs(step=21, learning_rate=1)

    assert output_1 == 1
    assert output_2 == 1
    assert output_3 == 0.9
    assert output_4 == 0.9
