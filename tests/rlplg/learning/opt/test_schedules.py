from rlplg.learning.opt import schedules


def test_learning_rate_schedule_call():
    def schedule(initial_learning_rate: float, step: int):
        return initial_learning_rate * (0.9**step)

    output_1 = schedules.LearningRateSchedule(
        initial_learning_rate=1.0, schedule=schedule
    )(step=1)
    output_2 = schedules.LearningRateSchedule(
        initial_learning_rate=1.0, schedule=schedule
    )(step=2)
    output_3 = schedules.LearningRateSchedule(
        initial_learning_rate=100.0, schedule=schedule
    )(step=2)

    assert output_1 == 0.9
    assert output_2 == 0.81  # epoch doesn't affect output
    assert output_3 == 81


def test_learning_rate_schedule_call_using_epoch_based_scheduler():
    def schedule(initial_learning_rate: float, step: int):
        if step < 10:
            return initial_learning_rate
        return initial_learning_rate * 0.9

    lrs = schedules.LearningRateSchedule(initial_learning_rate=1.0, schedule=schedule)

    output_1 = lrs(step=1)
    output_2 = lrs(step=9)
    output_3 = lrs(step=10)
    output_4 = lrs(step=21)

    assert output_1 == 1
    assert output_2 == 1
    assert output_3 == 0.9
    assert output_4 == 0.9
