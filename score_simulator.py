def select_worker(observation):
    return "1"

def fps_column_name():
    return "fps"

def main(observations):
    SWITCH_TIME = 1000
    FPS_TRESHOLD = 10
    OBSERVATION_INTERTIME = 1000

    last_worker = "1"

    underpermorming_time = 0
    worker2_time = 0
    total_switch_time = 0

    for obs in observations:
        worker = select_worker(obs)
        if worker != last_worker:
            total_switch_time += SWITCH_TIME
            last_worker = worker
            continue

        fps = obs[fps_column_name()]
        if fps < FPS_TRESHOLD and worker == "1":
            underpermorming_time += OBSERVATION_INTERTIME

        if worker == "2":
            worker2_time += OBSERVATION_INTERTIME

    print(f"{underpermorming_time=}")
    print(f"{worker2_time=}")


if __name__ == '__main__':
    main()