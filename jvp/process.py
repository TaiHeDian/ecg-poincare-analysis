from jvp.filter import baseline_correct


def filter_dict_data(data: dict, fs=1000, cutoff=5, order=2):
    processed_data = data.copy()

    for value in processed_data.values():
        value[:, 1] = baseline_correct(value[:, 1], fs, cutoff, order)

    return processed_data
