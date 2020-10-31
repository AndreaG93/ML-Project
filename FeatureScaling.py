def _get_standardized_data(values):
    mean = values.mean()
    standard_deviation = values.std()

    return (values - mean) / standard_deviation


def _get_normalized_data(values):
    min_value = values.min()
    max_value = values.max()

    return (values - min_value) / (max_value - min_value)


def perform_feature_scaling(values, feature_scaling=''):
    output = values

    if feature_scaling.__eq__('normalization'):
        output = _get_normalized_data(values)

    if feature_scaling.__eq__('standardization'):
        output = _get_standardized_data(values)

    return output
