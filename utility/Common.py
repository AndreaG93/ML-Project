def show_shape(input_data, target_data):
    """
    An utility function to show 'shapes' of our CNN inputs.
    """
    print("Expected: (num_samples, timestamps, data)")
    print(" Input Data -> {}".format(input_data.shape))
    print("Target Data -> {}".format(target_data.shape))
