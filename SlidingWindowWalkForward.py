class SlidingWindowWalkForward(object):

    def __init__(self, dataset_size):
        """
        Implementation of 'Sliding Windows' version of 'Walk Forward' validation technique.
        In this way, our model will be trained only on the most recent observations.
        """

        # Parameter used to represent the number of observations to ignore.
        self._ignore_observations_up_to = 0

        # Testing set's number of observations is the same as stated by in project's specifications: 16 days of data!
        self._testing_set_size = 60 * 60 * 24 * 16

        # Training set's number of observations is 2.39 higher than that of testing set.
        self._training_set_size = (dataset_size - 2 * self._testing_set_size)

        # Sliding windows step
        self._sliding_window_step = self._testing_set_size

        # Iteration index
        self._iteration_index = 0

    def get_next_iteration_indexes(self, debug=False):
        training_set_first_observation = self._ignore_observations_up_to
        training_set_last_observation = training_set_first_observation + self._training_set_size

        test_set_first_observation = training_set_last_observation
        test_set_last_observation = training_set_last_observation + self._testing_set_size

        if debug:
            print('Current iteration   : {}'.format(self._iteration_index))
            print('Training set        : [{}:{}]'.format(training_set_first_observation, training_set_last_observation))
            print('Testing set         : [{}:{}]'.format(test_set_first_observation, test_set_last_observation))
            print('Ignored observations: {}\n'.format(self._ignore_observations_up_to))

        self._ignore_observations_up_to += self._sliding_window_step
        self._iteration_index += 1

        return training_set_first_observation, training_set_last_observation, test_set_first_observation, test_set_last_observation


"""

class SlidingWindowWalkForward(object):

    private final Instances[] parts;
    private final int numOfAllInstance;

    public WalkForward(Instances allInstancesOfDataset) {

        this.numOfAllInstance = allInstancesOfDataset.numInstances();

        int datasetSubSetsAmount = allInstancesOfDataset.numDistinctValues(0);

        this.parts = new Instances[datasetSubSetsAmount];

        int[] datasetSubSetsCardinality = new int[datasetSubSetsAmount];

        for (int index = 0; index < allInstancesOfDataset.numInstances(); index++) {

            Instance instance = allInstancesOfDataset.instance(index);
            int releaseIndex = Integer.parseInt(instance.toString(0));

            datasetSubSetsCardinality[releaseIndex]++;
        }

        int firstIndex = 0;
        for (int index = 0; index < datasetSubSetsAmount; index++) {

            this.parts[index] = new Instances(allInstancesOfDataset, firstIndex, datasetSubSetsCardinality[index]);

            firstIndex += datasetSubSetsCardinality[index];
        }
    }

    private Instances getDataSetSubsetForTraining(int runIndex) {

        Instances output = new Instances(parts[0]);

        for (int index = 1; index <= runIndex; index++) {

            Enumeration<Instance> enumeration = parts[index].enumerateInstances();
            while (enumeration.hasMoreElements())
                output.add(enumeration.nextElement());

        }

        return output;
    }

    private Instances getDataSetSubsetForTesting(int runIndex) {
        return new Instances(parts[runIndex + 1]);
    }


    public List<WalkForwardRunInput> getWalkForwardRunInputs() {

        List<WalkForwardRunInput> output = new ArrayList<>();

        for (int runIndex = 0; runIndex < (parts.length - 1); runIndex++) {

            Instances currentTrainingSet = getDataSetSubsetForTraining(runIndex);
            Instances currentTestingSet = getDataSetSubsetForTesting(runIndex);

            output.add(new WalkForwardRunInput(currentTrainingSet, currentTestingSet, runIndex + 1, this.numOfAllInstance));
        }

        return output;
    }


"""
