package opt.test;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.text.DecimalFormat;
import java.util.*;

public class SA_Temp {
    private static Instance[] instances = initializeInstances();

    private static int inputLayer = 5, outputLayer = 1, trainingIterations = 300;
    private static FeedForwardNeuralNetworkFactory factory = new FeedForwardNeuralNetworkFactory();
    private static ErrorMeasure measure = new SumOfSquaresError();
    private static DataSet set = new DataSet(instances);
    private static FeedForwardNetwork networks[] = new FeedForwardNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];
    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";
    private static Map<Double, List<Double>> oaResultsTrain = new HashMap<>();
    private static Map<Double, List<Double>> oaResultsTest = new HashMap<>();
    private static DecimalFormat df = new DecimalFormat("0.000");
    private static Double[] temperatures = {1E13};

    public static void main(String[] args) {
        System.out.println("SA_Temp");
        new RandomOrderFilter().filter(set);
        TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
        ttsf.filter(set);
        DataSet train = ttsf.getTrainingSet();
        DataSet test = ttsf.getTestingSet();

        for (int k = 0; k < temperatures.length; k++) {
            oaResultsTrain.put(temperatures[k], new ArrayList<>());
            oaResultsTest.put(temperatures[k], new ArrayList<>());
        }

        for (int k = 0; k < temperatures.length; k++) {
            double temperature = temperatures[k];
            System.out.println("\nTemperature" + temperature + "\n");

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], measure);
            }

            //oa[0] = new RandomizedHillClimbing(nnop[0]);
            oa[1] = new SimulatedAnnealing(temperature, 0.65, nnop[1]);
            //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

            for (int i = 1; i < 2; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i], train, test, temperature); //trainer.train();
                end = System.nanoTime();
                trainingTime = end - start;
                trainingTime /= Math.pow(10, 9);

                Instance optimalInstance = oa[i].getOptimal();
                networks[i].setWeights(optimalInstance.getData());

                double predicted, actual;
                start = System.nanoTime();
                for (int j = 0; j < instances.length; j++) {
                    networks[i].setInputValues(instances[j].getData());
                    networks[i].run();

                    predicted = Double.parseDouble(instances[j].getLabel().toString());
                    actual = Double.parseDouble(networks[i].getOutputValues().toString());

                    double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
                }
                end = System.nanoTime();
                testingTime = end - start;
                testingTime /= Math.pow(10, 9);

                results += "\nResults for " + oaNames[i] + " Temperature " + temperature + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                System.out.println(results);
            }
        }

        try {
            FileWriter fw = new FileWriter(new File("src/opt/test/sa_temp_train_1E13_2.csv"));
            fw.write("Iterations,Temp=1E9 Training,Temp=1E10 Training,Temp=1E11 Training,Temp=1E12 Training,Temp=1E13 Training\n");
            for (int i = 0; i < trainingIterations; i++) {
                fw.write((i+1) + ",");
                for (int j = 0; j < temperatures.length; j++) {
                    if (j == temperatures.length - 1) {
                        fw.write(oaResultsTrain.get(temperatures[j]).get(i) + "\n");
                    } else {
                        fw.write(oaResultsTrain.get(temperatures[j]).get(i) + ",");
                    }

                }
            }
            fw.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        try {
            FileWriter fw2 = new FileWriter(new File("src/opt/test/sa_temp_test_1E13_2.csv"));
            fw2.write("Iterations,Temp=1E9 Testing,Temp=1E10 Testing,Temp=1E11 Testing,Temp=1E12 Testing,Temp=1E13 Testing\n");
            for (int i = 0; i < trainingIterations; i++) {
                fw2.write((i+1) + ",");
                for (int j = 0; j < temperatures.length; j++) {
                    if (j == temperatures.length - 1) {
                        fw2.write(oaResultsTest.get(temperatures[j]).get(i) + "\n");
                    } else {
                        fw2.write(oaResultsTest.get(temperatures[j]).get(i) + ",");
                    }

                }
            }
            fw2.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName, DataSet train, DataSet test, double temperature) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        Instance[] trainInstances = train.getInstances();
        Instance[] testInstances = test.getInstances();

        boolean first = true;

        for(int i = 0; i < trainingIterations; i++) {
            oa.train();

            int trainCorrect = 0, trainIncorrect=0;
            double trainPred, trainActual;
            for(int j = 0; j < trainInstances.length; j++) {
                network.setInputValues(trainInstances[j].getData());
                network.run();

                trainPred = Double.parseDouble(instances[j].getLabel().toString());
                trainActual = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(trainPred - trainActual) < 0.5 ? trainCorrect++ : trainIncorrect++;
            }
            double trainError = (double)trainIncorrect / (trainCorrect + trainIncorrect) * 100;

            int testCorrect = 0, testIncorrect = 0;
            double testPred, testAct;
            for (int j = 0; j < testInstances.length; j++) {
                network.setInputValues(testInstances[j].getData());
                network.run();

                testPred = Double.parseDouble(instances[j].getLabel().toString());
                testAct = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(testPred - testAct) < 0.5 ? testCorrect++ : testIncorrect++;
            }

            double testError = (double)testIncorrect / (testCorrect + testIncorrect) * 100;

            System.out.println("Iteration " + String.format("%04d" ,i) + ": " + df.format(trainError) + " " + df.format(testError));
            oaResultsTrain.get(temperature).add(trainError);
            oaResultsTest.get(temperature).add(testError);
        }
    }

    private static Instance[] initializeInstances() {

        double[][][] attributes = new double[16280][][];

        try {
            BufferedReader br = new BufferedReader(new FileReader(new File("src/opt/test/census_income_normalized.csv")));

            for(int i = 0; i < attributes.length; i++) {
                Scanner scan = new Scanner(br.readLine());
                scan.useDelimiter(",");

                attributes[i] = new double[2][];
                attributes[i][0] = new double[5]; // 4 attributes
                attributes[i][1] = new double[1];

                for(int j = 0; j < 4; j++)
                    attributes[i][0][j] = Double.parseDouble(scan.next());

                attributes[i][1][0] = Double.parseDouble(scan.next());
            }
            br.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }

        Instance[] instances = new Instance[attributes.length];

        for(int i = 0; i < instances.length; i++) {
            instances[i] = new Instance(attributes[i][0]);
            instances[i].setLabel(new Instance(attributes[i][1][0]));
        }

        System.out.println(instances.length);
        return instances;
    }
}
