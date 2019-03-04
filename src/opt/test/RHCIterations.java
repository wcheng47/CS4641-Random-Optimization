package opt.test;

import func.nn.feedfwd.FeedForwardNetwork;
import func.nn.feedfwd.FeedForwardNeuralNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.RandomOrderFilter;
import shared.filt.TestTrainSplitFilter;

import java.io.*;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying abalone as having either fewer
 * or more than 15 rings.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class RHCIterations {
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
    private static List<Double> oaResultsTrain = new ArrayList<>();
    private static List<Double> oaResultsTest = new ArrayList<>();
    private static DecimalFormat df = new DecimalFormat("0.000");

    public static void main(String[] args) {
        for (int k = 0; k < 1; k++) {
            new RandomOrderFilter().filter(set);
            TestTrainSplitFilter ttsf = new TestTrainSplitFilter(70);
            ttsf.filter(set);
            DataSet train = ttsf.getTrainingSet();
            DataSet test = ttsf.getTestingSet();

            for(int i = 0; i < oa.length; i++) {
                networks[i] = factory.createClassificationNetwork(
                        new int[] {inputLayer, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, outputLayer});
                nnop[i] = new NeuralNetworkOptimizationProblem(train, networks[i], measure);
            }

            oa[0] = new RandomizedHillClimbing(nnop[0]);
            //oa[1] = new SimulatedAnnealing(1E12, .65, nnop[1]);
            //oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

            for (int i = 0; i < 1; i++) {
                double start = System.nanoTime(), end, trainingTime, testingTime, correct = 0, incorrect = 0;
                train(oa[i], networks[i], oaNames[i], train, test); //trainer.train();
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

                results += "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " instances." +
                        "\nIncorrectly classified " + incorrect + " instances.\nPercent correctly classified: "
                        + df.format(correct / (correct + incorrect) * 100) + "%\nTraining time: " + df.format(trainingTime)
                        + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
                System.out.println(results);
            }
        }

        try {
            FileWriter fw = new FileWriter(new File("src/opt/test/rhc_iterations.csv"));
            fw.write("Iterations,Training Error,Testing Error\n");
            for (int i = 0; i < trainingIterations; i++) {
                fw.write((i+1) + "," + oaResultsTrain.get(i) + "," + oaResultsTest.get(i) + "\n");
            }
            fw.close();
        }
        catch(Exception e) {
            e.printStackTrace();
        }
    }

    private static void train(OptimizationAlgorithm oa, FeedForwardNetwork network, String oaName, DataSet train, DataSet test) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");
        Instance[] trainInstances = train.getInstances();
        Instance[] testInstances = test.getInstances();

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
            oaResultsTrain.add(trainError);
            oaResultsTest.add(testError);
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
