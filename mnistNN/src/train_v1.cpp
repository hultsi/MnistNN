#include <array>
#include "mnistNN.h"
#include "statpack.h"

// TODO: Change all std::array<std::array<...>> to one dimensionals...
namespace mnistNN {

    void train_v1() {
        constexpr const int learnRate = 5;

        // HiddenLayer size
        constexpr const int hLayerN1 = 40;
        constexpr const int hLayerN2 = 20;

        std::array<std::array<int, 784>, hLayerN1> weights1; //28^2
        std::array<std::array<int, hLayerN1>, hLayerN2> weights2;
        std::array<std::array<int, hLayerN2>, 10> weights3;

        // Randomize initial weights
        for (int k = 0; k < hLayerN1; k++) {
            for (int i = 0; i < 784; i++) {
                weights1[k][i] = statpack::randomFloat(0,1) - .5f;
            }
            for (int i = 0; i < hLayerN2; i++) {
                weights2[i][k] = statpack::randomFloat(0,1) - .5f;
            }
        }
        for (int i = 0; i < hLayerN2; i++) {
            for (int k = 0; k < 10; k++) {
                weights3[k][i] = statpack::randomFloat(0,1) - .5f;
            }
        }

        std::array<float, 10> result;
        std::array<int, 10> targetResult;
        std::array<float, hLayerN1> wSum1;
        std::array<float, hLayerN2> wSum2;
        std::array<float, 10> wSum3;
        std::array<float, hLayerN1> hiddenNeuron1;
        std::array<float, hLayerN2> hiddenNeuron2;
        std::array<float, hLayerN1> deltaHiddenNeuron1;
        std::array<float, hLayerN2> deltaHiddenNeuron2;
        std::array<float, hLayerN1> bias1;
        std::array<float, hLayerN1> deltaBias1;
        std::array<float, hLayerN2> bias2;
        std::array<float, hLayerN2> deltaBias2;
        std::array<float, 10> bias3;
        std::array<float, 10> deltaBias3;
        std::array<std::array<float, 784>, hLayerN1> deltaWeights1;
        std::array<std::array<float, hLayerN1>, hLayerN2> deltaWeights2;
        std::array<std::array<float, hLayerN2>, 10> deltaWeights3;
        float costFunction = 0;
        float costFunctionAv = 0;
        float costFunctionLimit = 0.25;
        float guessProb = 0;
        float guessProbAv = 0;
        int iterations = 1;
        float backPropTerm = 0;
        int loopCounter = 0;
        int targetNumber = 0;

        // Randomize biases
        for (int i = 0; i < hLayerN1; i++) {
            bias1[i] = statpack::randomFloat(0,1) - .5f;
        }
        for (int i = 0; i < hLayerN2; i++) {
            bias2[i] = statpack::randomFloat(0,1) - .5f;
        }
        for (int i = 0; i < 10; i++) {
            bias3[i] = statpack::randomFloat(0,1) - .5f;
        }

        // Get starting time
        // DateFormat df = new SimpleDateFormat("dd/MM/yy HH:mm:ss");
        // Date dateobj = new Date();

        int epochLength = 0;
        // Neural network loop
        while (true) {
            guessProb = 0;
            loopCounter = 0;
            costFunctionAv = 0;

            while (loopCounter < epochLength) {
                costFunction = 0;

                // FORWARD PROPAGATION
                for (int i = 0; i < targetResult.size(); i++) {
                    targetResult[i] = 0;
                }
                targetNumber = loopCounter % 10;
                int imageNumber = statpack::randomInt(1, 5421); // TODO: Images are in bit format atm!!
                targetResult[targetNumber] = 1;

        //         double[] inputs = calculateBlackWhiteValues(targetNumber, imageNumber); // Get color shades from random image
        //         inputs = normalizeInputs(inputs); //Normalize inputs to a range of 0 to 1

        //         // Calculate wsum for each hLayerN1 neurons
        //         for (int i = 0; i < hLayerN1; i++) {
        //             wSum1[i] = weightedSum(inputs, weights1[i])+bias1[i];
        //             hiddenNeuron1[i] = sigmoid(wSum1[i]);
        //         }

        //         // Calculate wsum for each hLayerN2 neurons
        //         for (int i = 0; i < hLayerN2; i++) {
        //             wSum2[i] = weightedSum(hiddenNeuron1, weights2[i])+bias2[i];
        //             hiddenNeuron2[i] = sigmoid(wSum2[i]);
        //         }

        //         // Calculate wsum from the hidden layer to get the final result and then calculate error
        //         for (int i = 0; i < 10; i++) {
        //             wSum3[i] = weightedSum(hiddenNeuron2, weights3[i])+bias3[i];
        //             result[i] = sigmoid(wSum3[i]);

        //             costFunction += Math.pow(targetResult[i] - result[i], 2);// / (double)epochLength;
        //         }
        //         costFunctionAv += costFunction/(double)epochLength;

        //         // START BACKPROPAGATION
        //         for (int i = 0; i < 10; i++) {
        //             backPropTerm = -sigmoidDerivative(wSum3[i])*2*(targetResult[i]-result[i]);

        //             for (int k = 0; k < hLayerN2; k++) {
        //                 deltaWeights3[i][k] += hiddenNeuron2[k]*backPropTerm;///(double)epochLength;
        //                 deltaHiddenNeuron2[k] += weights2[i][k]*backPropTerm;
        //             }
        //             deltaBias3[i] += backPropTerm;///(double)epochLength;
        //         }

        //         for (int i = 0; i < hLayerN2; i++) {
        //             backPropTerm = sigmoidDerivative(wSum2[i])*deltaHiddenNeuron2[i];
        //             for (int k = 0; k < hLayerN1; k++) {
        //                 deltaWeights2[i][k] += hiddenNeuron1[k]*backPropTerm;///(double)epochLength;
        //                 deltaHiddenNeuron1[k] += weights1[i][k]*backPropTerm;
        //             }
        //             deltaBias2[i] += backPropTerm;///(double)epochLength;
        //             deltaHiddenNeuron2[i]=0;
        //         }

        //         for (int i = 0; i < hLayerN1; i++) {
        //             backPropTerm = sigmoidDerivative(wSum1[i])*deltaHiddenNeuron1[i];
        //             for (int k = 0; k < 784; k++) {
        //                 deltaWeights1[i][k] += inputs[k]*backPropTerm;///(double)epochLength;
        //             }
        //             deltaBias1[i] += backPropTerm;///(double)epochLength;
        //             deltaHiddenNeuron1[i]=0;
        //         }

        //         if (maxValInd(result) == targetNumber) {
        //             guessProb += 1;
        //         }

                loopCounter++;
            }

        //     iterations++;

        //     if (iterations % 50 == 0) {
        //         System.out.println(Double.toString(guessProb) + " / " + Integer.toString(epochLength) + " --- " + Double.toString(costFunctionAv) + " " + Integer.toString(iterations));
        //     }
        //     if (iterations % 5000 == 0) {
        //         if (iterations == 35000) {
        //             learnRate = 2;
        //         } else if (iterations == 75000) {
        //             learnRate = .3;
        //         } else if (iterations == 110000) {
        //             learnRate = .05;
        //         }
        //         System.out.println("Learn rate changed to: " + Double.toString(learnRate));
        //         List<String> saveWeights1 = new ArrayList<>();//Arrays.asList("The first line", "The second line");
        //         List<String> saveWeights2 = new ArrayList<>();
        //         List<String> saveWeights3 = new ArrayList<>();
        //         List<String> saveBiases1 = new ArrayList<>();
        //         List<String> saveBiases2 = new ArrayList<>();
        //         List<String> saveBiases3 = new ArrayList<>();
        //         for (int i = 0; i < hLayerN1; i++) {
        //             for (int k = 0; k < 784; k++) {
        //                 saveWeights1.add(Double.toString(weights1[i][k]));
        //             }
        //         }
        //         for (int i = 0; i < hLayerN2; i++) {
        //             for (int k = 0; k < hLayerN1; k++) {
        //                 saveWeights2.add(Double.toString(weights2[i][k]));
        //             }
        //         }
        //         for (int i = 0; i < 10; i++) {
        //             for (int k = 0; k < hLayerN2; k++) {
        //                 saveWeights3.add(Double.toString(weights3[i][k]));
        //             }
        //         }
        //         for (int i = 0; i < hLayerN1; i++) {
        //             saveBiases1.add(Double.toString(bias1[i]));
        //         }
        //         for (int i = 0; i < hLayerN2; i++) {
        //             saveBiases2.add(Double.toString(bias2[i]));
        //         }
        //         for (int i = 0; i < 10; i++) {
        //             saveBiases3.add(Double.toString(bias3[i]));
        //         }
        //         try {
        //             Path file = Paths.get("weights1_2.txt");
        //             Files.write(file, saveWeights1, Charset.forName("UTF-8"));
        //             file = Paths.get("weights2_2.txt");
        //             Files.write(file, saveWeights2, Charset.forName("UTF-8"));
        //             file = Paths.get("weights3_2.txt");
        //             Files.write(file, saveWeights3, Charset.forName("UTF-8"));
        //             file = Paths.get("biases1_2.txt");
        //             Files.write(file, saveBiases1, Charset.forName("UTF-8"));
        //             file = Paths.get("biases2_2.txt");
        //             Files.write(file, saveBiases2, Charset.forName("UTF-8"));
        //             file = Paths.get("biases3_2.txt");
        //             Files.write(file, saveBiases3, Charset.forName("UTF-8"));
        //             //Files.write(file, saveWeights2, Charset.forName("UTF-8"), StandardOpenOption.APPEND);
        //         } catch (IOException e) {
        //             System.out.println("WARNING: EXCEPTION OCCURRED!");
        //         }

        //         NetworkTest nwTest = new NetworkTest();
        //         int correctGuesses = nwTest.correctGuesses(250);
        //         System.out.println("Correct guesses: " + Integer.toString(correctGuesses) + " / 2500");

        //         if (correctGuesses > 250*9) {
        //             System.out.println("Final results: \n" + Double.toString(guessProb) + " " +
        //                     Double.toString(costFunctionAv) + " " + Integer.toString(iterations));
        //             System.out.println(Integer.toString(targetNumber) + " " + Double.toString(result[0]) + " " + Double.toString(result[1]) + " " +
        //                     Double.toString(result[2]) + " " + Double.toString(result[3]) + " " +
        //                     Double.toString(result[4]) + " " + Double.toString(result[5]) + " " +
        //                     Double.toString(result[6]) + " " + Double.toString(result[7]) + " " +
        //                     Double.toString(result[8]) + " " + Double.toString(result[9]));
        //             System.out.println("Start time: " + df.format(dateobj));
        //             dateobj = new Date();
        //             System.out.println("End time: " + df.format(dateobj));
        //             return;
        //         }
        //         costFunctionLimit = costFunctionLimit*0.9;
        //     }

        //     for (int i = 0; i < 10; i++) {
        //         for (int k = 0; k < hLayerN2; k++) {
        //             weights3[i][k] += -(deltaWeights3[i][k])*learnRate/(double)epochLength;
        //             deltaWeights3[i][k] = 0;
        //         }
        //         bias3[i] += -deltaBias3[i]*learnRate/(double)epochLength;
        //         deltaBias3[i] = 0;
        //     }

        //     for (int i = 0; i < hLayerN2; i++) {
        //         for (int k = 0; k < hLayerN1; k++) {
        //             weights2[i][k] += -(deltaWeights2[i][k])*learnRate/(double)epochLength;
        //             deltaWeights2[i][k] = 0;
        //         }
        //         bias2[i] += -deltaBias2[i]*learnRate/(double)epochLength;
        //         deltaBias2[i] = 0;
        //     }

        //     for (int i = 0; i < hLayerN2; i++) {
        //         for (int k = 0; k < 784; k++) {
        //             weights1[i][k] += -(deltaWeights1[i][k])*learnRate/(double)epochLength;
        //             deltaWeights1[i][k] = 0;
        //         }
        //         bias1[i] += -deltaBias1[i]*learnRate/(double)epochLength;
        //         deltaBias1[i] = 0;
        //     }
        }
    }
}