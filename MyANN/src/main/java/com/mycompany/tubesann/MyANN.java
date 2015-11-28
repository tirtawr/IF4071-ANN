/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.tubesann;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

import java.util.HashMap;

/**
 *
 * @author Riady
 */


public class MyANN implements Classifier{
    // 1 PTR, 2 batch, 3 delta
    public static int PERCEPTRON_TRAINING_RULE = 1;
    public static int BATCH_GRADIENT_DESCENT = 2;
    public static int DELTA_RULE = 3;
    private int rule;
    public static double LEARNINGRATE = 1;
    public static double MOMENTUM = 0;
    private Node finalNode;
    private InputNode[] startNode;
    private double squareError = 0;
    private Double deltaMSE = null;
    private Integer maxIteration = null;
    private HashMap<Integer,Double> weight = new HashMap<Integer,Double>();
    private boolean isWeightRandom = false;

    public void perceptronTrainingRule(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(1);
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.updateWeight(desiredOutput[i]);
            error += (desiredOutput[i]-finalNode.getOutput())*(desiredOutput[i]-finalNode.getOutput());
        }
        squareError = 0.5 * error;
    }

    public void batchGradientDescent(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(0);
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.batchGradient(desiredOutput[i]);
            error += (desiredOutput[i]-finalNode.getOutput())*(desiredOutput[i]-finalNode.getOutput());
        }
        squareError = 0.5 * error;
        finalNode.updateWeightBatch();
    }

    public void deltaRule(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(0);
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.updateWeight(desiredOutput[i]);
            error += (desiredOutput[i]-finalNode.getOutput())*(desiredOutput[i]-finalNode.getOutput());
        }
        squareError = 0.5 * error;
    }

    public void backPropagation(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(2);
        //epoch

        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.updateWeightBackPropFinalNode(desiredOutput[i]);
        }
    }

    public void setRule(int rule) {
        this.rule = rule;
    }


    public void setFinalNode(Node finalNode) {
        this.finalNode = finalNode;
    }

    public void setStartNode(InputNode[] startNode) {
        this.startNode = startNode;
    }

    public void setDeltaMSE(Double deltaMSE) {
        this.deltaMSE = deltaMSE;
    }

    public void setMaxIteration(Integer maxIteration) {
        this.maxIteration = maxIteration;
    }

    private void initiate(Instances train) {
        startNode= new InputNode[train.numAttributes()];

        for(int i=0;i<startNode.length;i++){
            startNode[i] = new InputNode(i);
            startNode[i].setActivationFunction(1);
        }

        finalNode = new Node(train.numAttributes());
        finalNode.setActivationFunction(1);
        finalNode.setPrev(startNode);
        if (isWeightRandom) {
            double rangeMin = 0.0;
            double rangeMax = 1.0;
            for (int i=0;i<startNode.length;i++) {
                this.weight.put(i, new Double(Math.random() * (rangeMax - rangeMin) + rangeMin));
            }
        }
        finalNode.setPrevWeight(weight);
        MyANN.LEARNINGRATE = 0.1;
    }

    public void setWeight(double[] weight) {
        for (int i=0;i<weight.length;i++) {
            this.weight.put(i, new Double(weight[i]));
        }
    }

    public void randomWeight() {
        isWeightRandom = true;
    }

    public void buildClassifier(Instances train) throws Exception {
        train.setClassIndex(train.numAttributes()-1);

        initiate(train);

        double[][] testInput = new double[train.numInstances()][train.numAttributes()];
        double[] testDesiredOutput = new double[train.numInstances()];
        for(int i=0; i<train.numInstances(); i++)
        {
            testDesiredOutput[i] = train.instance(i).classValue();
            for(int j = 0; j<train.numAttributes()-1; j++)
            {
                testInput[i][j] = train.instance(i).value(j);
            }
        }
        boolean stop = false;
        int iterator = 1;
        while (!stop) {
            switch (rule) {
                case 1:
                    perceptronTrainingRule(testInput, testDesiredOutput);
                    break;
                case 2:
                    batchGradientDescent(testInput, testDesiredOutput);
                    break;
                case 3:
                    deltaRule(testInput, testDesiredOutput);
                    break;
                default:
                    break;
            }
            if (deltaMSE != null) {
                if (squareError < deltaMSE) {
                    stop = true;
                }
            }
            if (maxIteration != null) {
                if (iterator >= maxIteration) {
                    stop = true;
                }
            }
            iterator++;
        }
    }

    public double classifyInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
