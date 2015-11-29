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
import weka.filters.Filter;
import weka.filters.supervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

import java.util.*;

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
    private Node[] finalNode;
    private InputNode[] startNode;
    private double squareError = 0;
    private Double deltaMSE = null;
    private Integer maxIteration = null;
    private HashMap<Integer,Double> weight = new HashMap<Integer,Double>();
    private boolean isWeightRandom = false;
    private final Normalize normalizeFilter = new Normalize();
    private final NominalToBinary nominalToBinaryFilter = new NominalToBinary();

    public void perceptronTrainingRule(double[][] input,double[][] desiredOutput){
        for(int i=0;i<finalNode.length;i++){
            finalNode[i].setActivationFunction(1);
        }
        
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            for(int j=0;j<finalNode.length;j++){
                finalNode[j].updateWeight(desiredOutput[i][j]);
                System.out.println("output= "+j+" "+finalNode[j].getOutput());
                
                error += (desiredOutput[i][j]-finalNode[j].getOutput())*(desiredOutput[i][j]-finalNode[j].getOutput());
            }
        }
        squareError = 0.5 * error;
    }

    public void batchGradientDescent(double[][] input,double[][] desiredOutput){
        
        for(int i=0;i<finalNode.length;i++){
            finalNode[i].setActivationFunction(0);
        }
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            for(int j=0;j<finalNode.length;j++){
                finalNode[j].batchGradient(desiredOutput[i][j]);
                error += (desiredOutput[i][j]-finalNode[j].getOutput())*(desiredOutput[i][j]-finalNode[j].getOutput());
            }
        }
        squareError = 0.5 * error;
        for(int i=0;i<finalNode.length;i++){
            finalNode[i].updateWeightBatch();
        }
    }

    public void deltaRule(double[][] input,double[][] desiredOutput){
        for(int i=0;i<finalNode.length;i++){
            finalNode[i].setActivationFunction(0);
        }
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            for(int j=0;j<finalNode.length;j++){
                finalNode[j].updateWeight(desiredOutput[i][j]);
                error += (desiredOutput[i][j]-finalNode[j].getOutput())*(desiredOutput[i][j]-finalNode[j].getOutput());
            }
        }
        squareError = 0.5 * error;
    }

    public void backPropagation(double[][] input,double[][] desiredOutput){
        for(int i=0;i<finalNode.length;i++){
            finalNode[i].setActivationFunction(2);
        }
        //epoch
        double error = 0;
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            for(int j=0;j<finalNode.length;j++){
                finalNode[j].updateWeightBackPropFinalNode(desiredOutput[i][j]);
                error += (desiredOutput[i][j]-finalNode[j].getOutput())*(desiredOutput[i][j]-finalNode[j].getOutput());
            }
            
        }
        squareError = 0.5 * error;
    }

    public void setRule(int rule) {
        this.rule = rule;
    }


    public void setFinalNode(Node[] finalNode) {
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

    public static Instances setNominalToBinary(Instances instances) {
        NominalToBinary ntb = new NominalToBinary();
        Instances newInstances = null;
        try {
            ntb.setInputFormat(instances);
            newInstances = new Instances(Filter.useFilter(instances, ntb));
        } catch (Exception e) {
            e.printStackTrace();
        }

        return newInstances;
    }

    private void initiate(Instances train) throws Exception {
        startNode= new InputNode[train.numAttributes()];

        for(int i=0;i<startNode.length;i++){
            startNode[i] = new InputNode(i);
            startNode[i].setActivationFunction(1);
        }

        if (isWeightRandom) {
            double rangeMin = 0.0;
            double rangeMax = 1.0;
            for (int i=0;i<startNode.length;i++) {
                this.weight.put(i, new Double(Math.random() * (rangeMax - rangeMin) + rangeMin));
            }
        }

        finalNode = new Node[train.numClasses()];
        for (int i=0;i<train.numClasses();i++) {
            finalNode[i] = new Node(i+startNode.length);
            finalNode[i].setActivationFunction(1);
            finalNode[i].setPrev(startNode);
            HashMap<Integer,Double> tempWeight = new HashMap<Integer,Double>();
            tempWeight = (HashMap<Integer,Double>)weight.clone();
            finalNode[i].setPrevWeight(tempWeight);
        }
    }
    
    public void setHiddenLayer(int nHiddenLayer,int nNode){
        Node[][] hiddenLayer = new Node[nHiddenLayer][nNode];
        int currentId = startNode.length;
        for(int i=0;i<nHiddenLayer;i++){
            for(int j=0;j<nHiddenLayer;j++){
                hiddenLayer[i][j] = new Node(currentId);
                if(i==0){
                    hiddenLayer[i][j].setPrev(startNode);
                    hiddenLayer[i][j].setPrevWeight((HashMap<Integer,Double>)weight.clone());
                }
                else{
                    hiddenLayer[i][j].setPrev(hiddenLayer[i-1]);
                    hiddenLayer[i][j].setPrevWeight((HashMap<Integer,Double>)weight.clone());
                }
                hiddenLayer[i][j].setActivationFunction(2);
                currentId++;
            }
        }
        for(int i=0;i<finalNode.length;i++){
           finalNode[i].setPrev(hiddenLayer[hiddenLayer.length-1]);
        }
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

        initiate(train);

        double[][] testInput = new double[train.numInstances()][train.numAttributes()];
        double[][] testDesiredOutput = new double[train.numInstances()][train.numClasses()];
        for(int i=0; i<train.numInstances(); i++)
        {
            for (int j=0;j<train.numClasses();j++) {
                if (j == (int) train.instance(i).classValue()) {
                    testDesiredOutput[i][j] = 1;
                } else if (rule == 1) {
                    testDesiredOutput[i][j] = -1;
                } else {
                    testDesiredOutput[i][j] = 0;
                }
                System.out.println("Desired "+i+j+" "+testDesiredOutput[i][j]);
            }
            //testInput[i][0] = 0;
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

    public double classifyInstance(Instance instance) throws Exception {
        double result = 0;
        for (int i=0;i<instance.numAttributes()-1;i++) {
            startNode[i].setInput(instance.value(i));
        }
        List<Double> output = new ArrayList<Double>();
        for (int i=0;i<finalNode.length;i++) {
            output.add(finalNode[i].calculate());
            System.out.println("Output "+i+" "+output.get(i));
        }
        if (rule == 1) {
            boolean found = false;
            int i = 0;
            while (!found && i < output.size()) {
                if (output.get(i) == 1) {
                    result = (double) i;
                    found = true;
                }
                i++;
            }
        } else {
            double max = Collections.max(output);
            result = (double) output.indexOf(max);
        }
        return result;
    }

    public double[] distributionForInstance(Instance instnc) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

}
