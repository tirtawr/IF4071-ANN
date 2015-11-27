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

/**
 *
 * @author Riady
 */


public class MyANN implements Classifier{
    // 1 PTR, 2 batch, 3 delta
    private int rule;
    public static double LEARNINGRATE;
    private Node finalNode;
    private InputNode[] startNode;

    public void perceptronTrainingRule(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(1);
        //epoch
        
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.updateWeight(desiredOutput[i]);
        }
    }
    
    public void batchGradientDescent(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(0);
        //epoch
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.batchGradient(desiredOutput[i]);
        }
        finalNode.updateWeightBatch();
    }
    
    public void deltaRule(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(0);
        //epoch
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            finalNode.updateWeight(desiredOutput[i]);
        }
    }
    
    public void backPropagation(double[][] input,double[] desiredOutput){
        finalNode.setActivationFunction(2);
        //epoch
        for(int i=0;i<input.length;i++){
            //iterasi
            for(int j=0;j<input[i].length;j++){
                startNode[j].setInput(input[i][j]);
            }
            
            finalNode.updateWeight(desiredOutput[i]);
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

    public void buildClassifier(Instances i) throws Exception {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
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
