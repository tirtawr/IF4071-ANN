package com.mycompany.tubesann;

import weka.classifiers.Classifier;

/**
 * Hello world!
 *
 */
public class App 
{
    public static void main( String[] args )
    {
        MyANN myann = new MyANN();
        InputNode[] input= new InputNode[3];
        
        for(int i=0;i<input.length;i++){
            input[i] = new InputNode();
            input[i].setActivationFunction(1);
        }
        
        
        double[] weight = {0.5,0.2,0.8};
        
        Node finalNode = new Node();
        finalNode.setActivationFunction(1);
        finalNode.setPrev(input);
        finalNode.setPrevWeight(weight);
        MyANN.LEARNINGRATE = 1;
        
        myann.setFinalNode(finalNode);
        myann.setStartNode(input);
        
    }
}
