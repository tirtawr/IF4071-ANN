package com.mycompany.tubesann;

import java.util.HashMap;
import weka.classifiers.Classifier;

/**
 * Hello world!
 *
 */
public class App 
{
    @SuppressWarnings("empty-statement")
    public static void main( String[] args )
    {
        /*MyANN myann = new MyANN();
        InputNode[] input= new InputNode[5];
        
        for(int i=0;i<input.length;i++){
            input[i] = new InputNode(i);
            input[i].setActivationFunction(1);
        }
        
        
        double[] weight = {0,0,0,0,0};
        HashMap<Integer,Double> temp = new HashMap<Integer,Double>();
        temp.put(0, new Double(0));
        temp.put(1, new Double(0));
        temp.put(2, new Double(0));
        temp.put(3, new Double(0));
        temp.put(4, new Double(0));
        
        Node finalNode = new Node(5);
        finalNode.setActivationFunction(1);
        finalNode.setPrev(input);
        finalNode.setPrevWeight(temp);
        MyANN.LEARNINGRATE = 0.1;
        
        myann.setFinalNode(finalNode);
        myann.setStartNode(input);
        
        double[][] testInput = {{1,5.1,3.5,1.4,0.2},{1,4.9,3,1.4,0.2},{1,4.7,3.2,1.3,0.2},{1,7,3.2,4.7,1.4},{1,6.4,3.2,4.5,1.5},{1,6.9,3.1,4.9,1.5}};
        double[] testDesiredOutput = {1,1,1,-1,-1,-1};
        
        myann.deltaRule(testInput, testDesiredOutput);
        myann.deltaRule(testInput, testDesiredOutput);
        */
        MyANN myann = new MyANN();
        InputNode[] input= new InputNode[5];
        
        int inc = 0;
        for(int i=0;i<input.length;i++){
            input[i] = new InputNode(inc);
            inc++;
            input[i].setActivationFunction(2);
        }
        
        
        double[] weight = {0,0,0,0,0};
        HashMap<Integer,Double> temp = new HashMap<Integer,Double>();
        temp.put(0, new Double(0));
        temp.put(1, new Double(0));
        temp.put(2, new Double(0));
        temp.put(3, new Double(0));
        temp.put(4, new Double(0));
        
        Node nodeh1 = new Node(inc);
        inc++;
        Node nodeh2 = new Node(inc);
        inc++;
        
        nodeh1.setPrev(input);
        nodeh1.setActivationFunction(2);
        nodeh1.setPrevWeight(temp);
        nodeh2.setPrev(input);
        nodeh2.setActivationFunction(2);
        nodeh2.setPrevWeight(temp);
        
        Node[] hiddenNode = new Node[2];
        hiddenNode[0] = nodeh1;
        hiddenNode[1] = nodeh2;
        
        HashMap<Integer,Double> tempp = new HashMap<Integer,Double>();
        tempp.put(nodeh1.getId(), new Double(0));
        tempp.put(nodeh2.getId(), new Double(0));
        
        Node finalNode = new Node(inc);
        finalNode.setActivationFunction(2);
        finalNode.setPrev(hiddenNode);
        finalNode.setPrevWeight(tempp);
        MyANN.LEARNINGRATE = 0.1;
        
        myann.setFinalNode(finalNode);
        myann.setStartNode(input);
        
        double[][] testInput = {{1,5.1,3.5,1.4,0.2},{1,4.9,3,1.4,0.2},{1,4.7,3.2,1.3,0.2},{1,7,3.2,4.7,1.4},{1,6.4,3.2,4.5,1.5},{1,6.9,3.1,4.9,1.5}};
        double[] testDesiredOutput = {1,1,1,-1,-1,-1};
        
        myann.backPropagation(testInput, testDesiredOutput);
        
     }
}
