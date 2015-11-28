/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.tubesann;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Queue;

/**
 *
 * @author Riady
 */
public class Node {
    private int id;
    private Node[] next;
    private Node[] prev;
    protected double output;
    private double[] nextWeight;
    private HashMap<Integer,Double> prevWeight;
    private HashMap<Integer,Double> deltaPrevWeight;
    private HashMap<Integer,Double> prevDeltaPrevWeight;
    private HashMap<Integer,Double> prevDeltaPrevWeightEpoch;
    private static Queue<Node> queue;

    public void setPrevWeight(HashMap<Integer, Double> prevWeight) {
        this.prevWeight = prevWeight;
        deltaPrevWeight = new HashMap<Integer,Double>();
        for ( Integer key : prevWeight.keySet() ) {
            deltaPrevWeight.put(key, new Double(0));
        }
        prevDeltaPrevWeight = new HashMap<Integer,Double>();
        for ( Integer key : prevWeight.keySet() ) {
            prevDeltaPrevWeight.put(key, new Double(0));
        }
    }
    private double error;
    
    //0 untuk tidak ada, 1 untuk sign, 2 untuk sigmoid
    private int activationFunction = 0;

    public int getId() {
        return id;
    }

    
    
    public void setActivationFunction(int activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double getOutput() {
        return output;
    }
    
    public Node(){
        id=0;
    }

    public Node(int id) {
        this.id = id;
    }
    
    
    
    public void setNext(Node[] next) {
        this.next = next;
    }

    public void setPrev(Node[] prev) {
        this.prev = prev;
        for(int i=0;i<prev.length;i++){
            Node[] nextNode = new Node[1];
            nextNode[0] = this;
            prev[i].setNext(nextNode);
            
        }
    }


    public double calculate(){
        double ret = 0;
        
        //jumlahin semua prev beserta weightnya
        for(int i=0;i<prev.length;i++){
            ret+=prev[i].calculate() * prevWeight.get(prev[i].id);
        }
        //dikenakan activasion function
        if(activationFunction == 1){
            ret = ActivationFunction.signFunction(ret);
        }
        else if(activationFunction == 2){
            ret = ActivationFunction.sigmoidFunction(ret);
        }
        output = ret;
        return ret;
    }
    
    public void updateWeight(double desiredOutput){
        calculate();
        for(int i=0;i<prev.length;i++){ 
            System.out.println(prevWeight.get(prev[i].id));
            deltaPrevWeight.put(prev[i].id,(MyANN.LEARNINGRATE*(desiredOutput - output)*prev[i].output) + MyANN.MOMENTUM*deltaPrevWeight.get(prev[i].id));
            prevWeight.put(prev[i].id,prevWeight.get(prev[i].id)+deltaPrevWeight.get(prev[i].id));
            System.out.println(prevWeight.get(prev[i].id));
            System.out.println("");
        }
        
    }
    
    public void batchGradient(double desiredOutput){
        calculate();
        for(int i=0;i<prev.length;i++){ 
            deltaPrevWeight.put(prev[i].id,deltaPrevWeight.get(prev[i].id)+MyANN.LEARNINGRATE*(desiredOutput - output)*prev[i].output);    
        }
    }
    
    public void updateWeightBatch(){
        for(int i=0;i<prev.length;i++){ 
            System.out.println(prevWeight.get(prev[i].id));
            
            prevWeight.put(prev[i].id, prevWeight.get(prev[i].id) + deltaPrevWeight.get(prev[i].id)+prevDeltaPrevWeight.get(prev[i].id)*MyANN.MOMENTUM);
            prevDeltaPrevWeight.put(prev[i].id,+ deltaPrevWeight.get(prev[i].id));
            deltaPrevWeight.put(prev[i].id, new Double(0));
            System.out.println(prevWeight.get(prev[i].id));
            System.out.println("");
        }
    }
    
    public void updateWeightBackPropFinalNode(double desiredOutput){
        calculate();
        queue = new LinkedList<Node>();
        error = MyANN.LEARNINGRATE*output*(1-output)*(desiredOutput-output);
        
        for(int i=0;i<prev.length;i++){
            System.out.println("id="+prev[i].id+"  "+prevWeight.get(prev[i].id));
            prevWeight.put(prev[i].id,prevWeight.get(prev[i].id)+error*prev[i].output);
            System.out.println(prevWeight.get(prev[i].id));
            System.out.println();
            queue.add(prev[i]);
        }
        while(!queue.isEmpty()){
            queue.remove().updateWeightBackProp();
        }
    }
    
    public void updateWeightBackProp(){
        double tempError = 0;
        for(int i=0;i<next.length;i++){
            tempError+=next[i].error*next[i].prevWeight.get(id);
        }
        error = MyANN.LEARNINGRATE*output * (1-output) * tempError;
        if(prev!=null){
            for(int i=0;i<prev.length;i++){
                prevWeight.put(prev[i].id,prevWeight.get(prev[i].id)+error*prev[i].output);
                queue.add(prev[i]);
            }
        }
        
    }
}
