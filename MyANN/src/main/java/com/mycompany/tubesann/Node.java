/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package com.mycompany.tubesann;

/**
 *
 * @author Riady
 */
public class Node {
    private Node[] next;
    private Node[] prev;
    protected double output;
    private double[] prevWeight;
    private double[] deltaPrevWeight;
    
    
    //0 untuk tidak ada, 1 untuk sign, 2 untuk sigmoid
    private int activationFunction = 0;

    public void setActivationFunction(int activationFunction) {
        this.activationFunction = activationFunction;
    }

    public double getOutput() {
        return output;
    }
    
    public Node(){
    
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

    public void setPrevWeight(double[] prevWeight) {
        this.prevWeight = prevWeight;
        deltaPrevWeight = new double[prevWeight.length];
        for(int i= 0;i<deltaPrevWeight.length;i++){
            deltaPrevWeight[i] = 0;
        }
    }

    public double calculate(){
        double ret = 0;
        
        //jumlahin semua prev beserta weightnya
        for(int i=0;i<prev.length;i++){
            ret+=prev[i].calculate() * prevWeight[i];
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
            prevWeight[i] += MyANN.LEARNINGRATE*(desiredOutput - output)*prev[i].output;
        }
        
    }
    
    public void batchGradient(double desiredOutput){
        calculate();
        for(int i=0;i<prev.length;i++){ 
            deltaPrevWeight[i] += MyANN.LEARNINGRATE*(desiredOutput - output)*prev[i].output;
        }
    }
    
    public void updateWeightBatch(){
        for(int i=0;i<prev.length;i++){ 
            prevWeight[i] += deltaPrevWeight[i];
            deltaPrevWeight[i] = 0;
        }
    }
}
